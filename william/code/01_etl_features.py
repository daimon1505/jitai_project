"""
Milestone 1 - Data ETL + feature extraction for the RELAX dataset.

For every momentary EMA response (questionnaire "ifb") we cut a window of
wearable data that ends at the response timestamp, clean the signals, and
compute HRV + accelerometer features. Output: one row per EMA response.

Usage:  [WINDOW_MIN=10] python 01_etl_features.py <RELAXDataset dir> <output dir> [n_workers]
"""
import sys
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import lombscargle

WINDOW_MIN = int(os.environ.get("WINDOW_MIN", 10))  # minutes before each EMA
MIN_CLEAN_BEATS = 60       # windows with fewer clean beats get NaN HRV features
MIN_ACC_COVERAGE = 0.2     # fraction of expected ACC samples required
ACC_HZ = 52.0
IBI_MIN, IBI_MAX = 300, 2000   # ms, physiologically plausible (30-200 bpm)
ERR_MAX = 50                   # ms, max device error estimate accepted
SUCC_DEV = 0.20                # reject beats deviating >20% from local median


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_ibi(df):
    """Return cleaned IBI frame + per-step counts (for the cleaning report)."""
    stats = {"ibi_raw": len(df)}
    df = df[~df["ibi_blocker"]]
    stats["ibi_after_blocker"] = len(df)
    df = df[(df["ibi_ppi"] >= IBI_MIN) & (df["ibi_ppi"] <= IBI_MAX)]
    stats["ibi_after_range"] = len(df)
    df = df[df["ibi_errorEstimate"] <= ERR_MAX]
    stats["ibi_after_error"] = len(df)
    # ectopic / artefact rejection: compare to rolling median of 5 beats
    med = df["ibi_ppi"].rolling(5, center=True, min_periods=1).median()
    df = df[(df["ibi_ppi"] - med).abs() <= SUCC_DEV * med]
    stats["ibi_clean"] = len(df)
    return df.reset_index(drop=True), stats


# --------------------------------------------------------------------------
# Feature functions
# --------------------------------------------------------------------------
def hrv_features(ts_s, rr):
    """ts_s: beat times in seconds, rr: RR intervals in ms (cleaned)."""
    out = {"n_beats": len(rr)}
    keys = ["mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr", "sd_hr",
            "cv_rr", "lf", "hf", "lf_hf"]
    if len(rr) < MIN_CLEAN_BEATS:
        out.update({k: np.nan for k in keys})
        return out
    # successive differences only between truly consecutive beats
    # (a gap left by removed beats would create a fake large difference)
    d_t = np.diff(ts_s)
    d_rr = np.diff(rr)
    consecutive = np.abs(d_t * 1000 - rr[1:]) < 0.25 * rr[1:]
    sd = d_rr[consecutive]
    hr = 60000.0 / rr
    out["mean_rr"] = rr.mean()
    out["sdnn"] = rr.std(ddof=1)
    out["rmssd"] = np.sqrt(np.mean(sd ** 2)) if len(sd) > 10 else np.nan
    out["pnn50"] = np.mean(np.abs(sd) > 50) if len(sd) > 10 else np.nan
    out["mean_hr"] = hr.mean()
    out["sd_hr"] = hr.std(ddof=1)
    out["cv_rr"] = out["sdnn"] / out["mean_rr"]
    # frequency domain: Lomb-Scargle handles uneven sampling + gaps
    f = np.linspace(0.04, 0.40, 145)
    x = (rr - rr.mean())
    p = lombscargle(ts_s - ts_s[0], x, 2 * np.pi * f, normalize=False)
    lf = np.trapezoid(p[f < 0.15], f[f < 0.15])
    hf = np.trapezoid(p[f >= 0.15], f[f >= 0.15])
    out["lf"], out["hf"] = np.log1p(lf), np.log1p(hf)
    out["lf_hf"] = np.log((lf + 1e-9) / (hf + 1e-9))
    return out


def acc_features(xyz):
    """xyz: (n,3) raw accelerometer in mg."""
    keys = ["acc_mag_mean", "acc_mag_std", "acc_enmo", "acc_active_frac",
            "acc_axis_std"]
    expected = WINDOW_MIN * 60 * ACC_HZ
    cov = len(xyz) / expected
    out = {"acc_coverage": min(cov, 1.0)}
    if cov < MIN_ACC_COVERAGE:
        out.update({k: np.nan for k in keys})
        return out
    mag = np.sqrt((xyz ** 2).sum(1))
    enmo = np.clip(mag - 1000.0, 0, None)          # Euclidean norm minus 1 g
    # 1-second epochs -> fraction of seconds with movement
    n = int(len(mag) // ACC_HZ) * int(ACC_HZ)
    if n >= ACC_HZ:
        ep = mag[:n].reshape(-1, int(ACC_HZ)).std(1)
        active = np.mean(ep > 50)                   # >50 mg wobble in a second
    else:
        active = np.nan
    out.update({
        "acc_mag_mean": mag.mean(),
        "acc_mag_std": mag.std(),
        "acc_enmo": enmo.mean(),
        "acc_active_frac": active,
        "acc_axis_std": xyz.std(0).mean(),
    })
    return out


# --------------------------------------------------------------------------
# Per-subject processing
# --------------------------------------------------------------------------
def process_subject(args):
    data_dir, uid, ema = args
    sdir = os.path.join(data_dir, "data", str(uid))
    ends = ema["t_ns"].to_numpy()
    starts = ends - np.int64(WINDOW_MIN * 60 * 1e9)

    # ---- IBI (small: load whole file) ----
    ibi = pq.read_table(os.path.join(sdir, "ibi_data.parquet")).to_pandas()
    ibi = ibi.sort_values("timestamp")
    ibi_clean, cstats = clean_ibi(ibi)
    cstats["user"] = uid
    cstats["ibi_hours_raw"] = ibi["ibi_ppi"].sum() / 3.6e6
    t_ibi = ibi_clean["timestamp"].astype("int64").to_numpy()
    rr = ibi_clean["ibi_ppi"].to_numpy().astype(float)

    rows = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        a, b = np.searchsorted(t_ibi, [s, e])
        f = hrv_features(t_ibi[a:b] / 1e9, rr[a:b])
        f["ema_idx"] = ema.index[i]
        rows.append(f)
    feats = pd.DataFrame(rows).set_index("ema_idx")

    # ---- ACC (large: stream row groups, keep only window slices) ----
    pf = pq.ParquetFile(os.path.join(sdir, "acc_data.parquet"))
    chunks = {i: [] for i in range(len(ends))}
    n_acc = 0
    for g in range(pf.num_row_groups):
        tb = pf.read_row_group(g)
        t = tb.column("timestamp").cast("int64").to_numpy()
        n_acc += len(t)
        if not np.all(t[1:] >= t[:-1]):
            o = np.argsort(t, kind="stable")
        else:
            o = None
        xyz = np.column_stack([tb.column(c).to_numpy() for c in
                               ("acc_x", "acc_y", "acc_z")])
        if o is not None:
            t, xyz = t[o], xyz[o]
        del tb
        lo, hi = np.searchsorted(t, starts), np.searchsorted(t, ends)
        for i in np.nonzero(hi > lo)[0]:
            chunks[i].append(xyz[lo[i]:hi[i]].copy())
        del t, xyz
    cstats["acc_samples"] = n_acc
    acc_rows = []
    for i in range(len(ends)):
        xyz = np.vstack(chunks[i]) if chunks[i] else np.empty((0, 3))
        f = acc_features(xyz)
        f["ema_idx"] = ema.index[i]
        acc_rows.append(f)
    feats = feats.join(pd.DataFrame(acc_rows).set_index("ema_idx"))
    print(f"subject {uid}: {len(ema)} EMAs done", flush=True)
    return ema.join(feats), cstats


def main():
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(out_dir, exist_ok=True)
    xl = os.path.join(data_dir, "questionnaire_responses.xlsx")

    ifb = pd.read_excel(xl, sheet_name="ifb")
    users = pd.read_excel(xl, sheet_name="users")
    # "answers/ifb-10-1" -> "ifb_10"
    ema = ifb.rename(columns=lambda c: "ifb_" + c.split("-")[1]
                     if c.startswith("answers/") else c)
    # manual_date is epoch-ms (UTC) -> same clock as the sensor data
    ema["t_utc"] = pd.to_datetime(ema["manual_date"], unit="ms", utc=True)
    ema["t_ns"] = ema["t_utc"].astype("int64")
    n0 = len(ema)
    ema = ema.drop_duplicates(subset=["user", "manual_date"])
    print(f"EMA rows: {n0}, after duplicate removal: {len(ema)}")

    have = {int(d) for d in os.listdir(os.path.join(data_dir, "data"))}
    ema = ema[ema["user"].isin(have)]
    jobs = [(data_dir, uid, g) for uid, g in ema.groupby("user")]
    with Pool(n_workers) as pool:
        res = pool.map(process_subject, jobs)

    df = pd.concat([r[0] for r in res]).sort_values(["user", "t_utc"])
    cs = pd.DataFrame([r[1] for r in res]).sort_values("user")
    meta = users.rename(columns={"metadata_json/birth_year_group": "birth_group",
                                 "metadata_json/gender": "gender",
                                 "metadata_json/height": "height",
                                 "metadata_json/weight": "weight"})
    df = df.merge(meta[["id", "birth_group", "gender"]], left_on="user",
                  right_on="id", how="left").drop(columns="id")
    df.drop(columns=["t_ns"]).to_parquet(
        os.path.join(out_dir, f"ema_windows_features_{WINDOW_MIN}min.parquet"))
    cs.to_csv(os.path.join(out_dir, "cleaning_report.csv"), index=False)
    print(cs.to_string())


if __name__ == "__main__":
    main()
