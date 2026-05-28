from __future__ import annotations

import numpy as np
import pandas as pd


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["date"] = ts.dt.normalize()
    out["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    out["dow"] = ts.dt.dayofweek.astype(int)

    # Circadian encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    # "Night window" indicator (typical sleep hours)
    out["is_night"] = ((out["hour"] >= 22.0) | (out["hour"] <= 6.0)).astype(int)
    return out


def _rolling_group_features(
    df: pd.DataFrame, *, group_cols: list[str], sort_col: str, window: int
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(group_cols + [sort_col])
    g = out.groupby(group_cols, sort=False)

    # Rolling HR stats as HRV proxies on wearables without RR intervals
    out["hr_roll_mean"] = g["heart_rate"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 3)).mean())
    out["hr_roll_std"] = g["heart_rate"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 3)).std())
    out["hr_roll_rmssd_proxy"] = g["heart_rate"].transform(
        lambda s: (s.diff().pow(2).rolling(window, min_periods=max(3, window // 3)).mean()).pow(0.5)
    )

    # Activity intensity/load
    out["steps_roll_sum"] = g["steps"].transform(lambda s: s.rolling(window, min_periods=max(2, window // 3)).sum())
    out["intensity_roll_mean"] = g["activity_intensity"].transform(
        lambda s: s.rolling(window, min_periods=max(2, window // 3)).mean()
    )

    # Short-term dynamics
    out["hr_delta_1"] = g["heart_rate"].transform(lambda s: s.diff(1))
    out["steps_delta_1"] = g["steps"].transform(lambda s: s.diff(1))
    out["intensity_delta_1"] = g["activity_intensity"].transform(lambda s: s.diff(1))

    return out


def aggregate_daily_features(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time-series wearable data into a daily feature table per user_id/date.
    """
    df = _add_time_features(ts_df)

    # 1 hour (12 * 5min) rolling window features, within user_id/date
    df = _rolling_group_features(df, group_cols=["user_id", "date"], sort_col="timestamp", window=12)

    # Sleep stage summaries
    df["is_sleep"] = (df["sleep_stage"] > 0).astype(int)
    df["is_deep"] = (df["sleep_stage"] == 2).astype(int)
    df["is_rem"] = (df["sleep_stage"] == 3).astype(int)

    agg = df.groupby(["user_id", "date"], as_index=False).agg(
        # Heart rate distribution
        hr_mean=("heart_rate", "mean"),
        hr_std=("heart_rate", "std"),
        hr_p10=("heart_rate", lambda s: float(np.nanpercentile(s, 10))),
        hr_p90=("heart_rate", lambda s: float(np.nanpercentile(s, 90))),
        hr_night_mean=("heart_rate", lambda s: float(np.nanmean(s[df.loc[s.index, "is_night"] == 1]))),
        hr_night_std=("heart_rate", lambda s: float(np.nanstd(s[df.loc[s.index, "is_night"] == 1]))),
        # Rolling HRV proxies
        hrv_rmssd_proxy_mean=("hr_roll_rmssd_proxy", "mean"),
        hrv_rmssd_proxy_night=("hr_roll_rmssd_proxy", lambda s: float(np.nanmean(s[df.loc[s.index, "is_night"] == 1]))),
        hr_roll_std_mean=("hr_roll_std", "mean"),
        # Activity
        steps_total=("steps", "sum"),
        steps_night=("steps", lambda s: float(np.nansum(s[df.loc[s.index, "is_night"] == 1]))),
        intensity_mean=("activity_intensity", "mean"),
        intensity_p90=("activity_intensity", lambda s: float(np.nanpercentile(s, 90))),
        intensity_evening=("activity_intensity", lambda s: float(np.nanmean(s[(df.loc[s.index, "hour"] >= 18) & (df.loc[s.index, "hour"] <= 23)]))),
        # Circadian / schedule proxies
        night_fraction=("is_night", "mean"),
        dow=("dow", "first"),
        # Sleep composition proxies
        sleep_fraction=("is_sleep", "mean"),
        deep_fraction=("is_deep", "mean"),
        rem_fraction=("is_rem", "mean"),
    )

    # Derived features
    agg["steps_per_intensity"] = agg["steps_total"] / (1e-6 + agg["intensity_mean"])
    agg["restlessness_proxy"] = agg["steps_night"] / (1e-6 + agg["sleep_fraction"])
    agg["hr_night_minus_day"] = agg["hr_night_mean"] - agg["hr_mean"]
    agg["circadian_misalignment_proxy"] = np.abs(agg["intensity_evening"] - agg["intensity_mean"])

    # Clean up infinities/nans
    for c in agg.columns:
        if c in {"user_id", "date"}:
            continue
        agg[c] = pd.to_numeric(agg[c], errors="coerce")
        agg[c] = agg[c].replace([np.inf, -np.inf], np.nan)

    return agg

