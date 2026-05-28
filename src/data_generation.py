from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 7
    freq_minutes: int = 5
    start_date: str = "2026-01-01"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_wearable_timeseries(
    *,
    n_users: int = 200,
    days_per_user: int = 10,
    cfg: SyntheticConfig = SyntheticConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      ts_df: time-series rows with columns
        [user_id, timestamp, heart_rate, steps, activity_intensity, sleep_stage]
      labels_df: daily labels with columns
        [user_id, date, sleep_quality]
    """
    rng = np.random.default_rng(cfg.seed)

    freq = f"{cfg.freq_minutes}min"
    start = pd.Timestamp(cfg.start_date).tz_localize(None)
    periods_per_day = int((24 * 60) / cfg.freq_minutes)

    rows: list[dict] = []
    label_rows: list[dict] = []

    for u in range(n_users):
        # User baseline physiology + behavior
        resting_hr = rng.normal(62, 6)
        fitness = rng.normal(0.0, 1.0)  # higher fitness => lower HR, higher activity tolerance
        chronotype = rng.uniform(-1.0, 1.0)  # -1 early, +1 late

        for d in range(days_per_user):
            day_start = start + pd.Timedelta(days=d)
            idx = pd.date_range(day_start, day_start + pd.Timedelta(days=1), freq=freq, inclusive="left")

            # Create a sleep window affected by chronotype + noise
            sleep_onset_hour = 22.5 + 1.3 * chronotype + rng.normal(0, 0.4)
            sleep_duration_hours = np.clip(rng.normal(7.2, 0.9), 4.5, 9.5)
            sleep_offset_hour = sleep_onset_hour + sleep_duration_hours

            # Stress / recovery drivers influencing label and signals
            late_caffeine = rng.binomial(1, 0.25)
            alcohol = rng.binomial(1, 0.2)
            high_training = rng.binomial(1, 0.3)
            screen_late = rng.binomial(1, 0.35)

            # Day activity pattern (workday vs weekend-ish)
            is_weekend = (day_start.dayofweek >= 5)
            base_steps = rng.normal(7500 if not is_weekend else 9500, 2200)
            base_steps = float(np.clip(base_steps + 900 * fitness, 1500, 18000))

            # Latent sleep quality score; convert to probability for label
            # Higher: better. Negative: worse.
            score = (
                +0.65 * (sleep_duration_hours - 7.0)
                +0.30 * fitness
                -0.55 * late_caffeine
                -0.60 * alcohol
                -0.35 * screen_late
                -0.25 * high_training
                -0.10 * max(0.0, (sleep_offset_hour - 8.5))  # too late wake time slightly worse
                +rng.normal(0, 0.6)
            )
            p_high = float(_sigmoid(np.array([score]))[0])
            sleep_quality = int(rng.random() < p_high)

            label_rows.append(
                {"user_id": u, "date": day_start.normalize(), "sleep_quality": sleep_quality}
            )

            # Generate time-series signals with circadian rhythm
            for ts in idx:
                hour = ts.hour + ts.minute / 60.0

                # Circadian component: lower HR at night, higher daytime
                circ = math.sin(2 * math.pi * (hour - 4.0) / 24.0)
                circ2 = math.sin(4 * math.pi * (hour - 3.0) / 24.0)

                # Determine asleep based on sleep window (wrap over midnight)
                # Sleep window may cross midnight. We'll treat it in [0, 24) using modular.
                def in_sleep_window(h: float) -> bool:
                    onset = sleep_onset_hour % 24.0
                    offset = sleep_offset_hour % 24.0
                    if sleep_duration_hours >= 24:
                        return True
                    if onset < offset:
                        return onset <= h < offset
                    return (h >= onset) or (h < offset)

                asleep = in_sleep_window(hour)

                # Activity intensity: mostly daytime; some nighttime noise
                if asleep:
                    intensity = max(0.0, rng.normal(0.05, 0.03))
                else:
                    # Peaks morning/evening; fitness affects ability to be active
                    peak = 0.8 * max(0.0, math.sin(2 * math.pi * (hour - 7.5) / 24.0)) + 0.5 * max(
                        0.0, math.sin(2 * math.pi * (hour - 17.5) / 24.0)
                    )
                    intensity = np.clip(rng.normal(0.25 + 0.55 * peak + 0.08 * fitness, 0.12), 0, 1.5)

                # Steps per interval derived from intensity and daily goal
                # Allocate more steps during waking hours.
                waking_factor = 0.15 + 0.85 * (0.5 + 0.5 * math.sin(2 * math.pi * (hour - 8.0) / 24.0))
                steps = 0.0 if asleep else rng.poisson(max(0.1, (base_steps / periods_per_day) * waking_factor * (0.6 + 1.2 * intensity)))

                # Heart rate depends on resting HR, activity, stressors, and sleep quality
                stress_boost = 0.0
                if late_caffeine and (hour >= 15.0 and hour <= 23.5):
                    stress_boost += 3.0
                if alcohol and asleep:
                    stress_boost += 2.0
                if screen_late and (hour >= 21.0 or hour <= 1.0):
                    stress_boost += 2.0

                # If low-quality sleep, nighttime HR tends to be higher and more variable
                low_quality_penalty = (1 - sleep_quality)
                sleep_hr_offset = -4.0 if asleep else 0.0
                hr = (
                    resting_hr
                    + 10.0 * intensity
                    + 4.0 * circ
                    + 1.5 * circ2
                    + stress_boost
                    + sleep_hr_offset
                    + low_quality_penalty * (2.0 if asleep else 0.5)
                    - 2.0 * fitness
                    + rng.normal(0, 2.2 if not asleep else 1.4 + 0.8 * low_quality_penalty)
                )
                hr = float(np.clip(hr, 38, 190))

                # Sleep stage proxy (only meaningful when asleep)
                # 0=awake,1=light,2=deep,3=rem
                if not asleep:
                    stage = 0
                else:
                    # Deep more likely earlier; REM more likely later
                    # Use fraction through sleep window as proxy.
                    # Compute minutes since onset in modular time.
                    onset = sleep_onset_hour % 24.0
                    # convert to minutes from onset, modulo 24h
                    mins_since_onset = ((hour - onset) % 24.0) * 60.0
                    frac = mins_since_onset / max(1.0, sleep_duration_hours * 60.0)
                    deep_p = max(0.05, 0.40 * (1 - frac))
                    rem_p = max(0.05, 0.30 * frac)
                    light_p = max(0.05, 1.0 - deep_p - rem_p)
                    probs = np.array([0.0, light_p, deep_p, rem_p], dtype=float)
                    probs = probs / probs.sum()
                    stage = int(rng.choice(np.array([0, 1, 2, 3]), p=probs))

                rows.append(
                    {
                        "user_id": u,
                        "timestamp": ts.to_pydatetime().replace(tzinfo=None),
                        "heart_rate": hr,
                        "steps": float(steps),
                        "activity_intensity": float(intensity),
                        "sleep_stage": stage,
                    }
                )

    ts_df = pd.DataFrame(rows)
    labels_df = pd.DataFrame(label_rows)
    return ts_df, labels_df

