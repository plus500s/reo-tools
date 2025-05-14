import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from reo_tools.progress import progress_bar
from reo_tools.settings import PipelineSettings
from reo_tools.utils import (
    align_zero_crossings,
    apply_median_filter,
    average_segment,
    butter_filter,
    calculate_phases,
    compute_fft,
    compute_periods,
    define_periods_for_resampling_by_two_segments,
    detrend_signal,
    estimate_pulse,
    extend_signal_as_periods,
    extract_first_harmonic,
    filter_valid_periods,
    find_min_max_distance_from_curve_to_line,
    find_peaks_around_points,
    find_zeros,
    get_smoothed_derivative,
    normalize_segments,
    period_to_samples,
    rescale_set_min_preserve_mean,
    rescale_signal,
    slice_signal_by_segments,
    smooth_original_signal,
)


class SignalPipeline:
    def __init__(self, cfg: PipelineSettings) -> None:
        self.cfg = cfg

    def run(self, file_path: str | Path, scope: str | None = None) -> Mapping[str, pd.DataFrame]:
        if not self.cfg.recompute:
            cached = self.cfg.cache_backend.load(scope)
            if cached:
                return cached

        rvg_data = self.cfg.reader.read(file_path)
        result = self._process(rvg_data)

        self.cfg.cache_backend.save(scope, result)
        return result

    @progress_bar(enabled=lambda self: self.cfg.show_progress)
    def _process(self, data: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
        DEBUG = self.cfg.debug

        pulse_data = estimate_pulse(data)
        SAMPLING_PERIOD = pulse_data["sampling_period"]
        FREQUENCY = pulse_data["frequency"]
        PULSE = pulse_data["pulse"]

        if PULSE == 0:
            raise ValueError("Detected zero pulse")

        if DEBUG:
            print(f"Heart rate detected: {PULSE:.1f} BPM ({FREQUENCY:.3f} Hz)", flush=True)

        # 0.3 Smooth signal with floating window
        window_size = len(data["U1"]) // len(pulse_data["peaks"])
        data["U1 clean"] = smooth_original_signal(data["U1"], window_size)["cleaned_signal"]
        data["U2 clean"] = smooth_original_signal(data["U2"], window_size)["cleaned_signal"]

        # 1.0 Extract 1st harmonic
        data["U1-1"] = extract_first_harmonic(
            data["U1 clean"], FREQUENCY, SAMPLING_PERIOD, butter_order=1, num_iterations=5
        )
        data["U2-1"] = extract_first_harmonic(
            data["U2 clean"], FREQUENCY, SAMPLING_PERIOD, butter_order=1, num_iterations=5
        )

        # 2.0 Find points where 1st harmonic crosses zeros
        u1_1_zeros, u2_1_zeros = align_zero_crossings(data["U1-1"], data["U2-1"])

        data["U1-1 zeros"] = u1_1_zeros
        data["U2-1 zeros"] = u2_1_zeros

        # 3.0 Extract high frequency part
        low_cutoff = 0.5  # Hz
        high_cutoff = 40  # Hz
        filter_order = 3

        data["U1 clean*"] = butter_filter(data["U1 clean"], low_cutoff, "high", SAMPLING_PERIOD, order=filter_order)
        data["U1 clean*"] = butter_filter(data["U1 clean*"], high_cutoff, "low", SAMPLING_PERIOD, order=filter_order)

        data["U2 clean*"] = butter_filter(data["U2 clean"], low_cutoff, "high", SAMPLING_PERIOD, order=filter_order)
        data["U2 clean*"] = butter_filter(data["U2 clean*"], high_cutoff, "low", SAMPLING_PERIOD, order=filter_order)

        data["U1-2"] = data["U1 clean*"] - data["U1-1"]
        data["U2-2"] = data["U2 clean*"] - data["U2-1"]

        # 4.0 Compute derivative of high frequency part of signal and apply smoothing window
        data["U1-3"] = get_smoothed_derivative(data["U1-2"], window=5)
        data["U2-3"] = get_smoothed_derivative(data["U2-2"], window=5)

        # 5.0 Find peaks of derivative
        delta1 = 0.01  # 10 ms
        delta2 = 0.03  # 30 ms

        u1_3_peaks = find_peaks_around_points(
            data["U1-3"],
            u1_1_zeros,
            period_to_samples(delta1, SAMPLING_PERIOD),
            period_to_samples(delta2, SAMPLING_PERIOD),
        )
        data["t"].loc[u1_3_peaks["peak_idx"]]

        u2_3_peaks = find_peaks_around_points(
            data["U2-3"],
            u2_1_zeros,
            period_to_samples(delta1, SAMPLING_PERIOD),
            period_to_samples(delta2, SAMPLING_PERIOD),
        )
        data["t"].loc[u2_3_peaks["peak_idx"]]

        data["U1-3 peaks"] = u1_3_peaks.set_index("peak_idx")["peak_value"]
        data["U2-3 peaks"] = u2_3_peaks.set_index("peak_idx")["peak_value"]

        # 11.0 Find phase shift (prepare peaks and throws)

        u1_1_peaks = find_peaks_around_points(
            data["U1-1"],
            u1_1_zeros,
            period_to_samples(0.002, SAMPLING_PERIOD),
            period_to_samples(0.350, SAMPLING_PERIOD),
        )
        u1_1_throws = find_peaks_around_points(
            -data["U1-1"],
            u1_1_zeros,
            period_to_samples(0.350, SAMPLING_PERIOD),
            period_to_samples(0.002, SAMPLING_PERIOD),
        )

        u2_1_peaks = find_peaks_around_points(
            data["U2-1"],
            u2_1_zeros,
            period_to_samples(0.002, SAMPLING_PERIOD),
            period_to_samples(0.350, SAMPLING_PERIOD),
        )
        u2_1_throws = find_peaks_around_points(
            -data["U2-1"],
            u2_1_zeros,
            period_to_samples(0.350, SAMPLING_PERIOD),
            period_to_samples(0.002, SAMPLING_PERIOD),
        )

        u1_3_peaks_series = pd.Series(data=u1_3_peaks["peak_value"].values, index=u1_3_peaks["peak_idx"].values)
        u1_2_peaks = find_peaks_around_points(
            data["U1-2"],
            u1_3_peaks_series,
            period_to_samples(0.001, SAMPLING_PERIOD),
            period_to_samples(0.040, SAMPLING_PERIOD),
        )
        u1_2_throws = find_peaks_around_points(
            -data["U1-2"],
            u1_3_peaks_series,
            period_to_samples(0.040, SAMPLING_PERIOD),
            period_to_samples(0.001, SAMPLING_PERIOD),
        )

        u2_3_peaks_series = pd.Series(data=u2_3_peaks["peak_value"].values, index=u2_3_peaks["peak_idx"].values)
        u2_2_peaks = find_peaks_around_points(
            data["U2-2"],
            u2_3_peaks_series,
            period_to_samples(0.001, SAMPLING_PERIOD),
            period_to_samples(0.040, SAMPLING_PERIOD),
        )
        u2_2_throws = find_peaks_around_points(
            -data["U2-2"],
            u2_3_peaks_series,
            period_to_samples(0.040, SAMPLING_PERIOD),
            period_to_samples(0.001, SAMPLING_PERIOD),
        )

        # 14.1 Find peaks and throws for high frequency part
        data["U1-2 peaks"] = u1_2_peaks.set_index("peak_idx")["peak_value"]
        data["U1-2 throws"] = -u1_2_throws.set_index("peak_idx")["peak_value"]

        data["U2-2 peaks"] = u2_2_peaks.set_index("peak_idx")["peak_value"]
        data["U2-2 throws"] = -u2_2_throws.set_index("peak_idx")["peak_value"]

        # 12.a Find new HF extrema
        u1_2_throws_series = pd.Series(data=-u1_2_throws["peak_value"].values, index=u1_2_throws["peak_idx"].values)
        u1_2_peaks_series = pd.Series(data=u1_2_peaks["peak_value"].values, index=u1_2_peaks["peak_idx"].values)

        u1_2_extrema = find_min_max_distance_from_curve_to_line(
            u1_2_throws_series, u1_2_peaks_series, u1_1_zeros, data["U1-2"]
        )

        data["U1-2 throws *"] = pd.Series(
            data=u1_2_extrema["throw_value"].values, index=u1_2_extrema["throw_idx"].values
        )
        data["U1-2 peaks *"] = pd.Series(data=u1_2_extrema["peak_value"].values, index=u1_2_extrema["peak_idx"].values)

        u2_2_throws_series = pd.Series(data=-u2_2_throws["peak_value"].values, index=u2_2_throws["peak_idx"].values)
        u2_2_peaks_series = pd.Series(data=u2_2_peaks["peak_value"].values, index=u2_2_peaks["peak_idx"].values)

        u2_2_extrema = find_min_max_distance_from_curve_to_line(
            u2_2_throws_series, u2_2_peaks_series, u2_1_zeros, data["U2-2"]
        )

        data["U2-2 throws *"] = pd.Series(
            data=u2_2_extrema["throw_value"].values, index=u2_2_extrema["throw_idx"].values
        )
        data["U2-2 peaks *"] = pd.Series(data=u2_2_extrema["peak_value"].values, index=u2_2_extrema["peak_idx"].values)

        # 5.0 compute periods

        u1_2_throw_new_times = data["t"].loc[u2_2_extrema["throw_idx"]]

        periods1 = pd.DataFrame({"t": u1_2_throw_new_times[1:]})
        periods1["T1*"] = compute_periods(u1_2_throw_new_times)

        # 6.0 Apply median filter to periods
        periods1["T1**"] = apply_median_filter(periods1["T1*"])

        # 7.0 Find mean and sigma for periods
        T1_mean = periods1["T1**"].mean()
        T1_sigma = periods1["T1**"].std()
        T1_sigma_before_filter = periods1["T1*"].std()

        periods1["T1** mean"] = T1_mean
        periods1["+ 2 sigma1"] = T1_mean + 2 * T1_sigma
        periods1["- 2 sigma1"] = T1_mean - 2 * T1_sigma
        periods1["+ 2 sigma1 (before filter)"] = T1_mean + 2 * T1_sigma_before_filter
        periods1["- 2 sigma1 (before filter)"] = T1_mean - 2 * T1_sigma_before_filter

        # 8.0 Find consecutive period series
        periods1["T1** valid"] = filter_valid_periods(periods1["T1*"], T1_mean, T1_sigma)

        valid_periods = periods1["T1** valid"].reset_index(drop=True)
        t = periods1["t"].reset_index(drop=True)

        # 12.0 Find average peak to peak value for first harmonic

        first_harmonic_amplitudes = pd.DataFrame({"valid period": valid_periods})

        first_harmonic_amplitudes["p2p 1"] = u1_1_peaks["peak_value"] + u1_1_throws["peak_value"]
        first_harmonic_amplitudes["p2p 2"] = u2_1_peaks["peak_value"] + u2_1_throws["peak_value"]
        first_harmonic_amplitudes["idx"] = range(len(first_harmonic_amplitudes))

        invalid = first_harmonic_amplitudes["valid period"].isna()
        first_harmonic_amplitudes.loc[invalid, "p2p 1"] = None
        first_harmonic_amplitudes.loc[invalid, "p2p 2"] = None

        u1_1_p2p_mean = first_harmonic_amplitudes["p2p 1"].mean()
        u2_1_p2p_mean = first_harmonic_amplitudes["p2p 2"].mean()

        first_harmonic_amplitudes["p2p 1 mean"] = u1_1_p2p_mean
        first_harmonic_amplitudes["p2p 2 mean"] = u2_1_p2p_mean

        # 11.0 Find phase shift (prepare peaks and throws)

        phases = pd.DataFrame({"t": t, "valid period": valid_periods})

        phases["fi1"] = calculate_phases(u1_1_zeros.index, u1_2_throws["peak_idx"], T1_mean, SAMPLING_PERIOD).iloc[1:]
        phases["fi2"] = calculate_phases(u2_1_zeros.index, u2_2_throws["peak_idx"], T1_mean, SAMPLING_PERIOD).iloc[1:]
        phases["fi_hf"] = calculate_phases(
            u1_2_throws["peak_idx"], u2_2_throws["peak_idx"], T1_mean, SAMPLING_PERIOD
        ).iloc[1:]

        phases["fi1*"] = calculate_phases(u1_1_zeros.index, u1_2_extrema["throw_idx"], T1_mean, SAMPLING_PERIOD).iloc[
            1:
        ]
        phases["fi2*"] = calculate_phases(u2_1_zeros.index, u2_2_extrema["throw_idx"], T1_mean, SAMPLING_PERIOD).iloc[
            1:
        ]
        phases["fi_hf*"] = calculate_phases(
            u1_2_extrema["throw_idx"], u2_2_extrema["throw_idx"], T1_mean, SAMPLING_PERIOD
        ).iloc[1:]

        invalid_phases = phases["valid period"].isna()
        phases.loc[invalid_phases, "fi1"] = None
        phases.loc[invalid_phases, "fi2"] = None
        phases.loc[invalid_phases, "fi_hf"] = None
        phases.loc[invalid_phases, "fi1*"] = None
        phases.loc[invalid_phases, "fi2*"] = None
        phases.loc[invalid_phases, "fi_hf*"] = None

        fi1_mean = phases["fi1"].mean()
        fi2_mean = phases["fi2"].mean()
        fi_hf_mean = phases["fi_hf"].mean()

        fi1_mean_ = phases["fi1*"].mean()
        fi2_mean_ = phases["fi2*"].mean()
        fi_hf_mean_ = phases["fi_hf*"].mean()

        phases["fi1 mean"] = fi1_mean
        phases["fi2 mean"] = fi2_mean
        phases["fi_hf mean"] = fi_hf_mean

        phases["fi1 mean*"] = fi1_mean_
        phases["fi2 mean*"] = fi2_mean_
        phases["fi_hf mean*"] = fi_hf_mean_

        # 14.2 Find mean for alpha, gamma

        resampling_periods1 = define_periods_for_resampling_by_two_segments(
            u1_2_peaks["peak_idx"], u1_2_throws["peak_idx"], valid_periods
        )

        resampling_periods2 = define_periods_for_resampling_by_two_segments(
            u2_2_peaks["peak_idx"], u2_2_throws["peak_idx"], valid_periods
        )

        alpha1_mean = round((resampling_periods1["gamma_idx"] - resampling_periods1["alpha_idx"]).mean())
        gamma1_mean = round((resampling_periods1["end_idx"] - resampling_periods1["gamma_idx"]).mean())

        # alpha2_mean = round((resampling_periods2['gamma_idx'] - resampling_periods2['alpha_idx']).mean())
        # gamma2_mean = round((resampling_periods2['end_idx'] - resampling_periods2['gamma_idx']).mean())

        # 14.3 Resample period segments

        u1_1_aplha1_segments = slice_signal_by_segments(
            data["U1-1"], resampling_periods1["alpha_idx"], resampling_periods1["gamma_idx"] - 1
        )
        u1_1_gamma1_segments = slice_signal_by_segments(
            data["U1-1"], resampling_periods1["gamma_idx"], resampling_periods1["end_idx"] - 1
        )

        u1_2_aplha1_segments = slice_signal_by_segments(
            data["U1-2"], resampling_periods1["alpha_idx"], resampling_periods1["gamma_idx"] - 1
        )
        u1_2_gamma1_segments = slice_signal_by_segments(
            data["U1-2"], resampling_periods1["gamma_idx"], resampling_periods1["end_idx"] - 1
        )

        u2_1_aplha2_segments = slice_signal_by_segments(
            data["U2-1"], resampling_periods2["alpha_idx"], resampling_periods2["gamma_idx"] - 1
        )
        u2_1_gamma2_segments = slice_signal_by_segments(
            data["U2-1"], resampling_periods2["gamma_idx"], resampling_periods2["end_idx"] - 1
        )

        u2_2_aplha2_segments = slice_signal_by_segments(
            data["U2-2"], resampling_periods2["alpha_idx"], resampling_periods2["gamma_idx"] - 1
        )
        u2_2_gamma2_segments = slice_signal_by_segments(
            data["U2-2"], resampling_periods2["gamma_idx"], resampling_periods2["end_idx"] - 1
        )

        u1_1_aplha1_normalized_segments = normalize_segments(u1_1_aplha1_segments, alpha1_mean)
        u1_1_gamma1_normalized_segments = normalize_segments(u1_1_gamma1_segments, gamma1_mean)

        u1_2_aplha1_normalized_segments = normalize_segments(u1_2_aplha1_segments, alpha1_mean)
        u1_2_gamma1_normalized_segments = normalize_segments(u1_2_gamma1_segments, gamma1_mean)

        u2_1_aplha2_normalized_segments = normalize_segments(u2_1_aplha2_segments, alpha1_mean)
        u2_1_gamma2_normalized_segments = normalize_segments(u2_1_gamma2_segments, gamma1_mean)

        u2_2_aplha2_normalized_segments = normalize_segments(u2_2_aplha2_segments, alpha1_mean)
        u2_2_gamma2_normalized_segments = normalize_segments(u2_2_gamma2_segments, gamma1_mean)

        # 14.4 Calculate avg period

        u1_1_alpha1_avg = average_segment(u1_1_aplha1_normalized_segments)
        u1_1_gamma1_avg = average_segment(u1_1_gamma1_normalized_segments)

        u1_2_alpha1_avg = average_segment(u1_2_aplha1_normalized_segments)
        u1_2_gamma1_avg = average_segment(u1_2_gamma1_normalized_segments)

        u2_1_alpha2_avg = average_segment(u2_1_aplha2_normalized_segments)
        u2_1_gamma2_avg = average_segment(u2_1_gamma2_normalized_segments)

        u2_2_alpha2_avg = average_segment(u2_2_aplha2_normalized_segments)
        u2_2_gamma2_avg = average_segment(u2_2_gamma2_normalized_segments)

        # 15.0 Detrend average period

        u1_1_avg = pd.concat([u1_1_alpha1_avg, u1_1_gamma1_avg], ignore_index=True)
        u1_2_avg = pd.concat([u1_2_alpha1_avg, u1_2_gamma1_avg], ignore_index=True)
        u2_1_avg = pd.concat([u2_1_alpha2_avg, u2_1_gamma2_avg], ignore_index=True)
        u2_2_avg = pd.concat([u2_2_alpha2_avg, u2_2_gamma2_avg], ignore_index=True)

        u1_1_avg_detrended = detrend_signal(u1_1_avg)
        u1_2_avg_detrended = detrend_signal(u1_2_avg)
        u2_1_avg_detrended = detrend_signal(u2_1_avg)
        u2_2_avg_detrended = detrend_signal(u2_2_avg)

        u1_1_avg_detrended_ext = extend_signal_as_periods(u1_1_avg_detrended - u1_1_avg_detrended.mean())
        u1_2_avg_detrended_ext = extend_signal_as_periods(u1_2_avg_detrended - u1_2_avg_detrended.mean())
        u2_1_avg_detrended_ext = extend_signal_as_periods(u2_1_avg_detrended - u2_1_avg_detrended.mean())
        u2_2_avg_detrended_ext = extend_signal_as_periods(u2_2_avg_detrended - u2_2_avg_detrended.mean())

        coef_value = math.cos(math.radians(fi1_mean)) / math.cos(math.radians(fi2_mean))

        if self.cfg.coef_formula:
            variables = {
                "fi_lf_1": math.radians(fi1_mean),
                "fi_lf_2": math.radians(fi2_mean),
                "fi_hf": math.radians(fi_hf_mean),
                "fi_lf_1_new": math.radians(fi1_mean_),
                "fi_lf_2_new": math.radians(fi2_mean_),
                "fi_hf_new": math.radians(fi_hf_mean_),
                "cos": math.cos,
                "sin": math.sin,
                "pi": math.pi,
                "abs": abs,
            }

            coef_value = eval(self.cfg.coef_formula, {"__builtins__": None}, variables)

            if DEBUG:
                print(f"[formula eval] {self.cfg.coef_formula} = {coef_value}", flush=True)

        K = u1_1_p2p_mean / u2_1_p2p_mean * coef_value

        u1_avg_detrended_ext = u1_1_avg_detrended_ext + u1_2_avg_detrended_ext
        u2_avg_detrended_ext = u2_1_avg_detrended_ext + u2_2_avg_detrended_ext

        period_length_samples = len(u1_1_avg)
        # should be minus according to the previous calculations
        shift_samples = -int(round((fi_hf_mean / 360.0) * period_length_samples))
        u2_avg_detrended_ext_aligned = pd.Series(
            np.roll(u2_avg_detrended_ext.values * K, shift_samples), index=u2_avg_detrended_ext.index
        )

        u2_avg_detrended_ext_aligned_no_coef = pd.Series(
            np.roll(u2_avg_detrended_ext.values, shift_samples), index=u2_avg_detrended_ext.index
        )

        df = pd.DataFrame(
            {
                "idx": np.arange(len(u1_avg_detrended_ext)),
                "t": np.arange(len(u1_avg_detrended_ext)) * SAMPLING_PERIOD,
                "U1-1 avg period": u1_1_avg_detrended_ext,
                "U1-2 avg period": u1_2_avg_detrended_ext,
                "U2-1 avg period": u2_1_avg_detrended_ext,
                "U2-2 avg period": u2_2_avg_detrended_ext,
                "U1 avg period": u1_avg_detrended_ext,
                "U2 avg period": u2_avg_detrended_ext,
                "U2 avg period aligned": u2_avg_detrended_ext_aligned,
            }
        )

        # important points
        df["U1-3 avg period"] = get_smoothed_derivative(df["U1-2 avg period"], window=5)
        df["U2-3 avg period"] = get_smoothed_derivative(df["U2-2 avg period"], window=5)

        u1_1_avg_zeros = find_zeros(df["U1-1 avg period"])
        u2_1_avg_zeros = find_zeros(df["U2-1 avg period"])

        u1_3_avg_peaks = find_peaks_around_points(
            df["U1-3 avg period"],
            u1_1_avg_zeros,
            period_to_samples(delta1, SAMPLING_PERIOD),
            period_to_samples(delta2, SAMPLING_PERIOD),
        )
        u2_3_avg_peaks = find_peaks_around_points(
            df["U2-3 avg period"],
            u2_1_avg_zeros,
            period_to_samples(delta1, SAMPLING_PERIOD),
            period_to_samples(delta2, SAMPLING_PERIOD),
        )

        df["U1-1 avg period zeros"] = u1_1_avg_zeros
        df["U2-1 avg period zeros"] = u2_1_avg_zeros

        df["U1-3 avg period peaks"] = u1_3_avg_peaks.set_index("peak_idx")["peak_value"]
        df["U2-3 avg period peaks"] = u2_3_avg_peaks.set_index("peak_idx")["peak_value"]

        u1_3_avg_peaks_series = pd.Series(
            data=u1_3_avg_peaks["peak_value"].values, index=u1_3_avg_peaks["peak_idx"].values
        )
        u1_2_avg_peaks = find_peaks_around_points(
            df["U1-2 avg period"],
            u1_3_avg_peaks_series,
            period_to_samples(0.001, SAMPLING_PERIOD),
            period_to_samples(0.050, SAMPLING_PERIOD),
        )
        u1_2_avg_throws = find_peaks_around_points(
            -df["U1-2 avg period"],
            u1_3_avg_peaks_series,
            period_to_samples(0.050, SAMPLING_PERIOD),
            period_to_samples(0.001, SAMPLING_PERIOD),
        )

        u2_3_avg_peaks_series = pd.Series(
            data=u2_3_avg_peaks["peak_value"].values, index=u2_3_avg_peaks["peak_idx"].values
        )
        u2_2_avg_peaks = find_peaks_around_points(
            df["U2-2 avg period"],
            u2_3_avg_peaks_series,
            period_to_samples(0.001, SAMPLING_PERIOD),
            period_to_samples(0.050, SAMPLING_PERIOD),
        )
        u2_2_avg_throws = find_peaks_around_points(
            -df["U2-2 avg period"],
            u2_3_avg_peaks_series,
            period_to_samples(0.050, SAMPLING_PERIOD),
            period_to_samples(0.001, SAMPLING_PERIOD),
        )

        df["U1-2 avg period peaks"] = u1_2_avg_peaks.set_index("peak_idx")["peak_value"]
        df["U1-2 avg period throws"] = -u1_2_avg_throws.set_index("peak_idx")["peak_value"]
        df["U2-2 avg period peaks"] = u2_2_avg_peaks.set_index("peak_idx")["peak_value"]
        df["U2-2 avg period throws"] = -u2_2_avg_throws.set_index("peak_idx")["peak_value"]

        # Find new HF extrema
        u1_2_avg_throws_series = pd.Series(
            data=-u1_2_avg_throws["peak_value"].values, index=u1_2_avg_throws["peak_idx"].values
        )
        u1_2_avg_peaks_series = pd.Series(
            data=u1_2_avg_peaks["peak_value"].values, index=u1_2_avg_peaks["peak_idx"].values
        )

        u1_2_avg_extrema = find_min_max_distance_from_curve_to_line(
            u1_2_avg_throws_series, u1_2_avg_peaks_series, u1_1_avg_zeros, df["U1-2 avg period"]
        )

        df["U1-2 avg period throws *"] = pd.Series(
            data=u1_2_avg_extrema["throw_value"].values, index=u1_2_avg_extrema["throw_idx"].values
        )
        df["U1-2 avg period peaks *"] = pd.Series(
            data=u1_2_avg_extrema["peak_value"].values, index=u1_2_avg_extrema["peak_idx"].values
        )

        u2_2_avg_throws_series = pd.Series(
            data=-u2_2_avg_throws["peak_value"].values, index=u2_2_avg_throws["peak_idx"].values
        )
        u2_2_avg_peaks_series = pd.Series(
            data=u2_2_avg_peaks["peak_value"].values, index=u2_2_avg_peaks["peak_idx"].values
        )

        u2_2_avg_extrema = find_min_max_distance_from_curve_to_line(
            u2_2_avg_throws_series, u2_2_avg_peaks_series, u2_1_avg_zeros, df["U2-2 avg period"]
        )

        df["U2-2 avg period throws *"] = pd.Series(
            data=u2_2_avg_extrema["throw_value"].values, index=u2_2_avg_extrema["throw_idx"].values
        )
        df["U2-2 avg period peaks *"] = pd.Series(
            data=u2_2_avg_extrema["peak_value"].values, index=u2_2_avg_extrema["peak_idx"].values
        )

        # 16.0 Convert first channel average period to pressure curve

        min_pressure = 80
        max_pressure = 120

        pressure1, rescaling_koef = rescale_signal(u1_avg_detrended_ext, min_pressure, max_pressure)
        avg_pressure1 = pressure1.mean()

        # 17.0 Convert first channel average period to pressure curve

        pressure2 = avg_pressure1 + rescaling_koef * u2_avg_detrended_ext_aligned
        avg_pressure2 = pressure2.mean()

        pressure1_power = (pressure1 - avg_pressure1).abs().sum()
        u2_power = u2_avg_detrended_ext_aligned_no_coef.abs().sum()

        pressure2_aligned_by_power = avg_pressure1 + u2_avg_detrended_ext_aligned_no_coef * (pressure1_power / u2_power)
        avg_pressure2_aligned_by_power = pressure2_aligned_by_power.mean()

        pressure2_aligned_by_min_and_avg = rescale_set_min_preserve_mean(pressure2, pressure1.min())
        avg_pressure2_aligned_by_min_and_avg = pressure2_aligned_by_min_and_avg.mean()

        df["P arm"] = pressure1
        df["P aortal"] = pressure2
        df["P arm avg"] = avg_pressure1
        df["P aortal avg"] = avg_pressure2
        df["P aortal (same power)"] = pressure2_aligned_by_power
        df["P aortal (same power) avg"] = avg_pressure2_aligned_by_power
        df["P aortal (same min and avg)"] = pressure2_aligned_by_min_and_avg
        df["P aortal (same min and avg) avg"] = avg_pressure2_aligned_by_min_and_avg

        # 18.0 FFT analysis

        fft_freqs, fft_ampl_u1_clean = compute_fft(data["U1 clean"], SAMPLING_PERIOD)
        _, fft_ampl_u1_1 = compute_fft(data["U1-1"], SAMPLING_PERIOD)
        _, fft_ampl_u1_2 = compute_fft(data["U1-2"], SAMPLING_PERIOD)
        _, fft_ampl_u2_clean = compute_fft(data["U2 clean"], SAMPLING_PERIOD)
        _, fft_ampl_u2_1 = compute_fft(data["U2-1"], SAMPLING_PERIOD)
        _, fft_ampl_u2_2 = compute_fft(data["U2-2"], SAMPLING_PERIOD)

        fft_df = pd.DataFrame(
            {
                "Frequency [Hz]": fft_freqs,
                "U1 clean amplitude": fft_ampl_u1_clean,
                "U1-1 amplitude": fft_ampl_u1_1,
                "U1-2 amplitude": fft_ampl_u1_2,
                "U2 clean amplitude": fft_ampl_u2_clean,
                "U2-1 amplitude": fft_ampl_u2_1,
                "U2-2 amplitude": fft_ampl_u2_2,
            }
        )

        # =================================== PREPARE DATASETS ===========================================

        data.drop(columns=["U1 clean*", "U2 clean*"], inplace=True)

        data.set_index("t", inplace=True)
        periods1.set_index("t", inplace=True)
        phases.set_index("t", inplace=True)

        data["T1*"] = periods1["T1*"]
        data["T1**"] = periods1["T1**"]
        data["T1** mean"] = periods1["T1** mean"]
        data["+ 2 sigma1"] = periods1["+ 2 sigma1"]
        data["- 2 sigma1"] = periods1["- 2 sigma1"]
        data["T1** valid"] = periods1["T1** valid"]
        data["fi1"] = phases["fi1"]
        data["fi1 mean"] = phases["fi1 mean"]
        data["fi2"] = phases["fi2"]
        data["fi2 mean"] = phases["fi2 mean"]
        data["fi_hf"] = phases["fi_hf"]
        data["fi_hf mean"] = phases["fi_hf mean"]

        data["fi1*"] = phases["fi1*"]
        data["fi1 mean*"] = phases["fi1 mean*"]
        data["fi2*"] = phases["fi2*"]
        data["fi2 mean*"] = phases["fi2 mean*"]
        data["fi_hf*"] = phases["fi_hf*"]
        data["fi_hf mean*"] = phases["fi_hf mean*"]

        data["+ 2 sigma1 (before filter)"] = periods1["+ 2 sigma1 (before filter)"]
        data["- 2 sigma1 (before filter)"] = periods1["- 2 sigma1 (before filter)"]

        data.reset_index(inplace=True)
        df.drop(columns=["idx"], inplace=True)

        return {"1. Time series": data, "2. Average period": df, "3. FFT analysis": fft_df}
