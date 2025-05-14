import time

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, medfilt


def estimate_pulse(data, signal_column="U1", time_column="t", min_bpm=60):
    """
    Estimates pulse (in BPM) from a signal by detecting peaks.

    Parameters:
        data (pd.DataFrame): DataFrame with time and signal columns.
        signal_column (str): Column name for the signal (e.g., 'U1').
        time_column (str): Column name for the time values (e.g., 't').
        min_bpm (int): Minimal expected BPM (used to calculate peak distance).

    Returns:
        dict: {
            'sampling_period': float,
            'frequency': float,
            'pulse': int,
            'peaks': np.ndarray
        }
    """
    sampling_period = data[time_column].iloc[1] - data[time_column].iloc[0]
    min_peak_distance_sec = 60 / (2 * min_bpm)
    peaks_distance = min_peak_distance_sec / sampling_period

    peaks, _ = find_peaks(data[signal_column], distance=peaks_distance)

    duration_sec = sampling_period * len(data[signal_column])
    frequency = len(peaks) / duration_sec
    pulse = int(round(60 * frequency))

    return {"sampling_period": sampling_period, "frequency": frequency, "pulse": pulse, "peaks": peaks}


def min_max_normalize(series):
    return 2 * (series - series.min()) / (series.max() - series.min()) - 1


def smooth_original_signal(original_signal, window_size, num_iterations=3, include_iterations=False):
    kernel = np.ones(window_size) / window_size

    iteration_signals = []
    cleaned_signal = original_signal.copy()
    base_signal = cleaned_signal

    result = {}

    for _ in range(num_iterations):
        convolution_result = np.convolve(cleaned_signal, kernel, mode="valid")
        base_signal = np.pad(convolution_result, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
        cleaned_signal -= base_signal
        if include_iterations:
            iteration_signals.append(cleaned_signal.copy())

    result["cleaned_signal"] = cleaned_signal
    result["base_signal"] = original_signal - cleaned_signal
    if include_iterations:
        result["iteration_signals"] = iteration_signals

    return result


def butter_filter(data, cutoff, btype, sampling_period, order=3):
    sampling_frequency = 1 / sampling_period
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, data)


def extract_first_harmonic(signal, frequency, sampling_period, butter_order=3, num_iterations=1, track_time=False):
    result = signal.copy()

    if track_time:
        start_time = time.time()

    for _ in range(num_iterations):
        result = 2 * butter_filter(result, frequency, "low", sampling_period, butter_order)
        result = 2 * butter_filter(result, frequency, "high", sampling_period, 1)

    if track_time:
        end_time = time.time()
        print(
            "extract_first_harmonic",
            "butter_order",
            butter_order,
            "num_iterations",
            num_iterations,
            "exec time",
            end_time - start_time,
        )

    return result


def find_zero_crossing_indices(signal):
    return np.where(np.diff(np.sign(signal)) > 0)[0]


def align_zero_crossings(*signals):
    """
    Aligns multiple signals based on the second zero-crossing of the first (main) signal.

    Parameters:
        *signals: pandas Series objects (same length, same index ideally).

    Returns:
        list of pandas.Series: aligned zero-crossing values with original indices.
    """
    if len(signals) < 2:
        raise ValueError("Need at least two signals to align.")

    # Step 1: Find zero-crossing indices
    all_idxs = [find_zero_crossing_indices(sig) for sig in signals]
    main_idxs = all_idxs[0]

    if len(main_idxs) < 3:
        raise ValueError("Main signal has too few zero-crossings to align.")

    # Step 2: Define alignment window using 1st and 3rd zero-crossing of main signal
    main_window_start = main_idxs[0]
    main_window_end = main_idxs[2]

    # Step 3: Slice main signal from second zero
    aligned_idxs = [main_idxs[1:]]

    # Step 4: Align all other signals by matching a zero-crossing inside main's window
    for idxs in all_idxs[1:]:
        # Find first zero in the window of main signal
        valid = idxs[(idxs >= main_window_start) & (idxs <= main_window_end)]
        if len(valid) == 0:
            raise ValueError("Could not align signal within main zero-crossing window.")

        # Start from that aligned point
        start_pos = np.where(idxs == valid[0])[0][0]
        aligned_idxs.append(idxs[start_pos:])

    # Step 5: Trim to shortest length
    min_len = min(len(idxs) for idxs in aligned_idxs)
    aligned_idxs = [idxs[:min_len] for idxs in aligned_idxs]

    # Step 6: Build Series with original indices preserved
    aligned_signals = [
        pd.Series(data=sig.iloc[idxs].values, index=sig.iloc[idxs].index)
        for sig, idxs in zip(signals, aligned_idxs, strict=False)
    ]

    return aligned_signals


def find_zeros(signal):
    zero_indexes = np.where(np.diff(np.sign(signal)) > 0)[0]
    result = pd.Series(signal[zero_indexes])
    return result


def extract_high_freq_signal(signal, first_harmonic, low_cutoff, high_cutoff, sampling_period):
    result = butter_filter(signal, low_cutoff, "high", sampling_period)
    result = butter_filter(result, high_cutoff, "low", sampling_period)
    result -= first_harmonic
    return result


def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_period):
    result = butter_filter(signal, low_cutoff, "high", sampling_period)
    result = butter_filter(result, high_cutoff, "low", sampling_period)
    return result


def calculate_derivative(signal: pd.Series, rescale: bool = False) -> pd.Series:
    """
    Approximates the derivative of a signal using finite differences.
    Optionally rescales the derivative by a constant factor so that
    its maximum absolute value matches that of the original signal.

    Parameters:
        signal (pd.Series): The input signal.
        rescale (bool): Whether to rescale the derivative. Default is False.

    Returns:
        pd.Series: The approximated (and optionally rescaled) derivative.
    """
    # Central difference approximation
    derivative = (signal.shift(-1) - signal.shift(1)).copy()

    # Forward difference for the first point
    if len(signal) > 1:
        derivative.iloc[0] = signal.iloc[1] - signal.iloc[0]
        # Backward difference for the last point
        derivative.iloc[-1] = signal.iloc[-1] - signal.iloc[-2]

    if rescale:
        max_signal = signal.abs().max()
        max_derivative = derivative.abs().max()

        if max_derivative != 0:
            coef = max_signal / max_derivative * 0.75
            derivative *= coef
        else:
            derivative[:] = 0  # or leave it unscaled?

    return derivative


def apply_smooth_filter(signal: pd.Series, window: int = 3, drop_index=True) -> pd.Series:
    # Validate the rolling window size
    if window < 1 or window % 2 == 0:
        raise ValueError("Rolling window size must be a positive odd integer.")

    padded_signal = np.pad(signal, pad_width=window // 2, mode="edge")

    result = (
        pd.Series(padded_signal)
        .rolling(window=window, center=True)
        .mean()
        .iloc[window // 2 : -(window // 2)]
        .reset_index(drop=drop_index)
    )

    return result


def get_smoothed_derivative(signal: pd.Series, window: int = 3) -> pd.Series:
    """
    Calculates a smoothed derivative of a signal using finite differences
    and a rolling average for smoothing.

    Parameters:
        signal (pd.Series): The input signal.
        window (int): The size of the rolling window for smoothing (default is 3).

    Returns:
        pd.Series: The smoothed derivative of the signal.
    """
    # Validate the rolling window size
    if window < 1 or window % 2 == 0:
        raise ValueError("Rolling window size must be a positive odd integer.")

    # Compute the derivative using the calculate_derivative function
    derivative = calculate_derivative(signal, True)

    return apply_smooth_filter(derivative, window)


def period_to_samples(period: float, sampling_period: float) -> int:
    return int(period // sampling_period)


def find_peaks_around_points(signal, center_points, delta1, delta2, DEBUG=False):
    peaks = []

    max_extend_steps = 10

    for center_idx in center_points.index:
        start_idx = max(center_idx - delta1, 0)
        end_idx = min(center_idx + delta2, len(signal))

        window = signal.iloc[start_idx:end_idx]
        max_idx = window.idxmax()

        steps = 0
        while (max_idx == start_idx or max_idx == end_idx - 1) and steps < max_extend_steps:
            prev_start, prev_end = start_idx, end_idx
            start_idx = max(start_idx - delta1, 0)
            end_idx = min(end_idx + delta2, len(signal))
            window = signal.iloc[start_idx:end_idx]
            max_idx = window.idxmax()
            steps += 1

            if start_idx == prev_start and end_idx == prev_end:
                break

        if DEBUG and (max_idx == start_idx or max_idx == end_idx - 1):
            print(f"[DEBUG] Peak near edge for center {center_idx} " f"after {steps} extensions: max_idx = {max_idx}")

        peaks.append({"idx": center_idx, "peak_idx": max_idx, "peak_value": signal.iloc[max_idx]})

    return pd.DataFrame(peaks)


def compute_periods(points: pd.Series) -> pd.Series:
    """
    Calculates periods as difference between consecutive points in a pandas Series.

    Parameters:
    - points: pd.Series
        Input pandas Series.

    Returns:
    - pd.Series
        Series containing periods.
    """
    return points.diff().dropna()


def apply_median_filter(series: pd.Series, window_size=3) -> pd.Series:
    return medfilt(series, kernel_size=window_size)


def filter_valid_periods(series: pd.Series, mean: float, std: float) -> pd.Series:
    """
    Filters periods based on the following conditions:
    1. Periods must be within 2 standard deviations of the mean.
    2. Identifies consecutive subseries of at least 2 periods.
    3. Excludes the first period in each consecutive subseries.

    Parameters:
    - series: pd.Series
        A pandas Series representing the periods to filter.

    Returns:
    - pd.Series
        A pandas Series containing the periods that match the conditions.
    """
    # Create a temporary DataFrame with original data and indices
    temp_df = pd.DataFrame(
        {
            "original_index": series.index,
            "value": series.values,
        }
    )
    temp_df["reindex"] = range(len(temp_df))  # Sequential reindex

    # Filter rows within 2 standard deviations
    temp_df = temp_df[(temp_df["value"] >= mean - 2 * std) & (temp_df["value"] <= mean + 2 * std)]

    # Step 4: Identify consecutive subseries based on the reindex
    bad_periods = temp_df["reindex"].diff() != 1
    temp_df["group"] = bad_periods.cumsum()

    # Filter groups with at least 2 elements (we use 3 here as we need exclude also 1st period in the series)
    valid_groups = temp_df.groupby("group").filter(lambda g: len(g) >= 3)

    # Exclude the first period in each valid group
    valid_groups = valid_groups.groupby("group").apply(lambda g: g.iloc[1:])

    # Map back to the original Series
    valid_indices = valid_groups["original_index"]
    return series.loc[valid_indices]


def calculate_phases(zero_idx: pd.Series, extremum_idx: pd.Series, period: float, sampling_period: float) -> pd.Series:
    return 360 * (zero_idx - extremum_idx) * sampling_period / period


def define_periods_for_resampling(
    peak_idx: pd.Series, throw_idx: pd.Series, derivarive_peak_idx: pd.Series, period_marks: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "period_mark": period_marks,
            "start_idx": throw_idx,
            "alpha_idx": throw_idx,
            "beta_idx": derivarive_peak_idx,
            "gamma_idx": peak_idx,
        }
    )

    df["end_idx"] = (throw_idx.shift(-1) - 1).fillna(-1).astype(int)

    result = df[df["period_mark"].notna()]
    result = result.drop(columns=["period_mark"])

    return result


def define_periods_for_resampling_by_two_segments(
    peak_idx: pd.Series, throw_idx: pd.Series, period_marks: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"period_mark": period_marks, "start_idx": throw_idx, "alpha_idx": throw_idx, "gamma_idx": peak_idx}
    )

    df["end_idx"] = (throw_idx.shift(-1) - 1).fillna(-1).astype(int)

    result = df[df["period_mark"].notna()]
    result = result.drop(columns=["period_mark"])

    return result


def slice_signal_by_segments(signal: pd.Series, segment_starts: pd.Series, segment_ends: pd.Series) -> list:
    """
    Slices a signal series based on segment boundaries defined by start and end series.

    Parameters:
    - signal: pd.Series
        The signal data to slice.
    - segment_starts: pd.Series
        Series containing start indices of segments.
    - segment_ends: pd.Series
        Series containing end indices of segments (inclusive).

    Returns:
    - list of pd.Series
        A list of sliced signal segments, each as a pandas Series.
    """
    # Validate input lengths
    if len(segment_starts) != len(segment_ends):
        raise ValueError("segment_starts and segment_ends must have the same length.")

    sliced_segments = []
    for start, end in zip(segment_starts, segment_ends, strict=False):
        if start > end:
            raise ValueError(f"Start index {start} cannot be greater than end index {end}.")
        # Slice the signal
        sliced_segment = signal.iloc[start : end + 1]  # +1 to include the end index
        sliced_segments.append(sliced_segment)

    return sliced_segments


def normalize_segments(segments: list, average_length: int) -> list:
    """
    Normalizes segments to a specified average length using interpolation.

    Parameters:
    - segments: list of pd.Series
        List of signal segments to normalize.
    - average_length: int
        The target length for each normalized segment.

    Returns:
    - list of pd.Series
        List of normalized segments, each with the specified average length.
    """
    if average_length <= 0:
        raise ValueError("average_length must be a positive integer.")

    normalized_segments = []
    for segment in segments:
        original_length = len(segment)
        if original_length == 0:
            continue  # Skip empty segments

        # Original x values (indices of the segment)
        original_x = np.linspace(0, 1, original_length)
        # Target x values (for interpolation)
        target_x = np.linspace(0, 1, average_length)

        # Interpolate signal values to match the target length
        interpolated_values = np.interp(target_x, original_x, segment.values)
        normalized_segment = pd.Series(interpolated_values, index=np.arange(average_length))
        normalized_segments.append(normalized_segment)

    return normalized_segments


def average_segment(segments: list) -> pd.Series:
    """
    Calculate the average of multiple segments (all segments have the same length).

    Parameters:
    - segments: list of pandas Series
        List of segments (pandas Series) that should be averaged.

    Returns:
    - pd.Series
        The average segment.
    """
    # Stack the segments and calculate the mean across them
    avg_segment = pd.concat(segments, axis=1).mean(axis=1)

    return avg_segment


def detrend_signal(signal: pd.Series) -> pd.Series:
    """
    Removes the linear trend defined by the first and last points of the signal.

    Parameters:
    - signal: pd.Series
        The signal from which the linear trend should be subtracted.

    Returns:
    - pd.Series
        The detrended signal.
    """
    # Extract x and y values for the first and last points
    x_start, x_end = signal.index[0], signal.index[-1]
    y_start, y_end = signal.iloc[0], signal.iloc[-1]

    # Calculate slope (m) and intercept (b) of the line
    m = (y_end - y_start) / (x_end - x_start)
    b = y_start - m * x_start

    # Generate the linear trend line for the signal
    linear_trend = m * signal.index + b

    # Subtract the linear trend from the signal
    detrended_signal = signal - linear_trend

    return detrended_signal


def extend_signal_as_periods(
    signal: pd.Series, num_periods: int = 1, reset_index: bool = True, deduplicate_endpoint: bool = True
) -> pd.Series:
    """
    Extends a periodic signal by adding half a period before and after the repeated full periods.
    Ensures no duplicated points at stitching boundaries for perfect periodic tiling.

    Parameters:
    - signal: pd.Series
        One period of the signal.
    - num_periods: int
        Number of full periods to include.
    - reset_index: bool
        Whether to reset the index of the returned Series.
    - deduplicate_endpoint: bool
        If True, removes the last sample if it equals the first (to avoid double endpoint).

    Returns:
    - pd.Series
        The extended signal with smooth stitching and no point duplication.
    """
    if len(signal) < 2:
        raise ValueError("Signal must have at least 2 samples.")

    # Optionally deduplicate the endpoint
    if deduplicate_endpoint and np.isclose(signal.iloc[0], signal.iloc[-1]):
        signal = signal.iloc[:-1]

    L = len(signal)
    half1 = L // 2
    half2 = L - half1

    start = signal.iloc[-half1:]  # last half1 points
    middle = pd.concat([signal] * num_periods, ignore_index=True)
    end = signal.iloc[:half2]  # first half2 points

    extended = pd.concat([start, middle, end], ignore_index=True)

    return extended.reset_index(drop=True) if reset_index else extended


def rescale_signal(signal: pd.Series, min_value: float, max_value: float):
    """
    Rescales a signal's values to fit within a specified range [min_value, max_value].
    Returns the rescaled signal and the rescaling coefficient.

    Parameters:
    - signal: pd.Series
        The input signal to be rescaled.
    - min_value: float
        The desired minimum value of the rescaled signal.
    - max_value: float
        The desired maximum value of the rescaled signal.

    Returns:
    - pd.Series
        The rescaled signal.
    - float
        The rescaling coefficient used.
    """
    # Get the current min and max values of the signal
    current_min = signal.min()
    current_max = signal.max()

    # Avoid division by zero if the signal has constant values
    if current_max == current_min:
        return pd.Series([min_value] * len(signal), index=signal.index), 0.0

    # Compute the rescaling coefficient
    rescale_koef = (max_value - min_value) / (current_max - current_min)

    # Rescale the signal
    rescaled_signal = (signal - current_min) * rescale_koef + min_value

    return rescaled_signal, rescale_koef


def rescale_set_min_preserve_mean(series: pd.Series, new_min: float) -> pd.Series:
    """
    Rescale a pandas Series so that its minimum is set to `new_min` and its mean remains unchanged.
    """
    orig_min = series.min()
    orig_mean = series.mean()
    shifted = series - orig_min
    shifted_mean = shifted.mean()
    if shifted_mean == 0:
        # All values are the same; set all to new_min
        return pd.Series(new_min, index=series.index)
    scale = (orig_mean - new_min) / shifted_mean
    return shifted * scale + new_min


def prepare_peaks_sequence(series, required_length=12):
    # Ensure the series index is sorted
    series = series.sort_index()

    # Identify non-NaN regions
    sequences = []
    start = None

    for i, is_not_na in enumerate(series.notna()):
        if is_not_na:
            if start is None:
                start = series.index[i]
        else:
            if start is not None:
                sequences.append(series.loc[start : series.index[i - 1]])
                start = None

    # Add the last sequence if it ends at the end of the series
    if start is not None:
        sequences.append(series.loc[start:])

    # Debug: Print extracted sequences
    # print("Extracted sequences:")
    # for seq in sequences:
    #     print(seq)

    # Find the longest sequence or one with the required length
    target_sequence = None
    for seq in sequences:
        if len(seq) == required_length:
            target_sequence = seq
            break
    if target_sequence is None:
        target_sequence = max(sequences, key=len, default=None)

    if target_sequence is None or target_sequence.empty:
        # If no valid sequence is found, return all NaNs
        return pd.Series(np.nan, index=series.index)

    # Debug: Print target sequence
    # print("Target sequence:")
    # print(target_sequence)

    # Extend the sequence by alternately adding values from left and right neighbors
    index = next((i for i, seq in enumerate(sequences) if seq.equals(target_sequence)), -1)
    extended_sequence = target_sequence.copy()
    left_index = index - 1
    right_index = index + 1

    while len(extended_sequence) < required_length:
        if left_index >= 0 and len(extended_sequence) < required_length:
            left_value = sequences[left_index].iloc[-1:]
            extended_sequence = pd.concat([left_value, extended_sequence])
            sequences[left_index] = sequences[left_index].iloc[:-1]
            if sequences[left_index].empty:
                left_index -= 1

        if right_index < len(sequences) and len(extended_sequence) < required_length:
            right_value = sequences[right_index].iloc[:1]
            extended_sequence = pd.concat([extended_sequence, right_value])
            sequences[right_index] = sequences[right_index].iloc[1:]
            if sequences[right_index].empty:
                right_index += 1

        if left_index < 0 and right_index >= len(sequences):
            break

    # Truncate the sequence to the required length
    extended_sequence = extended_sequence.iloc[:required_length]

    # Map the extended sequence back to the original index
    result = pd.Series(np.nan, index=series.index)
    result.loc[extended_sequence.index] = extended_sequence.values

    return result


def find_min_max_distance_from_curve_to_line(p1_series, p2_series, center_points, curve_points):
    results = []
    curve_x = curve_points.index.to_numpy()
    curve_y = curve_points.values

    for center, p1, p2 in zip(center_points.items(), p1_series.items(), p2_series.items(), strict=False):
        center_idx, _ = center
        idx1, val1 = p1
        idx2, val2 = p2

        x_min, x_max = sorted([idx1, idx2])
        mask_x = (curve_x >= x_min) & (curve_x <= x_max)
        selected_x = curve_x[mask_x]
        selected_y = curve_y[mask_x]
        selected_indices = curve_points.index.to_numpy()[mask_x]

        if len(selected_x) == 0:
            continue

        A = np.array([idx1, val1])
        B = np.array([idx2, val2])

        # Full formula for distance from point to line
        # h = |(B[1] - A[1]) * x - (B[0] - A[0]) * y + B[0] * A[1] - B[1] * A[0]| / length_AB
        # where (x, y) is the point on the curve
        # length_AB = np.hypot(AB_dx, AB_dy)
        # but we don't need the absolute value here
        # because we are looking for min and max

        h = (B[1] - A[1]) * selected_x - (B[0] - A[0]) * selected_y + B[0] * A[1] - B[1] * A[0]
        # changing sign for convenience
        h = -h

        min_idx = np.argmin(h)
        max_idx = np.argmax(h)

        results.append(
            {
                "idx": center_idx,
                "throw_idx": selected_indices[min_idx],
                "throw_value": curve_points.loc[selected_indices[min_idx]],
                "peak_idx": selected_indices[max_idx],
                "peak_value": curve_points.loc[selected_indices[max_idx]],
            }
        )

    return pd.DataFrame(results)


def compute_fft(signal, sampling_period):
    n = len(signal)
    fft_result = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n, sampling_period)
    fft_ampl = np.abs(fft_result) / n
    # Only positive frequencies
    pos_mask = fft_freqs >= 0
    return fft_freqs[pos_mask], fft_ampl[pos_mask]
