import numpy as np
import obspy.core as oc
from scipy.signal import find_peaks
import math


def sliding_window(data, n_features, n_shift):
    """
    Return NumPy array of sliding windows. Which is basically a view into a copy of original data array.

    Arguments:
    data       -- numpy array to make a sliding windows on
    n_features -- length in samples of the individual window
    n_shift    -- shift between windows starting points
    """
    # Get sliding windows shape
    win_count = np.floor(data.shape[0]/n_shift - n_features/n_shift + 1).astype(int)
    shape = [win_count, n_features]

    # Single window strides (byte_shift_per_window, byte_shift_per_element)
    strides = [data.strides[0]*n_shift, data.strides[0]]

    # Get windows
    windows = np.lib.stride_tricks.as_strided(data, shape, strides)

    return windows.copy()


def cut_traces(*traces):
    """
    Cut traces to same timeframe (same start time and end time). Returns list of new traces.

    Positional arguments:
    Any number of traces (depends on the amount of channels). Unpack * if passing a list of traces.
    e.g. scan_traces(*trs)
    """
    start_time = max([x.stats.starttime for x in traces])
    end_time = max([x.stats.endtime for x in traces])

    return_traces = [x.slice(start_time, end_time) for x in traces]

    return return_traces


def normalize_traces(*traces, global_normalize = True):
    """
    Normalizes traces, alters argument traces.

    Positional arguments:
    Any number of traces (depends on the amount of channels). Unpack * if passing a list of traces.
    e.g. scan_traces(*trs)
    """
    if not global_normalize:
        for x in traces:
            x.normalize()
    else:
        st = oc.Stream(traces)
        st.normalize(global_max = True)


def pre_process_stream(stream, frequency):
    """
    Does preprocessing on the stream (changes it's frequency), does linear detrend and
    highpass filtering with frequency of 2 Hz.

    Arguments:
    stream      -- obspy.core.stream object to pre process
    frequency   -- required frequency
    """
    stream.detrend(type="linear")
    stream.filter(type="highpass", freq = 2)

    required_dt = 1. / frequency
    dt = stream[0].stats.delta

    if dt != required_dt:
        stream.interpolate(frequency)


def scan_traces(*traces, model = None, n_features = 400, shift = 10, global_normalize = True, batch_size = 100):
    """
    Get predictions on the group of traces.

    Positional arguments:
    Any number of traces (depends on the amount of channels). Unpack * if passing a list of traces.
    e.g. scan_traces(*trs)

    Keyword arguments
    model            -- NN model
    n_features       -- number of input features in a single channel
    shift            -- amount of samples between windows
    global_normalize -- normalize globaly all traces if True or locally if False
    batch_size       -- model.fit batch size
    """
    # Check input types
    for x in traces:
        if type(x) != oc.trace.Trace:
            raise TypeError('traces should be a list or containing obspy.core.trace.Trace objects')

    # Cut all traces to a same timeframe
    traces = cut_traces(*traces)

    # Normalize
    normalize_traces(*traces, global_normalize = global_normalize)

    # Get sliding window arrays
    l_windows = []
    #try:
    for x in traces:
        l_windows.append(sliding_window(x.data, n_features = n_features, n_shift = shift))
    #except Error as e:
        #return None

    # TODO: this is quick fix: remove later
    w_length = min([x.shape[0] for x in l_windows])

    # Prepare data
    windows = np.zeros((w_length, n_features, len(l_windows)))

    # print(f'x.data.shape: {x.data.shape}')
    for i in range(len(l_windows)):
        # print(f'{i}: {l_windows[i].shape}')
        windows[:, :, i] = l_windows[i][:w_length]

    # Predict
    # TODO: check if verbose can be numerical like .fit()
    # TODO: make predict through `getattr` method:
    #  predict = getattr(model, predict_method_name)
    #  predict(windows, *args, **kwargs)
    #  args and kwargs for predict pass as function arguments
    # TODO: or rather pass a method as a parameter, like predict(model, *args, **kwargs)
    scores = model.predict(windows, verbose = False, batch_size = batch_size)

    return scores


def restore_scores(scores, shape, shift):
    """
    Restores scores to original size using linear interpolation.

    Arguments:
    scores -- original 'compressed' scores
    shape  -- shape of the restored scores
    shift  -- sliding windows shift
    """
    new_scores = np.zeros(shape)
    for i in range(1, scores.shape[0]):

        for j in range(scores.shape[1]):

            start_i = (i - 1) * shift
            end_i = i * shift
            if end_i >= shape[0]:
                end_i = shape[0] - 1

            new_scores[start_i : end_i, j] = np.linspace(scores[i - 1, j], scores[i, j], shift + 1)[:end_i - start_i]

    return new_scores


def get_positives(scores, peak_indx, other_indxs, peak_dist = 10000, avg_window_half_size = 100, min_threshold = 0.8):
    """
    Returns positive prediction list in format: [[sample, pseudo-probability], ...]
    """
    positives = []

    x = scores[:, peak_indx]
    peaks = find_peaks(x, distance = peak_dist, height=[min_threshold, 1.])

    for i in range(len(peaks[0])):

        start_id = peaks[0][i] - avg_window_half_size
        if start_id < 0:
            start_id = 0

        end_id = start_id + avg_window_half_size*2
        if end_id > len(x):
            end_id = len(x) - 1
            start_id = end_id - avg_window_half_size*2

        # Get mean values
        peak_mean = x[start_id : end_id].mean()

        means = []
        for indx in other_indxs:

            means.append(scores[:, indx][start_id : end_id].mean())

        is_max = True
        for m in means:

            if m > peak_mean:
                is_max = False

        if is_max:
            positives.append([peaks[0][i], peaks[1]['peak_heights'][i]])

    return positives


def truncate(f, n):
    """
    Floors float to n-digits after comma.
    """
    return math.floor(f * 10 ** n) / 10 ** n


def print_results(detected_peaks, filename):
    """
    Prints out peaks in the file.
    """
    with open(filename, 'a') as f:

        for record in detected_peaks:

            line = ''
            # Print wave type
            line += f'{record["type"]} '

            # Print pseudo-probability
            line += f'{truncate(record["pseudo-probability"], 2):1.2f} '

            # Print station
            line += f'{record["station"]} '

            # Print location
            line += f'{record["location_code"]} '

            # Print net code
            line += f'{record["network_code"]}   '

            # Print time
            dt_str = record["datetime"].strftime("%d.%m.%Y %H:%M:%S")
            line += f'{dt_str}   '

            # Print channels
            line += f'{[ch for ch in record["channels"]]}\n'

            # Write
            f.write(line)
