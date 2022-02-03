import sys
import os
import sys
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import obspy
from scipy.signal import find_peaks
import obspy
import obspy.core as oc
from time import time
from collections import deque
from .seisan import generate_events, get_events, archive_to_path, order_group
from . import h5_tools


def pre_process_stream(stream, params, station, frequency=None):
    """
    Does preprocessing on the stream (changes it's frequency), does linear detrend and
    highpass filtering with frequency of 2 Hz.

    Arguments:
    stream      -- obspy.core.stream object to pre process
    frequency   -- required frequency
    """
    if not frequency:
        frequency = params[station, 'frequency']

    no_filter = params[station, 'no-filter']
    no_detrend = params[station, 'no-detrend']

    if not no_detrend:
        stream.detrend(type="linear")
    if not no_filter:
        # stream.filter('bandpass', freqmin=2, freqmax=5)
        stream.filter(type="highpass", freq=2)

    required_dt = 1. / frequency
    dt = stream[0].stats.delta

    if dt != required_dt:
        stream.interpolate(frequency)


def combined_traces(streams, params):
    """
    Gets a list of combined traces from streams, each trace has the same start and end time.
    :param streams: List of streams to
    :return: list lists of obspy.Trace objects:
                [
                  [trace11, trace12, trace13],
                  [trace21, trace22, trace23],
                  ...
                ]
              each row is a group of traces, while each column is a channel (original stream).
              Order of streams preserved.
    """
    # Gather time spans for every existing trace in a stack
    time_span_stacks = [[] for _ in streams]
    for i, stream in enumerate(streams):
        j = len(stream) - 1
        while j >= 0:
            time_span_stacks[i].append([stream[j].stats.starttime, stream[j].stats.endtime, j])
            j -= 1
    for stack in time_span_stacks:
        stack.sort(key=itemgetter(0), reverse=True)

    # Process time spans
    result_time_spans = []
    spans_remaining = True
    while spans_remaining:

        max_start = max([stack[-1][0] for stack in time_span_stacks])

        # Check for spans outside avaliable time
        spans_removed = False
        for stack in time_span_stacks:
            if stack[-1][1] < max_start:
                stack.pop()
                spans_removed = True

        if not spans_removed:
            # Sync spans start
            traces_ids = []
            for stack in time_span_stacks:
                stack[-1][0] = max_start
                traces_ids.append(stack[-1][2])

            # Sync by end time
            min_end = min([stack[-1][1] for stack in time_span_stacks])

            for stack in time_span_stacks:
                if stack[-1][1] > min_end:
                    stack[-1][0] = min_end
                else:
                    stack.pop()

            # Create a time span if it is longer than set in variable
            result_time_spans.append((max_start, min_end, traces_ids))

        for stack in time_span_stacks:
            if not len(stack):
                spans_remaining = False
                break
    result_time_spans.sort(key=itemgetter(0))

    traces = []
    params.data['invalid_combined_traces_groups'] = 0
    total_length = 0
    for i, x in enumerate(result_time_spans):
        sliced_data = []
        for j, stream in enumerate(streams):

            trace = stream[x[2][j]]
            freq = trace.stats.sampling_rate
            start_pos = int((x[0] - trace.stats.starttime)*freq)
            end_pos = int((x[1] - trace.stats.starttime)*freq)

            sliced_data.append((trace.data[start_pos:end_pos], x[0], freq))

        min_length = min([len(data[0]) for data in sliced_data])
        max_length = max([len(data[0]) for data in sliced_data])
        length_diff = max_length - min_length
        if length_diff >= params['main', 'combine-traces-min-length-difference-error']:
            print(f'Warning: Traces of unequal length during combined_traces ({length_diff} samples difference)!',
                  file=sys.stderr)

        if length_diff:
            sliced_data = [(data[0][:min_length], data[1], data[2]) for data in sliced_data]

        sliced_traces = []
        for data in sliced_data:
            sliced_trace = obspy.Trace(data[0])
            sliced_trace.stats.starttime = data[1]
            sliced_trace.stats.sampling_rate = data[2]
            sliced_traces.append(sliced_trace)
        traces.append(sliced_traces)
        total_length += min_length

    return traces, total_length


def trim_streams(streams, station_name, start=None, end=None):
    """
    Trims streams to the same overall time span.
    :return: list of trimmed streams
    """
    max_start_time = start
    min_end_time = end

    for stream in streams:

        current_start = min([x.stats.starttime for x in stream])
        current_end = max([x.stats.endtime for x in stream])

        if not max_start_time:
            max_start_time = current_start
        if not min_end_time:
            min_end_time = current_end

        if current_start > max_start_time:
            max_start_time = current_start
        if current_end < min_end_time:
            min_end_time = current_end

    cut_streams = []

    if max_start_time > end:
        print(f'\nSkipping archives for {station_name}: archive(s) starts '
              f'({max_start_time.strftime("%Y-%m-%d %H:%M:%S")}) after the end of scanning window '
              f'({max_start_time.strftime("%Y-%m-%d %H:%M:%S")})!')
        return None
    for st in streams:
        cut_streams.append(st.slice(max_start_time, min_end_time))

    # Check if streams has any traces
    for stream in cut_streams:
        if len(stream) == 0:
            return None

    return cut_streams


def progress_bar(progress, characters_count=20,
                 erase_line=True,
                 empty_bar='.', filled_bar='=', filled_edge='>',
                 prefix='', postfix='',
                 add_space_around=True):
    """
    Prints progress bar.
    :param progress: percentage (0..1) of progress, or int number of characters filled in progress bar.
    :param characters_count: length of the bar in characters.
    :param erase_line: preform return carriage.
    :param empty_bar: empty bar character.
    :param filled_bar: progress character.
    :param filled_edge: progress character on the borderline between progressed and empty,
                        set to None to disable.
    :param prefix: progress bar prefix.
    :param postfix: progress bar postfix.
    :param add_space_around: add space after prefix and before postfix.
    :return:
    """

    space_characters = ' \t\n'
    if add_space_around:
        if len(prefix) > 0 and prefix[-1] not in space_characters:
            prefix += ' '

        if len(postfix) > 0 and postfix[0] not in space_characters:
            postfix = ' ' + postfix

    if erase_line:
        print('\r', end='')

    progress_num = int(characters_count * progress)
    if filled_edge is None:
        print(prefix + filled_bar * progress_num + empty_bar * (characters_count - progress_num) + postfix, end='')
    else:
        bar_str = prefix + filled_bar * progress_num
        bar_str += filled_edge * min(characters_count - progress_num, 1)
        bar_str += empty_bar * (characters_count - progress_num - 1)
        bar_str += postfix

        print(bar_str, end='')


def cut_traces(*_traces):
    """
    Cut traces to same timeframe (same start time and end time). Returns list of new traces.

    Positional arguments:
    Any number of traces (depends on the amount of channels). Unpack * if passing a list of traces.
    e.g. scan_traces(*trs)
    """
    _start_time = max([x.stats.starttime for x in _traces])
    _end_time = max([x.stats.endtime for x in _traces])

    return_traces = [x.slice(_start_time, _end_time) for x in _traces]

    return return_traces


def sliding_window(data, n_features, n_shift):
    """
    Return NumPy array of sliding windows. Which is basically a view into a copy of original data array.

    Arguments:
    data       -- numpy array to make a sliding windows on
    n_features -- length in samples of the individual window
    n_shift    -- shift between windows starting points
    """
    # Get sliding windows shape
    win_count = np.floor(data.shape[0] / n_shift - n_features / n_shift + 1).astype(int)
    shape = (win_count, n_features)

    try:
        windows = np.zeros(shape)
    except ValueError:
        raise

    for _i in range(win_count):
        _start_pos = _i * n_shift
        _end_pos = _start_pos + n_features

        windows[_i][:] = data[_start_pos: _end_pos]

    return windows.copy()


def sliding_window_strided(data, n_features, n_shift, copy=False):
    """
    Return NumPy array of sliding windows. Which is basically a view into a copy of original data array.

    Arguments:
    data       -- numpy array to make a sliding windows on. Shape (n_samples, n_channels)
    n_features -- length in samples of the individual window
    n_shift    -- shift between windows starting points
    copy       -- copy data or return a view into existing data? Default: False
    """
    from numpy.lib.stride_tricks import as_strided

    # Get sliding windows shape
    stride_shape = (data.shape[0] - n_features + n_shift) // n_shift
    stride_shape = [stride_shape, n_features, data.shape[-1]]

    strides = [data.strides[0] * n_shift, *data.strides]

    windows = as_strided(data, stride_shape, strides)

    if copy:
        return windows
    else:
        return windows.copy()


def normalize_windows_global(windows):
    """
    Normalizes sliding windows array. IMPORTANT: windows should have separate memory, striped windows would break.
    :param windows:
    :return:
    """
    # Shape (windows_number, n_features, channels_number)
    n_win = windows.shape[0]
    ch_num = windows.shape[2]

    for _i in range(n_win):
        win_max = np.max(np.abs(windows[_i, :, :]))
        windows[_i, :, :] = windows[_i, :, :] / win_max


def normalize_global(data):
    """
    Normalizes sliding windows array. IMPORTANT: windows should have separate memory, striped windows would break.
    :param data: NumPy array to normalize
    :return:
    """
    # Shape (windows_number, n_features, channels_number)
    m = np.max(np.abs(data[:]))
    data /= m


def plot_positives(scores, windows, threshold):
    idx = 0
    save_name = 'positive_' + str(idx) + '.jpg'
    while os.path.exists(save_name):
        idx += 1
        save_name = 'positive_' + str(idx) + '.jpg'

    for i in range(len(scores)):

        if scores[i][1] > threshold:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

            ax1.set_ylabel('N', rotation=0.)
            ax1.plot(windows[i, :, 0], 'r')

            ax2.set_ylabel('E', rotation=0.)
            ax2.plot(windows[i, :, 1], 'g')

            ax3.set_ylabel('Z', rotation=0.)
            ax3.plot(windows[i, :, 2], 'y')

            plt.savefig(save_name)
            plt.clf()


def plot_oririnal_positives(scores, original_windows, threshold, original_scores=None):
    idx = 0
    save_name = 'original_positive_' + str(idx) + '.jpg'
    while os.path.exists(save_name):
        idx += 1
        save_name = 'original_positive_' + str(idx) + '.jpg'

    for i in range(len(scores)):

        if scores[i][1] > threshold:
            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

            ax1.set_ylabel('N', rotation=0.)
            ax1.plot(original_windows[i, :, 0], 'r')

            ax2.set_ylabel('E', rotation=0.)
            ax2.plot(original_windows[i, :, 1], 'g')

            ax3.set_ylabel('Z', rotation=0.)
            ax3.plot(original_windows[i, :, 2], 'y')

            plt.savefig(save_name)
            plt.clf()


def scan_traces(*_traces, params=None, n_features=400, shift=10, original_data=None, station='main'):
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
    batch_size = params['main', 'batch-size']
    model = params[station, 'model-object']

    # Check input types
    for x in _traces:
        if type(x) != oc.trace.Trace:
            raise TypeError('traces should be a list or containing obspy.core.trace.Trace objects')

    # Cut all traces to a same timeframe
    _traces = cut_traces(*_traces)

    if not params[station, 'trace-normalization']:
        # Get sliding window arrays
        l_windows = []
        for x in _traces:
            l_windows.append(sliding_window(x.data, n_features=n_features, n_shift=params[station, 'shift']))

        if params[station, 'plot-positives-original']:
            original_l_windows = []
            for x in original_data:
                original_l_windows.append(sliding_window(x.data,
                                                         n_features=n_features,
                                                         n_shift=params[station, 'shift']))

        w_length = min([x.shape[0] for x in l_windows])

        # Prepare data
        windows = np.zeros((w_length, n_features, len(l_windows)))
        for _i in range(len(l_windows)):
            windows[:, :, _i] = l_windows[_i][:w_length]

        if params[station, 'plot-positives-original']:
            original_windows = np.zeros((w_length, n_features, len(original_l_windows)))
            for _i in range(len(original_l_windows)):
                original_windows[:, :, _i] = original_l_windows[_i][:w_length]

        # Global max normalization:
        normalize_windows_global(windows)
        if params[station, 'plot-positives-original']:
            normalize_windows_global(original_windows)

    else:
        min_size = min([tr.data.shape[0] for tr in _traces])

        data = np.zeros((min_size, len(_traces)))

        for i, tr in enumerate(_traces):
            data[:, i] = tr.data[:min_size]

        normalize_global(data)

        windows = sliding_window_strided(data, 400, params[station, 'shift'], False)

        if params[station, 'plot-positives-original']:
            original_windows = windows.copy()

    # Predict
    start_time = time()
    _scores = model.predict(windows, verbose=False, batch_size=batch_size)
    performance_time = time() - start_time
    # TODO: create another flag for this, e.g. --culculate-original-probs or something
    if params[station, 'plot-positives-original']:
        original_scores = model.predict(original_windows, verbose=False, batch_size=batch_size)

    # Positives plotting
    threshold = 0.95
    if params[station, 'plot-positives']:
        plot_positives(_scores, windows, threshold)
    if params[station, 'plot-positives-original']:
        plot_oririnal_positives(_scores, original_windows, threshold, original_scores)

    return _scores, performance_time


def restore_scores(_scores, shape, shift):
    """
    Restores scores to original size using linear interpolation.
    Arguments:
    scores -- original 'compressed' scores
    shape  -- shape of the restored scores
    shift  -- sliding windows shift
    """
    new_scores = np.zeros(shape)
    for i in range(1, _scores.shape[0]):

        for j in range(_scores.shape[1]):

            start_i = (i - 1) * shift
            end_i = i * shift
            if end_i >= shape[0]:
                end_i = shape[0] - 1

            new_scores[start_i: end_i, j] = np.linspace(_scores[i - 1, j], _scores[i, j], shift + 1)[:end_i - start_i]

    return new_scores


def get_positives(_scores, peak_idx, other_idxs, peak_dist=10000, avg_window_half_size=100, threshold=0.8):
    """
    Returns positive prediction list in format: [[sample, pseudo-probability], ...]
    """
    _positives = []

    x = _scores[:, peak_idx]

    peaks = find_peaks(x, distance=peak_dist, height=[threshold, 1.])

    for _i in range(len(peaks[0])):

        start_id = peaks[0][_i] - avg_window_half_size
        if start_id < 0:
            start_id = 0

        end_id = start_id + avg_window_half_size * 2
        if end_id > len(x):
            end_id = len(x) - 1
            start_id = end_id - avg_window_half_size * 2

        # Get mean values
        peak_mean = x[start_id: end_id].mean()

        means = []
        for idx in other_idxs:
            means.append(_scores[:, idx][start_id: end_id].mean())

        is_max = True
        for m in means:

            if m > peak_mean:
                is_max = False

        if is_max:
            _positives.append([peaks[0][_i], peaks[1]['peak_heights'][_i]])

    return _positives


def truncate(f, n):
    """
    Floors float to n-digits after comma.
    """
    import math
    return math.floor(f * 10 ** n) / 10 ** n


def print_results(_detected_peaks, params, station, upper_case=True, last_station=None):
    """
    Prints out peaks in the file.
    """
    precision = params[station, 'print-precision']
    filename = params[station, 'out']

    with open(filename, 'a') as f:

        if station != last_station:
            f.write('\n')
            f.write(f'[{station}]\n')

        for record in _detected_peaks:

            line = ''
            # Print station if provided
            if station:
                line += f'{station} '

            # Print wave type
            tp = record['type'].upper() if upper_case else record['type']
            line += f'{tp} '

            # Print pseudo-probability
            line += f'{truncate(record["pseudo-probability"], precision):1.{precision}f} '

            # Print time
            dt_str = record["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0')
            line += f'{dt_str}\n'

            # Write
            f.write(line)


def combine_by_filename(detections, params, single_filename=False):
    """
    Combines detections (represented as lists, indexed by stations) by filenames.
    """
    combined_by_filename = {}
    for station, items in detections.items():

        if single_filename:
            filename = params['main', 'out']
        else:
            filename = params[station, 'out']

        # Convert relative filename to absolute

        # Combine
        if filename not in combined_by_filename:
            combined_by_filename[filename] = []

        combined_by_filename[filename].extend(items)

    # Sort by datetime
    def datetime_getter(x):
        return x['datetime']

    for _, items in combined_by_filename.items():
        items.sort(key=datetime_getter)

    return combined_by_filename


def combine_detections(detections, params,
                       input_mode=False, filename_grouping=True, combine_different_stations=True):

    detections = combine_by_filename(detections, params, single_filename=(not filename_grouping))

    file_groups = {}
    for filename, items in detections.items():

        # Add true flag to every event
        for x in items:
            x['avaliable'] = True

        # Build three lists:
        # All nodes within that node range
        nodes_in_range = [deque([i]) for i in range(len(items))]
        # All nodes which include this node in their range
        nodes_including = [[] for _ in range(len(items))]
        # Amount of connections node has, paired with id
        nodes_connections_count = [[i, 1] for i in range(len(items))]

        for i, x in enumerate(items):

            if input_mode:
                dt_range = params['main', 'combine-events-range']
            else:
                dt_range = params[x['station']['station'], 'combine-events-range']

            x_time = x['datetime']
            n_points = 1

            j = i - 1
            while j >= 0:
                y = items[j]
                dt = abs(x_time - y['datetime'])
                if dt > dt_range:
                    break
                if not combine_different_stations and x['station']['station'] != y['station']['station']:
                    j -= 1
                    continue

                nodes_in_range[i].appendleft(j)
                nodes_including[j].append(i)
                nodes_connections_count[i][1] += 1
                j -= 1

            for j in range(i + 1, len(items)):
                y = items[j]
                dt = abs(x_time - y['datetime'])
                if dt > dt_range:
                    break
                if not combine_different_stations and x['station']['station'] != y['station']['station']:
                    continue
                nodes_in_range[i].append(j)
                nodes_including[j].append(i)
                nodes_connections_count[i][1] += 1

        # Sorting by second item
        def connections_getter(x):
            return x[1]

        # Sorting by station
        def station_getter(x):
            return x['station']['station']

        def input_getter(x):
            return x['input']

        groups = []
        for i in range(len(nodes_connections_count)):

            group = []
            # Sort from most to least connections
            nodes_connections_count.sort(key=connections_getter, reverse=True)

            idx = nodes_connections_count[i][0]

            for in_idx in nodes_in_range[idx]:

                x = items[in_idx]
                if not x['avaliable']:
                    continue

                group.append(x)
                # Remove positive from other groups
                for j in range(i + 1, len(nodes_connections_count)):
                    if nodes_connections_count[j][0] in nodes_including[idx]:
                        nodes_connections_count[j][1] -= 1
                x['avaliable'] = False

            if len(group):
                if input_mode:
                    group.sort(key = input_getter, reverse=True)
                else:
                    group.sort(key = station_getter, reverse=True)
                groups.append([group, items[idx]['datetime']])

        file_groups[filename] = groups

    return file_groups


def print_final_predictions(detections, params, upper_case=True, open_mode='w', input_mode=False):
    """
    Prints final predictions into a file. As an input takes predictions, sturctured
    as dictionary, indexed by output file name, where each element is a pair:
    (group of positives, datetime).
    Group of positives is a list of positive predictions. Each prediction is a dictionary of fields,
    describing the prediction (datetime, station, etc.).
    """
    groups_started = False
    singles_started = False
    for filename, groups in detections.items():
        with open(filename, 'w') as f:
            for group, datetime in groups:

                if not groups_started and len(group) > 1:
                    f.write(f'***COMBINED EVENTS***\n\n')
                    groups_started = True
                if not singles_started and len(group) == 1:
                    f.write(f'***SINGLE DETECTIONS***\n\n')
                    singles_started = True

                s_datetime = datetime.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0')
                f.write(f'[{s_datetime}]\n')

                for record in group:

                    if input_mode:
                        station = record['input']
                    else:
                        station = record['station']['station']
                    precision = params[station, 'print-precision']

                    line = ''
                    # Print station if provided
                    if station:
                        line += f'{station} '

                    # Print wave type
                    tp = record['type'].upper() if upper_case else record['type']
                    line += f'{tp} '

                    # Print pseudo-probability
                    line += f'{truncate(record["pseudo-probability"], precision):1.{precision}f} '

                    # Print time
                    dt_str = record["datetime"].strftime("%Y-%m-%d %H:%M:%S.%f").rstrip('0')
                    line += f'{dt_str}\n'

                    # Write
                    f.write(line)

                f.write('\n')


def split_detections(detections, params, input_mode=False):

    r_detections = {}
    for filename, groups in detections.items():

        r_groups = []
        r_singles = []
        for group, datetime in groups:
            if len(group) > 1:
                r_groups.append((group, datetime))
            else:
                r_singles.append((group, datetime))

        def get_datetime(item):
            return item[1]

        r_groups.sort(key=get_datetime)
        r_singles.sort(key=get_datetime)
        r_detections[filename] = [*r_groups, *r_singles]

    return r_detections


def finalize_predictions(detections, params, upper_case=True, input_mode=False):
    """
    Prints out all predictions with additional visual enhancements.
    """
    detections = combine_detections(detections, params, input_mode=input_mode)

    detections = split_detections(detections, params, input_mode=input_mode)

    print_final_predictions(detections, params, upper_case=True, input_mode=input_mode)

    if not input_mode:
        generate_events(detections, params)


def combine_daily_detections(detections):
    """
    Combines list of detections by days. Resulting data structure is a list of dictionaries:
    [
        {'day': UTCDateTime, 'detections': [detections]},
        ...
    ]
    """
    combined = []
    current_day = None
    daily_detections = None
    for event in detections:
        event = event[0]  # event[1] is event's datetime, event[0] - event's detections
        for x in event:
            x_date = x['datetime']
            if not current_day:
                current_day = x_date
                daily_detections = {'detections': []}
            if current_day.year != x_date.year or current_day.month != x_date.month or current_day.day != x_date.day:
                daily_detections['day'] = current_day
                combined.append(daily_detections)
                current_day = x_date
                daily_detections = {'detections': []}
            daily_detections['detections'].append(x)

    if len(daily_detections['detections']):
        daily_detections['day'] = current_day
        combined.append(daily_detections)

    return combined


def combine_by_filed(detections, key):
    """
    Combines list by filed into dictionaries
    """
    combined = {}
    for x in detections:
        key_value = x[key]
        if key_value not in combined:
            combined[key_value] = []
        combined[key_value].append(x)
    return combined


def get_false_positive(traces_groups, start, end, params):
    """
    Loads data from archive, processes it and then returns as NumPy array.
    """
    for traces in traces_groups:
        starttime = traces[0].stats.starttime
        endtime = traces[0].stats.endtime
        if starttime < start < endtime and starttime < end < endtime:
            data = [trace.data for trace in traces]
            data_length = int(params['main', 'false-positives-length']*params['main', 'false-positives-frequency'])
            p_start = int((start - starttime)*params['main', 'false-positives-frequency'])
            p_end = p_start + data_length
            for x in data:
                if p_end >= x.shape[0]:
                    return None
            sliced_data = np.zeros((data_length, len(data)))
            for i, x in enumerate(data):
                sliced_data[:, i] = x[p_start:p_end]
            data = sliced_data
            max_value = np.max(np.abs(data))
            data /= max_value
            return data
    return None


def get_archive_false_positives(station, day, dates, params):
    """
    Returns NumPy array of processed false positives in shape: (n_false_positives, n_samples, n_channels)
    :param station:
    :param dates:
    :param params:
    :return:
    """
    # Get data
    archives = archive_to_path(station, day, params['main', 'archives'])
    archives = order_group(archives['paths'], params[station['station'], 'channel-order'])
    if not archives:
        return None
    streams = [obspy.read(path) for path in archives]
    for st in streams:
        pre_process_stream(st, params, station['station'], params['main', 'false-positives-frequency'])
    streams = trim_streams(streams, station['station'], params['main', 'start'], params['main', 'end'])
    traces_groups, _ = combined_traces(streams, params)

    data = []
    for date in dates:
        start = date - params['main', 'false-positives-length'] / 2
        end = start + params['main', 'false-positives-length']

        if start.day != end.day:
            continue
        x = get_false_positive(traces_groups, start, end, params)
        if x is not None:
            data.append(x)
    return data


def gather_false_positives(detections, params):
    """
    Collects all false positives and saves them in .h5 file.
    :param detections:
    :param params:
    :return:
    """
    detections = combine_detections(detections, params, filename_grouping=False, combine_different_stations=False)
    detections = detections[params['main', 'out']]

    # Combine by days
    detections = combine_daily_detections(detections)

    # For each day - read all s-files
    false_positives_count = 0
    for day in detections:

        true_positives = get_events(day['day'], params, start=params['main', 'start'], end=params['main', 'end'])
        if len(true_positives):
            true_positives = combine_by_filed(true_positives, 'station')

        # Compare against picks and determine false positives
        daily_detections = day['detections']
        false_positives = {}
        for x in daily_detections:
            x_datetime = x['datetime']
            x_station = x['station']['station']

            check_passed = True
            if x_station in true_positives:
                for tp in true_positives[x_station]:
                    diff = abs(tp['datetime'] - x_datetime)
                    if diff < params['main', 'false-positives-range']:
                        check_passed = False
                        break
            if check_passed:
                if x_station not in false_positives:
                    false_positives[x_station] = {
                        'station': x['station'],
                        'dates': []
                    }
                false_positives[x_station]['dates'].append(x_datetime)

        # Check channels number
        n_channels = None
        for _, fp in false_positives.items():
            station = fp['station']
            n = len(params[station['station'], 'channel-order'])
            if not n_channels:
                n_channels = n
            elif n != n_channels:
                print('Warning: false positives picking - not equal number of channels for every station in the list!',
                      file=sys.stderr)
                return None

        # Prepare and save false positive
        data = []
        for _, fp in false_positives.items():
            x = get_archive_false_positives(fp['station'], day['day'], fp['dates'], params)
            if x is not None:
                data.extend(x)

        # Save data
        data = np.array(data)
        h5_tools.write_batch(params['main', 'false-positives'], 'false-positives', data)
        false_positives_count += data.shape[0]

    print(f'\nFalse positives collected: {data.shape[0]} (saved into "{params["main", "false-positives"]}")!')


def parse_archive_csv(path):
    """
    Parses archives names file. Returns list of filename lists: [[archive1, archive2, archive3], ...]
    :param path:
    :return:
    """
    with open(path) as f:
        lines = f.readlines()

    _archives = []
    for line in lines:
        _archives.append(
            {
                'paths': [x for x in line.split()],
                'station': None
            }
        )

    return _archives


def plot_wave_scores(file_token, wave, scores,
                     start_time, predictions, right_shift=0,
                     channel_names=['N', 'E', 'Z'],
                     score_names=['P', 'S', 'N']):
    """
    Plots waveform and prediction scores as an image
    """
    channels_num = wave.shape[1]
    classes_num = scores.shape[1]
    scores_length = scores.shape[0]

    # TODO: Make figure size dynamically chosen, based on the input length
    fig = plt.figure(figsize=(9.8, 7.), dpi=160)
    axes = fig.subplots(channels_num + classes_num, 1, sharex=True)

    # Plot wave
    for i in range(channels_num):
        axes[i].plot(wave[:, i], color='#000000', linewidth=1.)
        axes[i].locator_params(axis='both', nbins=4)
        axes[i].set_ylabel(channel_names[i])

    # Process events and ticks
    freq = 100.  # TODO: implement through Trace.stats
    labels = {'p': 0, 's': 1}  # TODO: configure labels through options
    # TODO: make sure that labels are not too close.
    ticks = [100, scores_length - 100]
    events = {}

    for label, index in labels.items():

        label_events = []
        for pos, _ in predictions[label]:
            pos += right_shift
            label_events.append(pos)
            ticks.append(pos)

        events[index] = label_events

    # Plot scores
    for i in range(classes_num):

        axes[channels_num + i].plot(scores[:, i], color='#0022cc', linewidth=1.)

        if i in events:
            for pos in events[i]:
                axes[channels_num + i].plot([pos], scores[:, i][pos], 'r*', markersize=7)

        axes[channels_num + i].set_ylabel(score_names[i])

    # Set x-ticks
    for ax in axes:
        ax.set_xticks(ticks)

    # Configure ticks labels
    xlabels = []
    for pos in axes[-1].get_xticks():
        c_time = start_time + pos / freq
        micro = c_time.strftime('%f')[:2]
        xlabels.append(c_time.strftime('%H:%M:%S') + f'.{micro}')

    axes[-1].set_xticklabels(xlabels)

    # Add date text
    date = start_time.strftime('%Y-%m-%d')
    fig.text(0.095, 1., date, va='center')

    # Finalize and save
    fig.tight_layout()
    fig.savefig(file_token + '.jpg')
    fig.clear()


def print_scores(data, scores, predictions, file_token, window_length=400):
    """
    Prints scores and waveforms.
    """
    right_shift = window_length // 2

    shapes = [d.data.shape[0] for d in data] + [scores.shape[0]]
    shapes = set(shapes)

    if len(shapes) != 1:
        raise AttributeError('All waveforms and scores must have similar length!')

    length = shapes.pop()

    waveforms = np.zeros((length, len(data)))
    for i, d in enumerate(data):
        waveforms[:, i] = d.data

    # Shift scores
    shifted_scores = np.zeros((length, len(data)))
    shifted_scores[right_shift:] = scores[:-right_shift]

    plot_wave_scores(file_token, waveforms, shifted_scores, data[0].stats.starttime, predictions,
                     right_shift=right_shift)

    np.save(f'{file_token}_wave.npy', waveforms)
    np.save(f'{file_token}_scores.npy', shifted_scores)
