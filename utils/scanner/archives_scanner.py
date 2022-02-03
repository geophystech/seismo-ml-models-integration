import sys
from obspy import read
from time import time

from ..progress_bar import ProgressBar
import utils.scan_tools as stools


def init_progress_bar(char_length=30, char_empty='.', char_fill='=', char_point='>', use_station=False):

    progress_bar = ProgressBar()

    progress_bar.set_length(char_length)

    progress_bar.set_empty_character(char_empty)
    progress_bar.set_progress_character(char_fill)
    progress_bar.set_current_progress_char(char_point)

    if use_station:
        progress_bar.set_prefix_expression('{archive} out of {total_archives} ({station}) [')
    else:
        progress_bar.set_prefix_expression('{archive} out of {total_archives} [')
    progress_bar.set_postfix_expression('] - Batch: {start} - {end}')

    progress_bar.set_max(data=1., inter=1.)

    return progress_bar


def archive_scan(archives, params, input_mode=False):
    """

    :param archives:
    :param params:
    :param input_mode:
    :return:
    """
    if input_mode:
        progress_bar = init_progress_bar()
    else:
        progress_bar = init_progress_bar(use_station=True)
    progress_bar.set_prefix_arg('total_archives', len(archives))

    # Performance time tracking
    total_performance_time = 0.
    archives_time = []
    archives_walltime = []
    batch_time = []

    all_positives = {}
    for n_archive, d_archives in enumerate(archives):

        # Time tracking
        if params['main', 'time-archive']:
            current_archive_time = {
                'archives': d_archives,
                'time': None
            }
        if params['main', 'walltime-archive']:
            current_archive_walltime = {
                'archives': d_archives,
                'time': None,
                'start-time': time()
            }
        if params['main', 'time-batch']:
            current_archive_batch_time = {
                'archives': d_archives,
                'batches': [],
            }

        # Unpack
        station = d_archives['station']
        if not station:
            station_name = None
        else:
            station_name = station['station']
        l_archives = d_archives['paths']

        # Update progress bar parameters
        progress_bar.set_prefix_arg('archive', n_archive + 1)
        progress_bar.set_prefix_arg('station', station_name)

        # Read data
        streams = []
        for path in l_archives:
            streams.append(read(path))

        # If --plot-positives-original, save original streams
        original_streams = None
        if params[station_name, 'plot-positives-original']:
            original_streams = []
            for path in l_archives:
                original_streams.append(read(path))

        # Pre-process data
        for st in streams:
            stools.pre_process_stream(st, params, station_name)

        # Cut archives to the same length
        if input_mode:
            streams = stools.trim_streams(streams, station_name)
        else:
            streams = stools.trim_streams(streams, station_name, params['main', 'start'], params['main', 'end'])

        if not streams:
            if station_name:
                print(f'\nSkipping station: {station_name}: no data in specified time span!', file=sys.stderr)
            else:
                print(f'\nSkipping archives: {d_archives}: no data in specified time span!', file=sys.stderr)
            continue
        if original_streams:
            original_streams = stools.trim_streams(original_streams, params['main', 'start'], params['main', 'end'])

        # Check if stream traces number is equal
        traces_groups, total_data_length = stools.combined_traces(streams, params)
        total_data_progress = 0

        if params.data['invalid_combined_traces_groups']:
            print(f'\nWARNING: invalid combined traces groups detected: '
                  f'{params.data["invalid_combined_traces_groups"]}', file=sys.stderr)

        # Update progress bar params
        progress_bar.change_max('data', total_data_length)
        progress_bar.set_progress(0, level='data')

        # Predict
        last_saved_station = None
        for i, traces in enumerate(traces_groups):

            progress_bar.set_progress(i, level='traces')

            original_traces = None
            if original_streams:
                original_traces = traces
                if traces[0].data.shape[0] != original_traces[0].data.shape[0]:
                    continue

            # Determine batch count
            l_trace = traces[0].data.shape[0]
            last_batch = l_trace % params['main', 'trace-size']
            batch_count = l_trace // params['main', 'trace-size'] + 1 \
                if last_batch \
                else l_trace // params['main', 'trace-size']

            freq = traces[0].stats.sampling_rate

            # Update progress bar parameters
            progress_bar.change_max('batches', batch_count)
            progress_bar.set_progress(0, level='batches')

            for b in range(batch_count):

                detected_peaks = []

                b_size = params['main', 'trace-size']
                if b == batch_count - 1 and last_batch:
                    b_size = last_batch

                start_pos = b * params['main', 'trace-size']
                end_pos = start_pos + b_size
                t_start = traces[0].stats.starttime

                batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq) for trace in traces]
                original_batches = None
                if original_traces:
                    original_batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq)
                                        for trace in original_traces]

                # Progress bar
                s_batch_start_time = batches[0].stats.starttime.strftime("%Y-%m-%d %H:%M:%S")
                s_batch_end_time = batches[0].stats.endtime.strftime("%Y-%m-%d %H:%M:%S")
                progress_bar.set_postfix_arg('start', s_batch_start_time)
                progress_bar.set_postfix_arg('end', s_batch_end_time)
                total_data_progress += len(batches[0])
                progress_bar.set_progress(total_data_progress, level='data')
                progress_bar.print()

                if params['main', 'time-batch']:
                    current_batch_time = {
                        'id': f'{s_batch_start_time} .. {s_batch_end_time}',
                    }

                try:
                    scores, performance_time = stools.scan_traces(*batches,
                                                                  params=params,
                                                                  station=station_name,
                                                                  original_data=original_batches)
                except ValueError:
                    scores, performance_time = None, 0

                total_performance_time += performance_time
                if params['main', 'time-archive']:
                    if current_archive_time['time'] is None:
                        current_archive_time['time'] = 0
                    current_archive_time['time'] += performance_time
                if params['main', 'time-batch']:
                    current_batch_time['time'] = performance_time
                    current_archive_batch_time['batches'].append(current_batch_time)

                if scores is None:
                    continue

                restored_scores = stools.restore_scores(scores,
                                                        (len(batches[0]), len(params['main', 'model-labels'])),
                                                        params[station_name, 'shift'])

                # Get indexes of predicted events
                predicted_labels = {}
                for label in params['main', 'positive-labels']:

                    other_labels = []
                    for k in params['main', 'model-labels']:
                        if k != label:
                            other_labels.append(params['main', 'model-labels'][k])

                    positives = stools.get_positives(restored_scores,
                                                     params['main', 'positive-labels'][label],
                                                     other_labels,
                                                     threshold=params[station_name, 'threshold'][label])

                    predicted_labels[label] = positives

                # Convert indexes to datetime
                predicted_timestamps = {}
                for label in predicted_labels:

                    tmp_prediction_dates = []
                    for prediction in predicted_labels[label]:
                        starttime = batches[0].stats.starttime

                        # Get prediction UTCDateTime and model pseudo-probability
                        tmp_prediction_dates.append([starttime +
                                                     (prediction[0] / params['main', 'frequency'])
                                                     + params['main', 'half-duration'],
                                                     prediction[1]])

                    predicted_timestamps[label] = tmp_prediction_dates

                # Prepare output data
                for typ in predicted_timestamps:
                    for pred in predicted_timestamps[typ]:
                        prediction = {'type': typ,
                                      'datetime': pred[0],
                                      'pseudo-probability': pred[1]}

                        detected_peaks.append(prediction)

                if params['main', 'print-scores']:
                    stools.print_scores(batches, restored_scores, predicted_labels, f't{i}_b{b}')

                stools.print_results(detected_peaks, params, station_name, last_station=last_saved_station)

                # Save extensive station information for every detection for later output!
                if input_mode:
                    last_saved_station = None
                    str_archives = ';'.join(l_archives)
                    if str_archives not in all_positives:
                        all_positives[str_archives] = []

                    for x in detected_peaks:
                        x['input'] = str_archives

                    all_positives[str_archives].extend(detected_peaks)
                else:
                    last_saved_station = station_name
                    if station_name not in all_positives:
                        all_positives[station_name] = []

                    for x in detected_peaks:
                        x['station'] = station

                    all_positives[station_name].extend(detected_peaks)

        if params['main', 'time-archive']:
            archives_time.append(current_archive_time)
        if params['main', 'walltime-archive']:
            current_archive_walltime['time'] = time() - current_archive_walltime['start-time']
            archives_walltime.append(current_archive_walltime)
        if params['main', 'time-batch']:
            batch_time.append(current_archive_batch_time)

    performance = {
        'total-performance-time': total_performance_time,
        'archives-time': archives_time,
        'archives-walltime': archives_walltime,
        'batch-time': batch_time,
    }
    return all_positives, performance
