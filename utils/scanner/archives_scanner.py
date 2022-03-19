import sys
from obspy import read
from time import time

from ..progress_bar import ProgressBar
from ..print_tools import plot_wave
import utils.scan_tools as stools
import utils.seisan as seisan


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


def archive_scan(archives, params, input_mode=False, advanced=False):
    """

    :param advanced:
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

    all_positives = []
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
        archive_start = d_archives['start'] if 'start' in d_archives else params['main', 'start']
        archive_end = d_archives['end'] if 'end' in d_archives else params['main', 'end']

        # Replace parameters for advanced search
        if advanced:
            if input_mode:
                original_threshold = params['main', 'threshold']
                original_shift = params['main', 'shift']
                params['main', 'threshold'] = params['main', 'advanced-search-threshold']
                params['main', 'shift'] = params['main', 'advanced-search-shift']
            else:
                original_threshold = params[station_name, 'threshold']
                original_shift = params[station_name, 'shift']
                params[station_name, 'threshold'] = params[station_name, 'advanced-search-threshold']
                params[station_name, 'shift'] = params[station_name, 'advanced-search-shift']

        # Update progress bar parameters
        progress_bar.set_prefix_arg('archive', n_archive + 1)
        progress_bar.set_prefix_arg('station', station_name)

        # Read data
        streams = []
        for path in l_archives:
            streams.append(read(path))

        # Pre-process data
        for st in streams:
            stools.pre_process_stream(st, params, station_name)

        # Cut archives to the same length
        if input_mode:
            streams = stools.trim_streams(streams, station_name)
        else:
            streams = stools.trim_streams(streams, station_name, archive_start, archive_end)

        if not streams:
            if station_name:
                print(f'\nSkipping station: {station_name}: no data in specified time span!', file=sys.stderr)
            else:
                print(f'\nSkipping archives: {d_archives}: no data in specified time span!', file=sys.stderr)
            continue

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

                b_start = t_start + start_pos / freq
                b_end = t_start + end_pos / freq
                batches = [trace.slice(b_start, b_end) for trace in traces]

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

                if params[station_name, 'plot-batches']:
                    save_name = 'batch_'
                    if advanced:
                        save_name += 'advanced_'
                    if station_name:
                        save_name += station_name + '_'
                    save_name += b_start.strftime("%Y-%m-%d_%H:%M:%S") + '__' + b_end.strftime("%Y-%m-%d_%H:%M:%S")
                    plot_wave(batches, save_name)

                try:
                    scores, performance_time = stools.scan_traces(*batches,
                                                                  params=params,
                                                                  station=station_name)
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

                if params[station_name, 'plot-scores']:
                    save_name = 'scores_'
                    if advanced:
                        save_name += 'advanced_'
                    if station_name:
                        save_name += station_name + '_'
                    save_name += b_start.strftime("%Y-%m-%d_%H:%M:%S") + '__' + b_end.strftime("%Y-%m-%d_%H:%M:%S")
                    plot_wave(restored_scores, save_name)

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
                else:
                    last_saved_station = station_name

                if input_mode:
                    for x in detected_peaks:
                        x['input'] = l_archives
                else:
                    for x in detected_peaks:
                        x['station'] = station
                        x['input'] = l_archives

                all_positives.extend(detected_peaks)

        # Return original parameter values after advanced search
        if advanced:
            if input_mode:
                params['main', 'threshold'] = original_threshold
                params['main', 'shift'] = original_shift
            else:
                params[station_name, 'threshold'] = original_threshold
                params[station_name, 'shift'] = original_shift

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


def advanced_search(events, params, input_mode=False):
    # Generate list of archives/timespans to search
    advanced_search_list = []
    search_range = int(params['main', 'advanced-search-range'])
    if params['main', 'advanced-search-all-stations']:
        for event in events:
            dt = event['datetime']
            detections_count = len(event['detections'])
            start = dt - search_range
            end = dt + search_range
            archives = seisan.get_archives_advanced(dt, params)
            search_list = []
            for archive in archives:
                search_list.append({
                    'paths': archive['paths'],
                    'station': archive['station'],
                    'start': start,
                    'end': end,
                })
            advanced_search_list.append({
                'datetime': dt,
                'search_list': search_list,
                'original_detections_count': detections_count,
            })
    else:
        for event in events:
            dt = event['datetime']
            detections_count = len(event['detections'])
            start = dt - search_range
            end = dt + search_range
            unique_archives = []
            search_list = []
            for detection in event['detections']:
                if detection['input'] not in unique_archives:
                    unique_archives.append(detection['input'])
                    search_list.append({
                        'paths': detection['input'],
                        'station': detection['station'],
                        'start': start,
                        'end': end,
                    })
            advanced_search_list.append({
                'datetime': dt,
                'search_list': search_list,
                'original_detections_count': detections_count,
            })

    # Advanced search
    advanced_events = []
    for search_item in advanced_search_list:
        archives = search_item['search_list']
        dt = search_item['datetime']
        print(f'\nPerforming advanced search for event at {dt.strftime("%Y-%m-%d %H:%M:%S")}..')
        all_positives, performance = archive_scan(archives, params, input_mode=input_mode, advanced=True)

        if params['main', 'advanced-search-combine']:
            current_advanced_events = stools.combine_detections_single_event(all_positives,
                                                                             original_detections_count=
                                                                             search_item['original_detections_count'])
        else:
            current_advanced_events = stools.combine_detections(all_positives, params, input_mode=input_mode)

        advanced_events.extend(current_advanced_events)

    return advanced_events
