import numpy as np
from obspy import read
import sys

from utils.script_args import archive_scan_params
import utils.scan_tools as stools
from utils.seisan import get_archives

# Silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    params = archive_scan_params()  # parse command line arguments

    if params['env', 'input']:
        archives = stools.parse_archive_csv(params['env', 'input'])  # parse archive names
    else:
        archives = get_archives(seisan=params['env', 'seisan'],
                                mulplt=params['env', 'mulplt'],
                                archives=params['env', 'archives'],
                                start=params['scan', 'start'], end=params['scan', 'end'])

    if params['info', 'cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Set label variables
    model_labels = {'p': 0, 's': 1, 'n': 2}
    positive_labels = {'p': 0, 's': 1}

    # Parse and validate thresholds
    threshold_labels = {}
    global_threshold = False
    if type(params['scan', 'threshold']) is str:

        split_thresholds = params['scan', 'threshold'].split(',')

        if len(split_thresholds) == 1:
            params['scan', 'threshold'] = float(params['scan', 'threshold'])
            global_threshold = True
        else:
            for split in split_thresholds:

                label_threshold = split.split(':')
                if len(label_threshold) != 2:
                    print('ERROR: Wrong --threshold format. Hint:'
                          ' --threshold "p: 0.95, s: 0.9901"', file=sys.stderr)
                    sys.exit(2)

                threshold_labels[label_threshold[0].strip()] = float(label_threshold[1])
    else:
        params['scan', 'threshold'] = float(params['scan', 'threshold'])
        global_threshold = True

    if global_threshold:
        for label in positive_labels:
            threshold_labels[label] = params['scan', 'threshold']
    else:
        positive_labels_error = False
        if len(positive_labels) != len(threshold_labels):
            positive_labels_error = True

        for label in positive_labels:
            if label not in threshold_labels:
                positive_labels_error = True

        if positive_labels_error:
            sys.stderr.write('ERROR: --threshold values do not match positive_labels.'
                             f' positive_labels contents: {[k for k in positive_labels.keys()]}')
            sys.exit(2)

    params['scan', 'threshold'] = threshold_labels

    # Set values
    half_duration = (params['model', 'features-number'] * 0.5) / params['scan', 'frequency']

    # Load model
    if params['model', 'model']:

        import importlib

        model_loader = importlib.import_module(params['model', 'model'])  # import loader module
        loader_call = getattr(model_loader, 'load_model')  # import loader function

        # Parse loader arguments
        loader_argv = params['model', 'loader-argv']

        argv_split = loader_argv.strip().split()
        argv_dict = {}

        for pair in argv_split:

            spl = pair.split('=')
            if len(spl) == 2:
                argv_dict[spl[0]] = spl[1]

        model = loader_call(**argv_dict)
    else:

        if params['model', 'cnn']:
            import utils.seismo_load as seismo_load
            model = seismo_load.load_cnn(params['model', 'weights'])
        elif params['model', 'gpd']:
            from utils.gpd_loader import load_model as load_gpd
            model = load_gpd(params['model', 'weights'])
        else:
            import utils.seismo_load as seismo_load
            model = seismo_load.load_performer(params['model', 'weights'])

    # Main loop
    total_performance_time = 0.
    for n_archive, l_archives in enumerate(archives):

        # Read data
        streams = []
        for path in l_archives:
            streams.append(read(path))

        # If --plot-positives-original, save original streams
        original_streams = None
        if params['info', 'plot-positives-original']:
            original_streams = []
            for path in l_archives:
                original_streams.append(read(path))

        # Pre-process data
        for st in streams:
            stools.pre_process_stream(st, params['scan', 'no-filter'], params['scan', 'no-detrend'])

        # Cut archives to the same length
        streams = stools.trim_streams(streams, params['scan', 'start'], params['model', 'end'])
        if original_streams:
            original_streams = stools.trim_streams(original_streams, params['scan', 'start'], params['scan', 'end'])

        # Check if stream traces number is equal
        lengths = [len(st) for st in streams]
        if len(np.unique(np.array(lengths))) != 1:
            continue

        n_traces = len(streams[0])

        # Progress bar preparations
        total_batch_count = 0
        for i in range(n_traces):
            traces = [st[i] for st in streams]

            l_trace = traces[0].data.shape[0]
            last_batch = l_trace % params['scan', 'trace-size']
            batch_count = l_trace // params['scan', 'trace-size'] + 1 \
                if last_batch \
                else l_trace // params['scan', 'trace-size']

            total_batch_count += batch_count

        # Predict
        current_batch_global = 0
        for i in range(n_traces):

            traces = stools.get_traces(streams, i)
            original_traces = None
            if original_streams:
                original_traces = stools.get_traces(original_streams, i)
                if traces[0].data.shape[0] != original_traces[0].data.shape[0]:
                    raise AttributeError('WARNING: Traces and original_traces have different sizes, '
                                         'check if preprocessing changes stream length!')

            # Determine batch count
            l_trace = traces[0].data.shape[0]
            last_batch = l_trace % params['scan', 'trace-size']
            batch_count = l_trace // params['scan', 'trace-size'] + 1 \
                if last_batch \
                else l_trace // params['scan', 'trace-size']

            freq = traces[0].stats.sampling_rate
            station = traces[0].stats.station

            for b in range(batch_count):

                detected_peaks = []

                b_size = params['scan', 'trace-size']
                if b == batch_count - 1 and last_batch:
                    b_size = last_batch

                start_pos = b * params['scan', 'trace-size']
                end_pos = start_pos + b_size
                t_start = traces[0].stats.starttime

                batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq) for trace in traces]
                original_batches = None
                if original_traces:
                    original_batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq)
                                        for trace in original_traces]

                # Progress bar

                if params['info', 'time']:
                    stools.progress_bar(current_batch_global / total_batch_count, 40, add_space_around=False,
                                        prefix=f'Group {n_archive + 1} out of {len(archives)} [',
                                        postfix=f'] - Batch: {batches[0].stats.starttime} '
                                                f'- {batches[0].stats.endtime} '
                                                f'Time: {total_performance_time:.6} seconds')
                else:
                    stools.progress_bar(current_batch_global / total_batch_count, 40, add_space_around=False,
                                        prefix=f'Group {n_archive + 1} out of {len(archives)} [',
                                        postfix=f'] - Batch: {batches[0].stats.starttime}'
                                                f' - {batches[0].stats.endtime}')
                current_batch_global += 1

                scores, performance_time = stools.scan_traces(*batches,
                                                              model=model,
                                                              params=params,
                                                              original_data=original_batches)  # predict
                total_performance_time += performance_time

                if scores is None:
                    continue

                restored_scores = stools.restore_scores(scores,
                                                        (len(batches[0]), len(model_labels)),
                                                        params['scan', 'shift'])

                # Get indexes of predicted events
                predicted_labels = {}
                for label in positive_labels:

                    other_labels = []
                    for k in model_labels:
                        if k != label:
                            other_labels.append(model_labels[k])

                    positives = stools.get_positives(restored_scores,
                                                     positive_labels[label],
                                                     other_labels,
                                                     threshold=threshold_labels[label])

                    predicted_labels[label] = positives

                # Convert indexes to datetime
                predicted_timestamps = {}
                for label in predicted_labels:

                    tmp_prediction_dates = []
                    for prediction in predicted_labels[label]:
                        starttime = batches[0].stats.starttime

                        # Get prediction UTCDateTime and model pseudo-probability
                        tmp_prediction_dates.append([starttime +
                                                     (prediction[0] / params['scan', 'frequency'])
                                                     + half_duration,
                                                     prediction[1]])

                    predicted_timestamps[label] = tmp_prediction_dates

                # Prepare output data
                for typ in predicted_timestamps:
                    for pred in predicted_timestamps[typ]:
                        prediction = {'type': typ,
                                      'datetime': pred[0],
                                      'pseudo-probability': pred[1]}

                        detected_peaks.append(prediction)

                if params['info', 'print-scores']:
                    stools.print_scores(batches, restored_scores, predicted_labels, f't{i}_b{b}')

                stools.print_results(detected_peaks, params['env', 'out'],
                                     precision=params['info', 'print-precision'], station=station)

            print('')

    if params['info', 'time']:
        print(f'Total model prediction time: {total_performance_time:.6} seconds')
