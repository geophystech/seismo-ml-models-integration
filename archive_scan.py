import numpy as np
from obspy import read
import sys

from utils.script_args import archive_scan_params
import utils.scan_tools as stools
from utils.seisan import get_archives
from utils.progress_bar import ProgressBar

# Silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def init_progress_bar(char_length=60, char_empty='.', char_fill='=', char_point='>'):

    progress_bar = ProgressBar()

    progress_bar.set_length(char_length)

    progress_bar.set_empty_character(char_empty)
    progress_bar.set_progress_character(char_fill)
    progress_bar.set_current_progress_char(char_point)

    progress_bar.set_prefix_expression('{archive} out of {total_archives} ({station}) [')
    progress_bar.set_postfix_expression('] - Batch: {start} - {end}')

    progress_bar.set_max(archives=len(archives), traces=1., batches=1., inter=1.)

    return progress_bar


def get_model_ref(data, name_weights):
    """
    Returns model reference if found by model name and model weights pair. Returns None otherwise.
    data - list of tuples (model_name, model_weights_path, model_ref)
    """
    for x in data:
        if name_weights == x[:2]:
            return x[2]
    return None


if __name__ == '__main__':

    params = archive_scan_params()  # parse command line arguments

    if params['main', 'input']:
        archives = stools.parse_archive_csv(params['main', 'input'])  # parse archive names
    else:
        archives = get_archives(seisan=params['main', 'seisan'],
                                mulplt=params['main', 'mulplt'],
                                archives=params['main', 'archives'],
                                params=params)

    if params['main', 'cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Set values
    model_labels = {'p': 0, 's': 1, 'n': 2}
    positive_labels = {'p': 0, 's': 1}
    half_duration = (params['main', 'features-number'] * 0.5) / params['main', 'frequency']

    # Load model(s)
    models_data = []
    for x in params.get_station_keys(main=True):

        model = get_model_ref(models_data, (params[x, 'model-name'], params[x, 'weights']))
        if model:
            params[x, 'model-object'] = model
            continue

        if params[x, 'model-name'] == 'custom-model':

            import importlib

            model_loader = importlib.import_module(params[x, 'model'])  # import loader module
            loader_call = getattr(model_loader, 'load_model')  # import loader function

            # Parse loader arguments
            loader_argv = params[x, 'loader-argv']

            argv_split = loader_argv.strip().split()
            argv_dict = {}

            for pair in argv_split:

                spl = pair.split('=')
                if len(spl) == 2:
                    argv_dict[spl[0]] = spl[1]

            model = loader_call(**argv_dict)
        else:

            if params[x, 'model-name'] == 'cnn':
                import utils.seismo_load as seismo_load
                model = seismo_load.load_cnn(params[x, 'weights'])
            elif params[x, 'model-name'] == 'gpd':
                from utils.gpd_loader import load_model as load_gpd
                model = load_gpd(params[x, 'weights'])
            elif params[x, 'model-name'] == 'favor':
                import utils.seismo_load as seismo_load
                model = seismo_load.load_performer(params[x, 'weights'])
            else:
                raise AttributeError('"model-name" is not specified correctly! If you see this message this is a bug!')

        params[x, 'model-object'] = model
        models_data.append((params[x, 'model-name'], params[x, 'weights'], model))

    # Main loop
    progress_bar = init_progress_bar()
    progress_bar.set_prefix_arg('total_archives', len(archives))
    total_performance_time = 0.

    if params['main', 'print-files']:
        for n_archive, d_archives in enumerate(archives):
            print(f'{n_archive}: {d_archives["paths"]}')
        print()

    for n_archive, d_archives in enumerate(archives):

        # Unpack
        station = d_archives['station']
        l_archives = d_archives['paths']

        # Update progress bar parameters
        progress_bar.set_progress(n_archive, level='archives')
        progress_bar.set_prefix_arg('archive', n_archive + 1)
        if station:
            progress_bar.set_prefix_arg('station', station)
        else:
            progress_bar.set_prefix_arg('station', 'none')

        # Read data
        try:
            streams = []
            for path in l_archives:
                streams.append(read(path))
        except FileNotFoundError:
            continue

        # If --plot-positives-original, save original streams
        original_streams = None
        if params[station, 'plot-positives-original']:
            original_streams = []
            for path in l_archives:
                original_streams.append(read(path))

        # Pre-process data
        for st in streams:
            stools.pre_process_stream(st, params, station)

        # Cut archives to the same length
        streams = stools.trim_streams(streams, params['main', 'start'], params['main', 'end'])
        if original_streams:
            original_streams = stools.trim_streams(original_streams, params['main', 'start'], params['main', 'end'])

        # Check if stream traces number is equal
        lengths = [len(st) for st in streams]
        if len(np.unique(np.array(lengths))) != 1:
            continue

        n_traces = len(streams[0])

        # Update progress bar params
        progress_bar.change_max('traces', n_traces)
        progress_bar.set_progress(0, level='traces')
        # Progress bar preparations
        total_batch_count = 0
        for i in range(n_traces):

            progress_bar.set_progress(i, level='traces')
            traces = [st[i] for st in streams]

            l_trace = traces[0].data.shape[0]
            last_batch = l_trace % params['main', 'trace-size']
            batch_count = l_trace // params['main', 'trace-size'] + 1 \
                if last_batch \
                else l_trace // params['main', 'trace-size']

            total_batch_count += batch_count

        # Predict
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
            last_batch = l_trace % params['main', 'trace-size']
            batch_count = l_trace // params['main', 'trace-size'] + 1 \
                if last_batch \
                else l_trace // params['main', 'trace-size']

            freq = traces[0].stats.sampling_rate

            # Update progress bar parameters
            progress_bar.change_max('batches', batch_count)
            progress_bar.set_progress(0, level='batches')

            for b in range(batch_count):

                progress_bar.set_progress(b, level='batches')

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
                progress_bar.set_postfix_arg('start', batches[0].stats.starttime)
                progress_bar.set_postfix_arg('end', batches[0].stats.endtime)
                progress_bar.print()

                try:
                    scores, performance_time = stools.scan_traces(*batches,
                                                                  params=params,
                                                                  station=station,
                                                                  original_data=original_batches)
                except ValueError:
                    scores, performance_time = None, 0

                total_performance_time += performance_time

                if scores is None:
                    continue

                restored_scores = stools.restore_scores(scores,
                                                        (len(batches[0]), len(model_labels)),
                                                        params[station, 'shift'])

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
                                                     threshold=params[station, 'threshold'][label])

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

                if params['main', 'print-scores']:
                    stools.print_scores(batches, restored_scores, predicted_labels, f't{i}_b{b}')

                stools.print_results(detected_peaks, params[station, 'out'],
                                     precision=params[station, 'print-precision'], station=station)
    print('')
    if params['main', 'time']:
        print(f'Total model prediction time: {total_performance_time:.6} seconds')
