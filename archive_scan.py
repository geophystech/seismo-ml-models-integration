from utils import predict
from utils.ini import parse_ini, convert_params
from utils.seisan import parse_multplt, parse_seisan_def, date_str, archive_to_path
from utils.print_tools import progress_bar

import importlib

from obspy.core.utcdatetime import UTCDateTime
from obspy import read

import numpy as np

import argparse
import sys

import matplotlib.pyplot as plt

# TODO: Fix this:
#      File "archive_scan.py", line 275, in <module>
#        scores = predict.scan_traces(*traces, model=model)  # predict
#      File "/opt/seisan/WOR/chernykh/dev/seismo-ml-models-integration/utils/predict.py", line 120, in scan_traces
#        windows = np.zeros((w_length, n_features, len(l_windows)))
#      MemoryError: Unable to allocate 7.72 GiB for an array with shape (863947, 400, 3) and data type float64

# TODO: adding sesmo model requires `pip install scikit-learn`: i either should update requirements.txt or
#       browse seismo_transformer.py and find where it is used and disable it, because it is probably
#       used for some train/test splitting, not for an actual model initialization.
#       Also requires: `pip install einops`

# TODO: i now batch the traces to save memory, but i should try to overlap this batches maybe..

if __name__ == '__main__':
    # Set default parameters
    params = {'day_length': 60. * 60 * 24,
              'default_start_shift': 60 * 60 * 35,
              'config': 'config.ini',
              'verbosity': 1,
              'frequency': 100.,
              'batch_size': 500000,
              'model_path': None,
              'weights_path': None,
              'start_date': None,
              'end_date': None,
              'verbose': 1,
              'threshold': 0.95,
              'plot_path': None,
              'debug': 0}

    param_set = []

    # Specify essential parameter types
    param_types = {'frequency': float,
                   'threshold': float,
                   'model_labels': int,
                   'positive_labels': int,
                   'day_length': float,
                   'verbose': int,
                   'debug': int,
                   'batch_size': int}

    # Params help messages
    param_helps = {'config': 'path to the config file, default: "config.ini"',
                   'verbose': '0 - non verbose, 1 - verbose, default: 1',
                   'debug': '0 - do not print debug info, 1 - print debug info, defalut: 0',
                   'frequency': 'base stream frequency, default: 100 (Hz)',
                   'output_file': 'output text file name, default: "out.txt"',
                   'multplt_path': 'path to MULTPLT.DEF file',
                   'seisan_path': 'path to SEISAN.DEF',
                   'model_path': 'path to model file, might be empty with in-code initialized models',
                   'weights_path': 'path to model weights file',
                   'plot_path': 'path for plot saving (do not specify to disable plots)',
                   'start_date': 'start date in ISO 8601 format:\n'
                                 '{year}-{month}-{day}T{hour}:{minute}:{second}.{microsecond}\n'
                                 'or\n'
                                 '{year}-{month}-{day}T{hour}:{minute}:{second}\n'
                                 'or\n'
                                 '{year}-{month}-{day}\n'
                                 'default: yesterday midnight',
                   'end_date': 'end date in ISO 8601 format:\n'
                               '{year}-{month}-{day}T{hour}:{minute}:{second}.{microsecond}\n'
                               'or\n'
                               '{year}-{month}-{day}T{hour}:{minute}:{second}\n'
                               'or\n'
                               '{year}-{month}-{day}\n'
                               'default: now',
                   'threshold': 'model prediction threshold'}

    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    for k in param_helps:

        parser.add_argument(f'--{k}',
                            help = param_helps[k])

    args = parser.parse_args()
    args = vars(args)

    for k in args:

        if args[k] is not None:
            params[k] = args[k]
            param_set.append(k)

    # Parse config file
    params = parse_ini(params['config'], params, param_set = param_set)  # load config
    convert_params(params, param_types)  # and convert types

    # TODO: add setup if no config specified. Some of it already stashed in git.

    # TODO: improve plotting: limit to params defined amount of seconds around the peak, highlight peak
    # TODO: check all path parameters for validity.
    #           also add / to all directory path parameters if missing.

    # Set start and end date
    def parse_date_param(params_dict, p_name):
        """
        Parse parameter from dictionary to UTCDateTime type.
        """
        if p_name not in params_dict:
            return None
        if params_dict[p_name] is None:
            return None

        try:
            return UTCDateTime(params_dict[p_name])
        except TypeError as e:
            print(f'Failed to parse "{p_name}" parameter (value: {params_dict[p_name]}).'
                  f' Use {__file__} -h for date format information.')
            sys.exit(1)
        except Exception as e:
            print(f'Failed to parse "{p_name}" parameter (value: {params_dict[p_name]}).'
                  f' Use {__file__} -h for date format information.')
            raise

    end_date = parse_date_param(params, 'end_date')
    start_date = parse_date_param(params, 'start_date')

    # Default start and/or end dates:
    if end_date is None:
        end_date = UTCDateTime()
    if start_date is None:
        start_date = UTCDateTime() - params['default_start_shift']

    # TODO: Add required arguments/parameters check, e.g. multplt_path, seisan_path, ...

    # Parse MULTPLT.DEF and SEISAN.DEF
    multplt_parsed = parse_multplt(params['multplt_path'])
    seisan_parsed = parse_seisan_def(params['seisan_path'],
                                     multplt_data = multplt_parsed,
                                     allowed_channels = params['allowed_channel_types'])

    # Load model
    # TODO: made parameters of the function through the array and unpack them to model loader.
    # TODO: make 4 separate config files for each model/weights
    # TODO: test them, fix bug, do PR
    try:
        model_loader = importlib.import_module(params['model_loader_module'])  # import loader module
        loader_call = getattr(model_loader, params['model_loader_name'])  # import loader function

        model = loader_call(params['model_path'], params['weights_path'])  # load model

    except ModuleNotFoundError as e:
        print(f'Cannot find module \'{params["model_loader_module"]}\' specified as \'model_loader_module\' parameter!')
        sys.exit(1)
    except AttributeError as e:
        print(f'Error while trying to access \'{params["model_loader_name"]}\' specified'
              f' as \'model_loader_name\' parameter: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'Error while trying to access \'{params["model_loader_name"]}\' specified'
              f' as \'model_loader_name\' parameter: {e}')
        raise

    # Main loop
    current_dt = start_date
    current_end_dt = None
    if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
        current_end_dt = end_date

    # FOR TESTING ONLY
    # TODO: disable this when done with everything else.
    # 2014.10.01
    allowed_archives = [params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHE.2014.274',
                        params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHN.2014.274',
                        params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHZ.2014.274']

    # TODO: Add timestamp print if debug build
    if params['debug'] > 0:
        print(f'DEBUG: start_date = {start_date}',
              f'DEBUG: end_date = {end_date}', sep = '\n')

    detected_peaks = []  # TODO: maybe print detected peaks for every trace, not for the whole dataset?

    plot_n = 0

    while current_dt < end_date:

        stream_count = 0  # .. for progress bar info

        for archive_list in seisan_parsed:

            stream_count += 1

            # Archives path and meta data
            archive_data = archive_to_path(archive_list, current_dt, params['archives_path'], params['channel_order'])

            # Read data
            streams = []

            # Data for detailed output
            streams_channels = {}  # .. channel type: [full channel, file name]

            try:
                for ch in params['channel_order']:

                    if ch in archive_data:

                        # if archive_data[ch] in allowed_archives:  # TODO: remove this condition
                        streams.append(read(archive_data[ch]))

                        channel = None
                        if ch in archive_data['meta']['channels']:
                            channel = archive_data['meta']['channels'][ch]

                        streams_channels[ch] = [channel, archive_data[ch]]
            except FileNotFoundError as e:
                # TODO: log this in warnings!
                continue

            if len(streams) != len(params['channel_order']):
                continue

            # Cut data
            cut_streams = []
            for st in streams:
                cut_streams.append(st.slice(current_dt, current_end_dt))

            streams = cut_streams
            del cut_streams

            # Pre-process data
            for st in streams:
                predict.pre_process_stream(st, params['frequency'])

            # Cut data to the same length
            max_start_time = None
            min_end_time = None

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
            for st in streams:
                cut_streams.append(st.slice(max_start_time, min_end_time))

            streams = cut_streams
            del cut_streams

            # Check if stream traces number is equal
            lengths = [len(x) for x in streams]
            if len(np.unique(np.array(lengths))) != 1:
                continue

            # Process data
            total_streams = len(streams[0])
            for i in range(total_streams):

                detected_peaks = []

                # TODO: Add info about which stream out of X
                if params['verbose'] > 0:

                    meta = archive_data['meta']

                    prefix = f'{current_dt.strftime("%d.%m.%y")} ' \
                             f'[{stream_count} archive out of {len(seisan_parsed)}] : ['
                    postfix = f'] ' \
                              f'{meta["station"]}.' \
                              f'{meta["network_code"]}' \
                              f'.{meta["location_code"]}'

                    progress_bar(i / total_streams, 40, add_space_around = False,
                                 prefix = prefix, postfix = postfix)

                traces = [stream[i] for stream in streams]  # get traces

                # TODO: add trace data batching
                # params['batch_size']

                start_time = max([trace.stats.starttime for trace in traces])
                end_time = min([trace.stats.endtime for trace in traces])

                # TODO: trim all traces

                for j in range(len(traces)):
                    traces[j] = traces[j].slice(start_time, end_time)

                # TODO: Move everything into another inner loop
                trace_length = traces[0].data.shape[0]
                last_batch = trace_length % params['batch_size']
                batch_count = trace_length // params['batch_size'] + 1 \
                    if last_batch \
                    else trace_length // params['batch_size']
                freq = traces[0].stats.sampling_rate

                # TODO: Do this loop through and generator object that will return batch_size for every iteration
                #       for b_size in batch_size_generator:
                #       google how other generators work (range, etc.) and how to write new one.
                for b in range(batch_count):

                    b_size = params['batch_size']
                    if b == batch_count - 1 and last_batch:
                        b_size = last_batch

                    start_pos = b * params['batch_size']
                    end_pos = start_pos + b_size
                    t_start = traces[0].stats.starttime

                    batches = [trace.slice(t_start + start_pos / freq, t_start + end_pos / freq) for trace in traces]

                    scores = predict.scan_traces(*batches, model=model)  # predict

                    if scores is None:
                        continue

                    restored_scores = predict.restore_scores(scores, (len(batches[0]), len(params['model_labels'])), 10)

                    # Get indexes of predicted events
                    predicted_labels = {}
                    for label in params['positive_labels']:

                        other_labels = []
                        for k in params['model_labels']:
                            if k != label:
                                other_labels.append(params['model_labels'][k])

                        positives = predict.get_positives(restored_scores,
                                                          params['positive_labels'][label],
                                                          other_labels,
                                                          min_threshold = params['threshold'])

                        predicted_labels[label] = positives

                    # Convert indexes to datetime
                    predicted_timestamps = {}
                    for label in predicted_labels:

                        tmp_prediction_dates = []
                        for prediction in predicted_labels[label]:
                            starttime = batches[0].stats.starttime

                            # Get prediction UTCDateTime and model pseudo-probability
                            tmp_prediction_dates.append([starttime + (prediction[0] / params['frequency']), prediction[1]])

                        predicted_timestamps[label] = tmp_prediction_dates

                    # Prepare output data
                    for typ in predicted_timestamps:
                        for pred in predicted_timestamps[typ]:

                            prediction = {'type': typ,
                                          'datetime': pred[0],
                                          'pseudo-probability': pred[1],
                                          'channels': streams_channels,
                                          'station': archive_data['meta']['station'],
                                          'location_code': archive_data['meta']['location_code'],
                                          'network_code': archive_data['meta']['network_code']}

                            detected_peaks.append(prediction)

                    predict.print_results(detected_peaks, params['output_file'])

                    # TODO: write function which plots results, it takes plot path, traces array and predicted_timestamps
                    if params['plot_path']:
                        predict.plot_results(detected_peaks, traces, params['plot_path'])

                    # TODO: Print detected peaks after done with the archive. Append them to output file.
                    #   Open file only when needed.
                    #   Catch file open exceptions, track exception on file occupied if this one even exists.
                    #   Create file with <file>.n (when n - first number available) and write into it.

        # Get new dates
        current_dt += params['day_length']
        current_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day))

        current_end_dt = None
        if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
            current_end_dt = end_date

        # predict.print_results(detected_peaks, params['output_file'])

    # TODO: Sort detected peaks by datetime
    #  Or maybe they are already sorted by the design of how algorithm is working?
    # predict.print_results(detected_peaks, params['output_file'])

    if params['verbose'] > 0:
        print('\n', end = '')
