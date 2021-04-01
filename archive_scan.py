from utils import predict
from utils.ini import parse_ini, convert_params
from utils.seisan import parse_multplt, parse_seisan_def, date_str, archive_to_path

import importlib

from obspy.core.utcdatetime import UTCDateTime
from obspy import read

import numpy as np

import argparse

if __name__ == '__main__':

    # Set default parameters
    params = {'day_length': 60. * 60 * 24,
              'config': 'config.ini',
              'verbosity': 1,
              'frequency': 100.,
              'model_path': None,
              'weights_path': None,
              'start_date': None,
              'end_date': None}

    param_set = []

    # Specify essential parameter types
    param_types = {'frequency': float,
                   'model_labels': int,
                   'positive_labels': int,
                   'day_length': float,
                   'verbosity': int}

    # Params help messages
    param_helps = {'config': 'path to the config file, default: "config.ini"',
                   'verbosity': '0 - non verbose, 1 - verbose',
                   'frequency': 'base stream frequency, default: 100 (Hz)',
                   'output_file': 'output text file name, default: "out.txt"',
                   'multplt_path': 'path to MULTPLT.DEF file',
                   'seisan_path': 'path to SEISAN.DEF',
                   'model_path': 'path to model file, might be empty with in-code initialized models',
                   'weights_path': 'path to model weights file',
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
                                 'default: now'}

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

    # TODO: Check start_date and end_date params and set them if not provided
    # TODO: Add required arguments/parameters check, e.g. multplt_path, seisan_path, ...

    # Parse MULTPLT.DEF and SEISAN.DEF
    multplt_parsed = parse_multplt(params['multplt_path'])
    seisan_parsed = parse_seisan_def(params['seisan_path'],
                                     multplt_data = multplt_parsed,
                                     allowed_channels = params['allowed_channel_types'])

    # Load model
    try:
        model_loader = importlib.import_module(params['model_loader_module'])  # import loader module
        loader_call = getattr(model_loader, params['model_loader_name'])  # import loader function

        model = loader_call(params['model_path'], params['weights_path'])  # load model

    except ModuleNotFoundError as e:
        print(f'Cannot find module \'{params["model_loader_module"]}\' specified as \'model_loader_module\' parameter!')
    except AttributeError as e:
        print(f'Error while trying to access \'{params["model_loader_name"]}\' specified'
              f' as \'model_loader_name\' parameter: {e}')

    # Main loop
    # TODO: start_dt and end_dt make as loadable arguments
    start_dt = UTCDateTime(date_str(2014, 9, 28, 12, 3, 2.5))
    end_dt = UTCDateTime(date_str(2014, 10, 5, 12, 3, 2.5))

    current_dt = start_dt
    current_end_dt = None
    if end_dt.year == current_dt.year and end_dt.julday == current_dt.julday:
        current_end_dt = end_dt

    # FOR TESTING ONLY
    allowed_archives = [params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHE.2014.274',
                        params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHN.2014.274',
                        params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHZ.2014.274']

    detected_peaks = []  # TODO: maybe print detected peaks for every trace, not for the whole dataset?

    # TODO: Implement progress bar
    #  Disable keras verbose output
    while current_dt < end_dt:

        for archive_list in seisan_parsed:

            # Archives path and meta data
            archive_data = archive_to_path(archive_list, current_dt, params['archives_path'], params['channel_order'])

            # Read data
            streams = []

            # Data for detailed output
            streams_channels = {}  # .. channel type: [full channel, file name]

            for ch in params['channel_order']:

                if ch in archive_data:

                    if archive_data[ch] in allowed_archives:  # TODO: remove following condition
                        streams.append(read(archive_data[ch]))

                        channel = None
                        if ch in archive_data['meta']['channels']:
                            channel = archive_data['meta']['channels'][ch]

                        streams_channels[ch] = [channel, archive_data[ch]]

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
            for i in range(len(streams[0])):

                traces = [stream[i] for stream in streams]  # get traces
                scores = predict.scan_traces(*traces, model=model)  # predict
                restored_scores = predict.restore_scores(scores, (len(traces[0]), len(params['model_labels'])), 10)

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
                                                      min_threshold=0.95)

                    predicted_labels[label] = positives

                # Convert indexes to datetime
                predicted_timestamps = {}
                for label in predicted_labels:

                    tmp_prediction_dates = []
                    for prediction in predicted_labels[label]:
                        starttime = traces[0].stats.starttime

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

        # Get new dates
        current_dt += params['day_length']
        current_dt = UTCDateTime(date_str(current_dt.year, current_dt.month, current_dt.day))

        current_end_dt = None
        if end_dt.year == current_dt.year and end_dt.julday == current_dt.julday:
            current_end_dt = end_dt

    predict.print_results(detected_peaks, params['output_file'])
