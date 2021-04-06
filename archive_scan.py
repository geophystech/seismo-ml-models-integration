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

if __name__ == '__main__':
    # Set default parameters
    params = {'day_length': 60. * 60 * 24,
              'default_start_shift': 60 * 60 * 35,
              'config': 'config.ini',
              'verbosity': 1,
              'frequency': 100.,
              'model_path': None,
              'weights_path': None,
              'start_date': None,
              'end_date': None,
              'verbose': 1,
              'threshold': 0.98}

    param_set = []

    # Specify essential parameter types
    param_types = {'frequency': float,
                   'model_labels': int,
                   'positive_labels': int,
                   'day_length': float,
                   'verbose': int}

    # Params help messages
    param_helps = {'config': 'path to the config file, default: "config.ini"',
                   'verbose': '0 - non verbose, 1 - verbose',
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
    try:
        model_loader = importlib.import_module(params['model_loader_module'])  # import loader module
        loader_call = getattr(model_loader, params['model_loader_name'])  # import loader function

        model = loader_call(params['model_path'], params['weights_path'])  # load model

    except ModuleNotFoundError as e:
        print(f'Cannot find module \'{params["model_loader_module"]}\' specified as \'model_loader_module\' parameter!')
        sys.exit(1)
    #except AttributeError as e:
        #print(f'Error while trying to access \'{params["model_loader_name"]}\' specified'
         #     f' as \'model_loader_name\' parameter: {e}')
        #sys.exit(1)
    # except Exception as e:
        # print(f'Error while trying to access \'{params["model_loader_name"]}\' specified'
              # f' as \'model_loader_name\' parameter: {e}')
        # raise

    # Main loop
    current_dt = start_date
    current_end_dt = None
    if end_date.year == current_dt.year and end_date.julday == current_dt.julday:
        current_end_dt = end_date

    # FOR TESTING ONLY
    # TODO: disable this when done with everything else.
    # allowed_archives = [params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHE.2014.274',
                        # params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHN.2014.274',
                        # params['archives_path'] + '/IM/ARGI/ARGI.IM.00.SHZ.2014.274']

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

            for ch in params['channel_order']:

                if ch in archive_data:

                    # if archive_data[ch] in allowed_archives:  # TODO: remove this condition
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
                scores = predict.scan_traces(*traces, model=model)  # predict

                if scores is None:
                    continue

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
                                                      min_threshold = params['threshold'])

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

                predict.print_results(detected_peaks, params['output_file'])

                if len(detected_peaks) > 0:
                    # plot_n += 1
                    for x in detected_peaks:
                         
                        f_name = f'{x["datetime"].strftime("%d_%m__%H_%M_%S")}.jpeg'
                        plot_n += 1
                        traces[0].plot(outfile = f'0_{f_name}')
                        traces[1].plot(outfile = f'1_{f_name}')
                        traces[2].plot(outfile = f'2_{f_name}')
                        # plt.savefig(f_name)
                        # plt.clf()
                    

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
