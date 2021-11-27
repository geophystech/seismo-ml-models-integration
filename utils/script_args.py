import argparse
from obspy import UTCDateTime
import sys
from os.path import isfile, isdir

from utils.params import Params, applied_function


def get_unsupported_station_parameters_list():
    # Not currently supported parameters:
    return [
        # Customizable models
        'model', 'loader-argv', 'features-number',
        # Custom dates
        'start', 'end',
        # Model input and batches customization
        'batch-size', 'trace-size', 'shift', 'frequency', 'detections-for-event',
        # Info output and computation restriction
        'time', 'cpu', 'print-files', 'generate-waveforms', 'wavetool-waveforms',
        'detection-stations', 'waveform-duration', 'generate-s-files', 'silence-wavetool',
        # Environment
        'input', 'seisan', 'mulplt', 'archives'
    ]


def get_args_dictionaries(args):
    """
    Returns converted to a dictionary arguments and a dictionary of arguments type
    :return: (dict, dict)
    """
    # Convert args namespace to a final nested dictionary-like structure of config file
    d_args = {
        'main': {
            'favor': args.favor,
            'cnn': args.cnn,
            'gpd': args.gpd,
            'model': args.model,
            'model-name': None,
            'weights': args.weights,
            'loader-argv': args.loader_argv,
            'features-number': args.features_number,
            'waveform-duration': args.waveform_duration,
            'start': args.start,
            'end': args.end,
            'threshold': args.threshold,
            'batch-size': args.batch_size,
            'trace-size': args.trace_size,
            'shift': args.shift,
            'generate-s-files': args.generate_s_files,
            'detections-for-event': args.detections_for_event,
            'generate-waveforms': args.generate_waveforms,
            'wavetool-waveforms': args.wavetool_waveforms,
            'detection-stations': args.detection_stations,
            'no-filter': args.no_filter,
            'no-detrend': args.no_detrend,
            'trace-normalization': args.trace_normalization,
            'frequency': args.frequency,
            'plot-positives': args.plot_positives,
            'silence-wavetool': args.silence_wavetool,
            'plot-positives-original': args.plot_positives_original,
            'print-scores': args.print_scores,
            'print-precision': args.print_precision,
            'combine-events-range': args.combine_events_range,
            'time': args.time,
            'cpu': args.cpu,
            'print-files': args.print_files,
            'config': args.config,
            'input': args.input,
            'out': args.out,
            'seisan': args.seisan,
            'mulplt': args.mulplt,
            'archives': args.archives,
            'channel-order': args.channel_order,
        },
    }

    # Default weights for models
    default_weights = {'favor': 'weights/w_model_performer_with_spec.hd5',
                       'cnn': 'weights/w_model_cnn_spec.hd5',
                       'gpd': 'weights/w_gpd_scsn_2000_2017.h5'}

    # Following functions will be applied to parameters to set default values, convert data types, etc.
    def apply_default_model_name(model_name, params, key):
        station_params = params[key]
        if station_params['favor']:
            return 'favor'
        if station_params['cnn']:
            return 'cnn'
        if station_params['gpd']:
            return 'gpd'
        if station_params['model']:
            return 'custom-model'
        if params['main', 'model-name']:
            return None
        raise AttributeError('Model is not specified, and default model did not apply! If you see this message, '
                             'this is a bug!')

    @applied_function(defaults=default_weights)
    def apply_default_weights(weight, params, key, defaults):
        params = params[key]
        if weight:
            return weight
        if params['model-name'] is None:
            return None
        if params['model-name'] not in defaults:
            return None
        return defaults[params['model-name']]

    # Type converters
    def type_converter(value, _, __, f_type):
        if value is None:
            return None
        if type(value) is f_type:
            return value
        return f_type(value)

    int_converter = applied_function(f_type=int)(type_converter)
    float_converter = applied_function(f_type=float)(type_converter)

    def string_trimmer(value, _, __):
        if type(value) is not str:
            return value
        return value.strip()

    def bool_converter(value, _, __):
        if value is None:
            return None
        if type(value) is bool:
            return value
        if type(value) is str:
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            else:
                return None
        return bool(value)

    def favor_default(value, params, key):
        params = params[key]
        if value is None and params['cnn'] is None and params['gpd'] is None and params['weights'] is None:
            return None
        if not params['cnn'] and not params['gpd'] and not params['model']:
            return True

    def utc_datetime_converter(date, _, __):
        """
        Parse parameter from dictionary to UTCDateTime type.
        """
        if date is None:
            return None
        if not date:
            return date
        try:
            return UTCDateTime(date)
        except Exception as e:
            return None

    def start_date_default(value, _, __):
        if value is None:
            return None
        if not value:
            date = UTCDateTime()
            return UTCDateTime(f'{date.year}-{date.month}-{date.day}')
        return value

    def end_date_default(value, params, key):
        params = params[key]
        if value is None:
            return None
        if not value:
            date = UTCDateTime()
            date = UTCDateTime(f'{date.year}-{date.month}-{date.day}')
            if params['start']:
                date = params['start']
            return date + 24 * 60 * 60 - 0.000001
        return value

    def trace_size_converter(value, params, key):
        """
        Converts trace size from seconds to samples
        """
        params = params[key]
        if value is None:
            return None
        if not value:
            return value
        if not params['frequency']:
            raise AttributeError('No frequency specified for trace-size argument!')
        return int(float(value) * params['frequency'])

    def threshold_converter(value, _, __):
        if value is None:
            return None

        # Set label variables
        positive_labels = {'p': 0, 's': 1}

        # Parse and validate thresholds
        threshold_labels = {}
        global_threshold = False
        if type(value) is str:

            split_thresholds = value.split(',')

            if len(split_thresholds) == 1:
                value = float(value)
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
            value = float(value)
            global_threshold = True

        if global_threshold:
            for label in positive_labels:
                threshold_labels[label] = value
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

        return threshold_labels

    def channel_order_converter(value, _, __):
        if value is None:
            return value
        return [x.split(',') for x in value.strip(',;').split(';')]

    # Dictionary of default values setter, type converters and other applied functions
    d_applied_functions = {
        'favor': [bool_converter, favor_default],
        'cnn': [bool_converter],
        'gpd': [bool_converter],
        'model-name': [apply_default_model_name],
        'weights': [apply_default_weights],
        'features-number': [int_converter],
        'waveform-duration': [float_converter],
        'start': [utc_datetime_converter, start_date_default],
        'end': [utc_datetime_converter, end_date_default],
        'threshold': [threshold_converter],
        'batch-size': [int_converter],
        'frequency': [float_converter],
        'trace-size': [float_converter, trace_size_converter],
        'shift': [int_converter],
        'generate-s-files': [string_trimmer],
        'detections-for-event': [int_converter],
        'generate-waveforms': [string_trimmer],
        'no-filter': [bool_converter],
        'no-detrend': [bool_converter],
        'trace-normalization': [bool_converter],
        'wavetool-waveforms': [bool_converter],
        'detection-stations': [bool_converter],
        'plot-positives': [bool_converter],
        'silence-wavetool': [bool_converter],
        'plot-positives-original': [bool_converter],
        'print-scores': [bool_converter],
        'print-precision': [int_converter],
        'combine-events-range': [float_converter],
        'time': [bool_converter],
        'cpu': [bool_converter],
        'print-files': [bool_converter],
        'channel-order': [channel_order_converter],
    }

    return d_args, d_applied_functions


def parse_seisan_def_env(path, params):
    """
    Reads SEISAN.DEF and parses main environment parameters (does not parse stations).
    """
    pattern = 'ARC_ARCHIVE'
    l_pattern = len(pattern)
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[:l_pattern] == pattern:
                archive_path = line.split()
                if len(archive_path) == 2:
                    params['main', 'archives'] = archive_path[1].strip()
                break


def parse_unix(params):
    """
    Parses environment variables in UNIX systems and passes them to params, if not set earlier
    (through config file or command line arguments).
    """
    import os

    seisan_top = os.environ.get('SEISAN_TOP')
    if not seisan_top:
        return

    seisan_path = os.path.join(seisan_top, 'DAT/SEISAN.DEF')
    mulplt_path = os.path.join(seisan_top, 'DAT/MULPLT.DEF')

    parse_seisan_def_env(seisan_path, params)

    if not params.key_exists(('main', 'seisan')):
        print('seisan!')
        params['main', 'seisan'] = seisan_path
    if not params.key_exists(('main', 'mulplt')):
        print('mulplt!')
        params['main', 'mulplt'] = mulplt_path


def parse_env(params):
    """
    Parses environment variables and passes them to params, if not set earlier (through config file or
    command line arguments).
    """
    parse_unix(params)


def default_params(params):
    """
    Applies default values to not set parameters.
    """
    pass


def archive_scan_params():

    # Command line arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--seisan', help='Path to SEISAN.DEF', default='')
    parser.add_argument('--mulplt', help='Path to MULPLT.DEF', default='')
    parser.add_argument('--archives', help='Path to archives directory', default='')
    parser.add_argument('--input', help='Path to file with archive names', default='')
    parser.add_argument('--config', '-c', help='Path to config file', default='')
    parser.add_argument('--weights', '-w', help='Path to model weights', default='')
    parser.add_argument('--favor', help='Use seismo-performer standard model, defaults to True '
                                        'if no other model selected.', action='store_true')
    parser.add_argument('--cnn', help='Use simple CNN model on top of spectrogram', action='store_true')
    parser.add_argument('--gpd', help='Use GPD model', action='store_true')
    parser.add_argument('--model', help='Custom model loader import, default: None', default='')
    parser.add_argument('--loader_argv', help='Custom model loader arguments, default: None', default='')
    parser.add_argument('--out', '-o', help='Path to output file with predictions', default='predictions.txt')
    parser.add_argument('--threshold', help='Positive prediction threshold, default: 0.95', default=0.95)
    parser.add_argument('--batch-size', help='Model batch size, default: 150 slices '
                                             '(each slice is: 4 seconds by 3 channels)',
                        default=150, type=int)
    parser.add_argument('--trace-size', help='Length of loaded and processed seismic data stream, '
                                             'default: 600 seconds',
                        default=600, type=float)
    parser.add_argument('--shift', help='Sliding windows shift, default: 40 samples (40 ms)',
                        default=40, type=int)
    parser.add_argument('--frequency', help='Model data required frequency, default: 100 Hz', default=100.)
    parser.add_argument('--features-number', help='Model single channel input length, default: 400 samples',
                        default=400)
    parser.add_argument('--waveform-duration', help='Duration of a waveform slice for potential events, '
                                                    'default: 600 seconds',
                        type=float, default=600.)
    parser.add_argument('--no-filter', help='Do not filter input waveforms', action='store_true')
    parser.add_argument('--no-detrend', help='Do not detrend input waveforms', action='store_true')
    parser.add_argument('--silence-wavetool', help='Do not output any wavetool messages', action='store_true')
    parser.add_argument('--plot-positives', help='Plot positives waveforms', action='store_true')
    parser.add_argument('--plot-positives-original', help='Plot positives original waveforms, before '
                                                          'pre-processing',
                        action='store_true')
    parser.add_argument('--print-scores', help='Prints model prediction scores and according wave forms data'
                                               ' in .npy files',
                        action='store_true')
    parser.add_argument('--print-precision', help='Floating point precision for results pseudo-probability output',
                        default=4, type=int)
    parser.add_argument('--combine-events-range', help='Maximum range (in seconds) inside which '
                                                       'positives are visually combined as a single event,'
                                                       ' default: 30 seconds', default=30., type=float)
    parser.add_argument('--generate-s-files', help='Generate s-files for potential events? "no", "yes", '
                                                   '"ask once" (ask once per launch), "ask each" '
                                                   '(ask for every event), default: ask once',
                        default='ask once', type=str)
    parser.add_argument('--detections-for-event', help='Amount of detections in a group, to be considered as '
                                                       'event, default: 2', default=2, type=int)
    parser.add_argument('--generate-waveforms', help='Waveform generation: "no", "yes", "ask once" '
                                                     '(ask once per launch), "ask each" (ask for every event), '
                                                     'default: ask once',
                        default='ask once', type=str)
    parser.add_argument('--wavetool-waveforms', help='If set, use seisan wavetool programm to generate waveforms, '
                                                     'otherwise use custom ObsPy based module, not set by default',
                        action='store_true')
    parser.add_argument('--detection-stations', help='If set, slice waveforms only for stations'
                                                                    ' with detections, otherwise, slice from'
                                                                    ' every station scan was performed on.',
                        action='store_true')
    parser.add_argument('--time', help='Print out performance time in stdout', action='store_true')
    parser.add_argument('--cpu', help='Disable GPU usage', action='store_true')
    parser.add_argument('--print-files', help='Print out all archive file names before scan',
                        action='store_true')
    parser.add_argument('--start', '-s', help='Earliest time stamp allowed for input waveforms,'
                                              ' format examples: "2021-04-01" or "2021-04-01T12:35:40"',
                        default='')
    parser.add_argument('--end', '-e', help='Latest time stamp allowed for input waveforms'
                                            ' format examples: "2021-04-01" or "2021-04-01T12:35:40"',
                        default='')
    parser.add_argument('--trace-normalization', help='Normalize input data per trace, otherwise - per full trace.'
                                                      ' Increases performance and reduces memory demand if set (at'
                                                      ' a cost of potential accuracy loss).',
                        action='store_true')
    parser.add_argument('--channel-order',
                        help='Order of channels, specify with comma separation,'
                             ' without whitespaces. It is possible to specify multiple'
                             ' configurations using semicolon as a group separator: N,E,Z;1,2,Z',
                        default='N,E,Z')

    args = parser.parse_args()  # parse arguments

    # Default config file path
    if not args.config:
        args.config = ['data/config.ini',
                       '/etc/archive_scan_config.ini',
                       '/etc/opt/archive_scan_config.ini',
                       '/etc/opt/seismo-ml-models-integration/archive_scan_config.ini',
                       '~/.config/archive_scan_config.ini',
                       '~/.config/seismo-ml-models-integration/archive_scan_config.ini']

    # Convert args to a dictionary
    d_args, d_applied_functions = get_args_dictionaries(args)

    # Parse config files
    params = None
    if type(d_args['main']['config']) is str:
        d_args['main']['config'] = [d_args['main']['config']]
    for x in d_args['main']['config']:
        if not isfile(x):
            continue
        params = Params(path=x, config=d_args, default_dictionary='config')
        break
    if not params:
        print('Config file not found, using only default values and command line arguments!', file=sys.stderr)
        params = Params(path=None, config=d_args, default_dictionary='config')

    # Default env values
    default_seisan = ['data/SEISAN.DEF']
    default_mulplt = ['data/MULPLT.DEF']
    default_archives = ['data/archives/']

    if not params['main', 'input']:
        if not params['main', 'seisan']:
            for x in default_seisan:
                if not isfile(x):
                    continue
                params['main', 'seisan'] = x
                break
        if not params['main', 'mulplt']:
            for x in default_mulplt:
                if not isfile(x):
                    continue
                params['main', 'mulplt'] = x
                break
        if not params['main', 'archives']:
            for x in default_archives:
                if not isdir(x):
                    continue
                params['main', 'archives'] = x
                break

    # Parse environment variables
    parse_env(params)
    # Apply default values if not set
    default_params(params)

    for key, functions in d_applied_functions.items():
        for f in functions:
            params.apply_function(key, f)

    # Check env parameters validity
    if not params['main', 'input']:
        if not params['main', 'seisan']:
            raise AttributeError('Either "input" or "seisan" attribute should be set with correct values '
                                 '(through config file or command line arguments)')
        if not params['main', 'mulplt']:
            raise AttributeError('Either "input" or "mulplt" attribute should be set with correct values '
                                 '(through config file or command line arguments)')
        if not params['main', 'archives']:
            raise AttributeError('Either "input" or "archives" attribute should be set with correct values '
                                 '(through config file or command line arguments)')

    l_not_supported = get_unsupported_station_parameters_list()
    for x in l_not_supported:
        params.check_unsupported_station_parameter(x)

    return params
