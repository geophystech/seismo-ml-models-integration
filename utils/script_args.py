import argparse
from obspy import UTCDateTime
import sys
from os.path import isfile

from utils.params import Params


# Set start and end date
def parse_date_param(args, *p_name):
    """
    Parse parameter from dictionary to UTCDateTime type.
    """
    if not args.__getitem__(p_name):
        return None
    try:
        return UTCDateTime(args.__getitem__(p_name))
    except Exception as e:
        pass


def archive_scan_params():
    # Default weights for models
    default_weights = {'favor': 'weights/w_model_performer_with_spec.hd5',
                       'cnn': 'weights/weights_model_cnn_spec.hd5',
                       'gpd': 'weights/w_gpd_scsn_2000_2017.h5'}

    # Command line arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to file with archive names')
    parser.add_argument('--config', '-c', help='Path to config file', default=None)
    parser.add_argument('--weights', '-w', help='Path to model weights', default=None)
    parser.add_argument('--cnn', help='Use simple CNN model on top of spectrogram', action='store_true')
    parser.add_argument('--gpd', help='Use GPD model', action='store_true')
    parser.add_argument('--model', help='Custom model loader import, default: None', default=None)
    parser.add_argument('--loader_argv', help='Custom model loader arguments, default: None', default=None)
    parser.add_argument('--out', '-o', help='Path to output file with predictions', default='predictions.txt')
    parser.add_argument('--threshold', help='Positive prediction threshold, default: 0.95', default=0.95)
    parser.add_argument('--batch-size', help='Model batch size, default: 150 slices '
                                             '(each slice is: 4 seconds by 3 channels)',
                        default=150, type=int)
    parser.add_argument('--trace-size', '-b', help='Length of loaded and processed seismic data stream, '
                                                   'default: 600 seconds',
                        default=600, type=float)
    parser.add_argument('--shift', help='Sliding windows shift, default: 40 samples (40 ms)',
                        default=40, type=int)
    parser.add_argument('--frequency', help='Model data required frequency, default: 100 Hz', default=100.)
    parser.add_argument('--features-number', help='Model single channel input length, default: 400 samples',
                        default=400)
    parser.add_argument('--no-filter', help='Do not filter input waveforms', action='store_true')
    parser.add_argument('--no-detrend', help='Do not detrend input waveforms', action='store_true')
    parser.add_argument('--plot-positives', help='Plot positives waveforms', action='store_true')
    parser.add_argument('--plot-positives-original', help='Plot positives original waveforms, before '
                                                          'pre-processing',
                        action='store_true')
    parser.add_argument('--print-scores', help='Prints model prediction scores and according wave forms data'
                                               ' in .npy files',
                        action='store_true')
    parser.add_argument('--print-precision', help='Floating point precision for results pseudo-probability output',
                        default=4, type=int)
    parser.add_argument('--time', help='Print out performance time in stdout', action='store_true')
    parser.add_argument('--cpu', help='Disable GPU usage', action='store_true')
    parser.add_argument('--start', '-s', help='Earliest time stamp allowed for input waveforms,'
                                              ' format examples: "2021-04-01" or "2021-04-01T12:35:40"',
                        default=None)
    parser.add_argument('--end', '-e', help='Latest time stamp allowed for input waveforms'
                                            ' format examples: "2021-04-01" or "2021-04-01T12:35:40"',
                        default=None)
    parser.add_argument('--trace-normalization', help='Normalize input data per trace, otherwise - per full trace.'
                                                      ' Increases performance and reduces memory demand if set (at'
                                                      ' a cost of potential accuracy loss).',
                        action='store_true')

    args = parser.parse_args()  # parse arguments

    # Set default weights paths
    if args.cnn and not args.weights:
        args.weights = default_weights['cnn']
    elif args.gpd and not args.weights:
        args.weights = default_weights['gpd']
    elif not args.model and not args.weights:
        args.weights = default_weights['favor']

    # Default config file path
    if not args.config:
        args.config = ['data/config.ini',
                       '/etc/archive_scan_config.ini',
                       '/etc/opt/archive_scan_config.ini',
                       '/etc/opt/seismo-ml-models-integration/archive_scan_config.ini',
                       '~/.config/archive_scan_config.ini',
                       '~/.config/seismo-ml-models-integration/archive_scan_config.ini']

    # Convert args to a dictionary
    d_args = {
        'model': {
            'weights': args.weights,
            'cnn': args.cnn,
            'gpd': args.gpd,
            'model': args.model,
            'loader-argv': args.loader_argv,
            'features-number': args.features_number,
        },
        'scan': {
            'start': args.start,
            'end': args.end,
            'threshold': args.threshold,
            'batch-size': args.batch_size,
            'trace-size': args.trace_size,
            'shift': args.shift,
            'no-filter': args.no_filter,
            'no-detrend': args.no_detrend,
            'trace-normalization': args.trace_normalization,
            'frequency': args.frequency,
        },
        'info': {
            'plot-positives': args.plot_positives,
            'plot-positives-original': args.plot_positives_original,
            'print-scores': args.print_scores,
            'print-precision': args.print_precision,
            'time': args.time,
            'cpu': args.cpu,
        },
        'env': {
            'config': args.config,
            'input': args.input,
            'out': args.out,
        },
    }

    # Parse config files
    params = None
    for x in d_args['env']['config']:
        if not isfile(x):
            continue
        params = Params(path=x, config=d_args, default_dictionary='config')
        break
    if not params:
        print('Config file not found, using only default values and command line arguments!', file=sys.stderr)
        params = Params(path=None, config=d_args)

    params['scan', 'end'] = parse_date_param(params, 'scan', 'end')
    params['scan', 'start'] = parse_date_param(params, 'scan', 'start')

    # Trace size from seconds to samples
    params['scan', 'trace-size'] = int(float(params['scan', 'trace-size']) * params['scan', 'trace-size'])

    return params
