import argparse
from obspy import UTCDateTime
import sys


def archive_scan_args():

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
                        default=150)
    parser.add_argument('--trace-size', '-b', help='Length of loaded and processed seismic data stream, '
                                                   'default: 600 seconds', default=600)
    parser.add_argument('--shift', help='Sliding windows shift, default: 40 samples (40 ms)', default=40)
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
                        default=4)
    parser.add_argument('--time', help='Print out performance time in stdout', action='store_true')
    parser.add_argument('--cpu', help='Disable GPU usage', action='store_true')
    parser.add_argument('--start', help='Earliest time stamp allowed for input waveforms,'
                                        ' format examples: "2021-04-01" or "2021-04-01T12:35:40"', default=None)
    parser.add_argument('--end', help='Latest time stamp allowed for input waveforms'
                                      ' format examples: "2021-04-01" or "2021-04-01T12:35:40"', default=None)
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

    # Set start and end date
    def parse_date_param(args, p_name):
        """
        Parse parameter from dictionary to UTCDateTime type.
        """
        if not getattr(args, p_name):
            return None

        try:
            return UTCDateTime(getattr(args, p_name))
        except TypeError as e:
            print(f'Failed to parse "{p_name}" parameter (value: {getattr(args, p_name)}).'
                  f' Use {__file__} -h for date format information.')
            sys.exit(1)
        except Exception as e:
            print(f'Failed to parse "{p_name}" parameter (value: {getattr(args, p_name)}).'
                  f' Use {__file__} -h for date format information.')
            raise

    args.end = parse_date_param(args, 'end')
    args.start = parse_date_param(args, 'start')

    # Convert args to a dictionary
    d_args = {
        'model': {
            'weights': args.weights,
            'cnn': args.cnn,
            'gpd': args.gpd,
            'model': args.model,
            'loader_argv': args.loader_argv,
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

    return d_args
