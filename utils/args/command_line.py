import argparse

def archive_scan_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seisan', help='Path to SEISAN.DEF')
    parser.add_argument('--mulplt', help='Path to MULPLT.DEF')
    parser.add_argument('--archives', help='Path to archives directory')
    parser.add_argument('--input', help='Path to file with archive names')
    parser.add_argument('--config', '-c', help='Path to config file')
    parser.add_argument('--weights', '-w', help='Path to model weights')
    parser.add_argument('--favor', help='Use seismo-performer standard model, defaults to True '
                                        'if no other model selected.', action='store_true')
    parser.add_argument('--cnn', help='Use simple CNN model on top of spectrogram', action='store_true')
    parser.add_argument('--gpd', help='Use GPD model', action='store_true')
    parser.add_argument('--model', help='Custom model loader import, default: None')
    parser.add_argument('--loader_argv', help='Custom model loader arguments, default: None')
    parser.add_argument('--out', '-o', help='Path to output file with predictions')
    parser.add_argument('--threshold', help='Positive prediction threshold, default: 0.95')
    parser.add_argument('--batch-size', help='Model batch size, default: 150 slices '
                                             '(each slice is: 4 seconds by 3 channels)',
                        type=int)
    parser.add_argument('--trace-size', help='Length of loaded and processed seismic data stream, '
                                             'default: 600 seconds',
                        type=float)
    parser.add_argument('--shift', help='Sliding windows shift, default: 40 samples (40 ms)',
                        type=int)
    parser.add_argument('--frequency', help='Model data required frequency, default: 100 Hz', type=float)
    parser.add_argument('--features-number', help='Model single channel input length, default: 400 samples',
                        type=int)
    parser.add_argument('--waveform-duration', help='Duration of a waveform slice for potential events, '
                                                    'default: 600 seconds',
                        type=float)
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
    parser.add_argument('--print-precision', help='Floating point precision for results '
                                                  'pseudo-probability output',
                        type=int)
    parser.add_argument('--combine-events-range', help='Maximum range (in seconds) inside which '
                                                       'positives are visually combined as a single event,'
                                                       ' default: 30 seconds', type=float)
    parser.add_argument('--generate-s-files', help='Generate s-files for potential events? "no", "yes", '
                                                   '"ask once" (ask once per launch), "ask each" '
                                                   '(ask for every event), default: ask once',
                        type=str)
    parser.add_argument('--detections-for-event', help='Amount of detections in a group, to be considered as '
                                                       'event, default: 2', type=int)
    parser.add_argument('--generate-waveforms', help='Waveform generation: "no", "yes", "ask once" '
                                                     '(ask once per launch), "ask each" (ask for every event), '
                                                     'default: ask once',
                        type=str)
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
                                              ' format examples: "2021-04-01" or "2021-04-01T12:35:40"')
    parser.add_argument('--end', '-e', help='Latest time stamp allowed for input waveforms'
                                            ' format examples: "2021-04-01" or "2021-04-01T12:35:40"')
    parser.add_argument('--trace-normalization', help='Normalize input data per trace, otherwise - per full trace.'
                                                      ' Increases performance and reduces memory demand if set (at'
                                                      ' a cost of potential accuracy loss).',
                        action='store_true')
    parser.add_argument('--channel-order',
                        help='Order of channels, specify with comma separation,'
                             ' without whitespaces. It is possible to specify multiple'
                             ' configurations using semicolon as a group separator: N,E,Z;1,2,Z')
    return parser.parse_args()


def archive_scan_defaults(args):
    """
    Sets default values for archive_scan.py command line arguments.
    """
    if not args.config:
        args.config = ['data/config.ini',
                       '/etc/archive_scan_config.ini',
                       '/etc/opt/archive_scan_config.ini',
                       '/etc/opt/seismo-ml-models-integration/archive_scan_config.ini',
                       '~/.config/archive_scan_config.ini',
                       '~/.config/seismo-ml-models-integration/archive_scan_config.ini']


def archive_scan():
    args = archive_scan_args()
    archive_scan_defaults(args)
    return args