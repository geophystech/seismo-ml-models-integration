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
        'time', 'cpu', 'print-files', 'generate-waveforms', 'wavetool-waveforms', 'register-events',
        'detection-stations', 'waveform-duration', 'generate-s-files', 'silence-wavetool',
        'use-default-database',
        # Environment
        'input', 'seisan', 'mulplt-def', 'archives', 'database', 'rea', 'wav',
        # Advanced search
        'advanced-search', 'advanced-search-range',
    ]


def defaults():
    """
    Returns dictionary of default values for parameters.
    """
    defs = {
        'main': {
            'input': '',
            'config': '',
            'weights': '',
            'favor': False,
            'cnn': False,
            'gpd': False,
            'model': '',
            'loader_argv': '',
            'out': 'predictions.txt',
            'threshold': 0.95,
            'batch-size': 150,
            'trace-size': 600,
            'shift': 10,
            'frequency': 100.,
            'features-number': 400,
            'waveform-duration': 600.,
            'no-filter': False,
            'no-detrend': False,
            'silence-wavetool': False,
            'plot-positives': False,
            'plot-positives-original': False,
            'print-scores': False,
            'print-precision': 4,
            'combine-events-range': 30.,
            'generate-s-files': 'ask once',
            'detections-for-event': 2,
            'generate-waveforms': 'ask once',
            'register-events': 'ask once',
            'use-default-database': False,
            'wavetool-waveforms': False,
            'detection-stations': False,
            'time': False,
            'cpu': False,
            'print-files': False,
            'start': '',
            'end': '',
            'trace-normalization': False,
            'run-configure': False,
            'channel-order': 'N,E,Z;1,2,Z;Z,Z,Z',
            'combine-traces-min-length-difference-error': 3,
            'false-positives': None,
            'advanced-search': False,
            'advanced-search-range': 40.,
            'advanced-search-threshold': 0.9,
            'advanced-search-shift': 1,
        },
    }
    return defs


def archive_scan():

    # Command line arguments parsing
    from .command_line import archive_scan as cmd_args
    args = cmd_args()

    # Parse config files
    from .config import archive_scan as config_params
    params = config_params(args)

    # Parse environment variables
    from .env import archive_scan as env_params
    env_params(params)

    # Apply default values
    d_defaults = defaults()
    for level_name, level in d_defaults.items():
        for name, arg in level.items():
            if not params[level_name, name]:
                params[level_name, name] = arg

    # Apply functions to parameters
    from .applied_functions import archive_scan as applied_functions
    applied_functions = applied_functions()

    for key, functions in applied_functions.items():
        for f in functions:
            params.apply_function(key, f)

    # Check env parameters validity
    if not params['main', 'input']:
        if not params['main', 'seisan']:
            raise AttributeError('Either "input" or "seisan" attribute should be set with correct values '
                                 '(through config file or command line arguments)')
        if not params['main', 'mulplt-def']:
            raise AttributeError('Either "input" or "mulplt-def" attribute should be set with correct values '
                                 '(through config file or command line arguments)')
        if not params['main', 'archives']:
            raise AttributeError('Either "input" or "archives" attribute should be set with correct values '
                                 '(through config file or command line arguments)')
    if params['main', 'database'] and len(params['main', 'database']) != 5:
        raise AttributeError(f'"database" attribute ({params["main", "database"]}) should '
                             f'have a length of 5 characters.')

    l_not_supported = get_unsupported_station_parameters_list()
    for x in l_not_supported:
        params.check_unsupported_station_parameter(x)

    return params
