import sys
import os

import utils.scan_tools as stools
from utils.args import archive_scan as archive_scan_params
from utils.seisan import get_archives
from utils.configure.configure_archive_scan import configure
from utils import utils
import utils.scanner as scanner

# Silence tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

    if params['main', 'run-configure']:
        configure()
        sys.exit(0)

    if params['main', 'print-params']:
        print(params)

    if params['main', 'input']:
        archives = stools.parse_archive_csv(params['main', 'input'])  # parse archive names
        input_mode = True
    else:
        archives = get_archives(seisan=params['main', 'seisan'],
                                mulplt=params['main', 'mulplt-def'],
                                archives=params['main', 'archives'],
                                params=params)
        input_mode = False

    if input_mode and params['main', 'evaluate']:
        raise AttributeError('Cannot have --input and --evaluate parameters both (cannot evaluate model without seisan'
                             ' database archive)!')
    if input_mode and params['main', 'false-positives']:
        raise AttributeError('Cannot have --input and --false-positives parameters both (cannot evaluate model '
                             'without seisan database archive)!')

    if params['main', 'cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Set values
    params['main', 'model-labels'] = {'p': 0, 's': 1, 'n': 2}
    params['main', 'positive-labels'] = {'p': 0, 's': 1}
    params['main', 'label-names'] = {'p': 'P', 's': 'S'}
    params['main', 'half-duration'] = (params['main', 'features-number'] * 0.5) / params['main', 'frequency']

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

    if params['main', 'print-files']:
        print('Scan archives:')
        for n_archive, d_archives in enumerate(archives):
            print(f'{n_archive + 1}: {d_archives["paths"]}')
        print()

    # Scan
    all_positives, performance = scanner.archive_scan(archives, params)

    # Re-write predictions files
    if params['main', 'false-positives']:
        stools.gather_false_positives(all_positives, params)

    all_positives, events = stools.finalize_predictions(all_positives, params, input_mode=input_mode)

    if params['main', 'advanced-search']:
        events = scanner.advanced_search(events, params, input_mode=input_mode)

    stools.output_predictions(all_positives, events, params, input_mode=input_mode)

    if params['main', 'time-batch']:
        utils.print_time_batch(performance['batch-time'])
    if params['main', 'time-archive']:
        utils.print_time_archive(performance['archives-time'])
    if params['main', 'walltime-archive']:
        utils.print_time_archive(performance['archives-walltime'], walltime=True)
    if params['main', 'time']:
        print(f'\nTotal prediction time: {performance["total-performance-time"]:.6} seconds')
