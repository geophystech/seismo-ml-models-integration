"""
This module primary function is configure() which will try to generate a new configuration file for
the archive_scan.py script run.
"""
import os
import re

from ..seisan import parse_seisan_def, parse_seisan_params, parse_multplt, \
    generate_mulplt_def, create_unique_file
from . import completer
from ..params import Params


n_channels = 3


def generate_config(path, **kwargs):
    print('generate_config called!')
    print('path: ', path)
    print('kwargs type: ', kwargs)
    print('kwargs: ', kwargs)


def ask_yes_no(question, repeat=True):
    """
    Asks a question with answer YES/NO. Returns True if YES, False otherwise.
    :param question - question to ask
    :param repeat - if True, will repeat a question until either positive or negative answer given,
        if False - any non-positive answer is treated as negative (no need to input exact negative answer, e.g.
        "NO" or "N", etc.)
    """
    print(question + ' [Y/N]: ', end='')

    while True:
        answer = input()
        answer = answer.strip().lower()
        if answer in ['y', 'yes']:
            return True
        if not repeat:
            return False
        if answer in ['n', 'no']:
            return False
        print('Please, enter either Y or N: ', end='')


def ask(question, default=None, validation=None):
    if default:
        while True:
            print(question, f' [{default}]: ', sep='', end='')
            answer = input().strip()
            if validation and not validation(answer):
                continue
            if answer:
                return answer
            return default
    if not default:
        while True:
            print(question, ': ', sep='', end='')
            answer = input().strip()
            if validation and not validation(answer):
                continue
            if answer:
                return answer
            print('Empty string is not accepted!')


def validate_channels(unique_channels, channel_orders):
    """
    Checks if list of unique channels lists (or sets) fully covered by provided channel orders.
    Returns tuple of two lists: list channel orders, which are covering provided channels and
    list of station channels, which are not covered. Former being empty indicates full coverege.
    :param unique_channels:
    :param channel_orders:
    :return: <orders used>, <channels not covered>
    """
    orders_used = []
    channels_not_fit = []

    for channels in unique_channels:
        passed = False
        for order in channel_orders:
            local_passed = True
            for x in order:
                if x not in channels:
                    local_passed = False
                    break
            if local_passed:
                if order not in orders_used:
                    orders_used.append(order)
                passed = True
                break

        if not passed:
            channels_not_fit.append(channels)

    return orders_used, channels_not_fit


def parse_channel_order(s_order):
    """
    Parses string to a channel order.
    If parsing is failed, returns str with error failure description!
    """
    channels = re.split('\W+', s_order)
    if len(channels) != n_channels:
        return f'Channel order should have {n_channels} channels but has {len(channels)}!'
    for x in channels:
        if len(x) != 1:
            return f'Each channel should be exactly one character long. Channel "{x}" breaks this rule!'
    return channels



def configure_unix():
    completer.init()

    d_params = {
        'main': {}
    }

    seisan_top = os.environ.get('SEISAN_TOP')
    seisan_top = ask('Enter path to top seisan directory (which contains DAT, REA, WAV, ect.)',
                     seisan_top)

    default_database = os.environ.get('DEF_BASE')
    default_database = ask('Enter default database name (5 or shorter characters)',
                           default_database)
    d_params['main']['database'] = default_database

    seisan = os.path.join(seisan_top, 'DAT', 'SEISAN.DEF')
    seisan = ask('Enter path to SEISAN.DEF', seisan)
    d_params['main']['seisan'] = seisan

    seisan_params_parsed = parse_seisan_params(seisan)

    archives = seisan_params_parsed['archives']
    archives = ask('Enter path to top archive directory', archives)
    d_params['main']['archives'] = archives

    mulplt_def_env = os.path.join(seisan_top, 'DAT', 'MULPLT.DEF')
    use_default_mulplt = False
    mulplt_parsed = parse_multplt(mulplt_def_env)
    l_stations = []

    if len(mulplt_parsed):
        print('MULPLT.DEF found!')
        use_default_mulplt = ask_yes_no(f'Generate list of stations for scanning from {mulplt_def_env}')

    print('Reading all avaliable stations from SEISAN.DEF..')
    stations = parse_seisan_def(seisan)

    # Create new stations list
    if not use_default_mulplt:
        station_names = [x['station'] for x in stations]
        station_names.append('quit')
        completer.set_completer(station_names)
        print('\nTo add station to a list, enter its name, '
              'note that auto-completion avaliable by pressing TAB '
              '(or double TAB for all avaliable options).')
        print('Enter "quit" to finish the process.')
        s_input = ''
        while s_input != 'quit':

            s_input = input()

            if s_input == 'quit':
                break_stations_input = True
                if not len(l_stations):
                    break_stations_input = ask_yes_no('No stations currently added, do you want to quit '
                                                      'stations list input')
                if break_stations_input:
                    break
                else:
                    continue

            if s_input in station_names:
                l_stations.append(s_input)
                print(f'Station {s_input} added!')
            else:
                print(f'Cannot find {s_input} on the list of stations from SEISAN.DEF!')

            print('Current list: ', l_stations)

    print('Stations: ', l_stations)

    # Form list of selected stations (with full data):
    selected_stations = []
    for name in l_stations:
        for x in stations:
            if x['station'] == name:
                selected_stations.append(x)

    print('Selecting stations channel orders to organize model input data..')
    # Create a list of unique channels (order does not matter).
    unique_channels = []
    for x in selected_stations:
        channels = {component[-1] for component in x['components']}
        if channels not in unique_channels:
            unique_channels.append(channels)

    channel_orders = [
        ['N', 'E', 'Z'],
        ['1', '2', 'Z'],
        ['Z', 'Z', 'Z']
    ]

    while True:

        channel_orders, channels_not_covered = validate_channels(unique_channels, channel_orders)

        if not len(channels_not_covered):
            break

        print('Failed to generate channel orders for some stations channels:')
        for i, x in enumerate(channels_not_covered):
            print(f'{i}. {x}')

        print('You can either enter channel orders manually or discard stations which are '
              'not covered by channel orders.')
        answer = ask('Enter a channel order (separated by commas or whitespaces) or "quit" to '
                     'finish and discard all not covered stations')

        if answer == 'quit':
            break

        parsed_channels = parse_channel_order(answer)
        # str return means parsing failed!
        if type(parsed_channels) is str:
            print(parsed_channels)
            break

        channel_orders.append(parsed_channels)
        print(f'Appended channel order: {parsed_channels}')

    print('Selected channel orders:')
    s_channel_order = ''
    first_order = True
    for i, x in enumerate(channel_orders):
        print(f'{i}. {x}')
        if first_order:
            first_order = False
        else:
            s_channel_order += ';'
        s_channel_order += ','.join(x)
    d_params['main']['channel-order'] = s_channel_order

    mulplt_def = 'data/MULPLT.DEF'
    mulplt_def = generate_mulplt_def('MULPLT.DEF', selected_stations, enforce_unique=True)
    print(f'Stations list saved as {mulplt_def}')
    d_params['main']['mulplt-def'] = mulplt_def

    params = Params(config=d_params, default_dictionary='config')
    print('\nGenerated archive_scan.py params:')
    print(params)

    config_path = 'data/config.ini'
    config_file, config_path = create_unique_file(config_path, 'w')
    params.save_ini(file=config_file)

    print(f'Config saved as {config_path}\n')
    print('USAGE (last 24 hours scan):')
    print(f'python archive_scan.py -c {config_path}\n')
    print('USAGE (custom time scan):')
    print(f'python archive_scan.py -s <START_DATE> -e <END_DATE> -c {config_path}')
    print('Date format:')
    print('YYYY-MM-DD or YYYY-MM-DDThh:mm:ss')
    print('2021-04-01 or 2021-04-01T12:35:50\n')


def configure():
    print('Running archive_scan.py configuration script!')
    configure_unix()
