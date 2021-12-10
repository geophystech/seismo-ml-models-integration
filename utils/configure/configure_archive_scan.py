"""
This module primary function is configure() which will try to generate a new configuration file for
the archive_scan.py script run.
"""
import os
from ..seisan import parse_seisan_def, parse_seisan_params, parse_multplt, generate_mulplt_def
from . import completer


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


def configure_unix():
    completer.init()

    seisan_top = os.environ.get('SEISAN_TOP')
    seisan_top = ask('Enter path to top seisan directory (which contains DAT, REA, WAV, ect.)',
                     seisan_top)

    default_database = os.environ.get('DEF_BASE')
    default_database = ask('Enter default database name (5 or shorter characters)',
                           default_database)

    seisan = os.path.join(seisan_top, 'DAT', 'SEISAN.DEF')
    seisan = ask('Enter path to SEISAN.DEF', seisan)
    seisan_params_parsed = parse_seisan_params(seisan)

    archives = seisan_params_parsed['archives']
    archives = ask('Enter path to top archive directory', archives)

    mulplt_def_env = os.path.join(seisan_top, 'DAT', 'MULPLT.DEF')
    use_default_mulplt = False
    mulplt_parsed = parse_multplt(mulplt_def_env)
    l_stations = []

    if len(mulplt_parsed):
        print('MULPLT.DEF found!')
        use_default_mulplt = ask_yes_no(f'Generate list of stations for scanning from {mulplt_def_env}')
        # TODO: make a better question. Ask either use MULPLT or generate your own list

    print('Reading all avaliable stations from SEISAN.DEF..')
    stations = parse_seisan_def(seisan)

    # Create new stations list
    if not use_default_mulplt:
        station_names = [x['station'] for x in stations]
        station_names.append('quit')
        completer.set_completer(station_names)
        # TODO: unset completer after done with stations
        print('\nTo add station to a list, enter its name, '
              'note that auto-completion avaliable by pressing TAB '
              '(or double TAB for all avaliable options).')
        print('Enter "quit" to finish the process.')
        # TODO: also give option to just use all stations
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

    default_channel_orders = [
        ['N', 'E', 'Z'],
        ['1', '2', 'Z'],
        ['Z', 'Z', 'Z']
    ]

    orders_used = []
    channels_not_fit = []

    # TODO: all raw list outputs replace by proper outputs

    # TODO: This needs to be a separate function "validate_channel_orders" which needed to be called in the loop
    #   ..until everything is fine, or "discard unfit stations" ("quit") option is selected.
    for channels in unique_channels:
        passed = False
        for order in default_channel_orders:
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

    if len(channels_not_fit):
        print('Failed to generate channel orders for some stations channels:')
        for i, x in enumerate(channels_not_fit):
            print(f'{i}. {x}')
        raise NotImplementedError('Channels order input is not implemented!')

    print('Selected channel orders:')
    for i, x in enumerate(orders_used):
        print(f'{i}. {x}')

    # TODO: also do the opposite! Gather unique_channels, which will not be fully covered by orders
    #   ..and ask to resolve them, if needed.
    # TODO: If all stations fall into default channel orders, then just generate new MULPLT.DEF
    #   ..watch for name uniqueness!
    # TODO: If not, ask to input channel orders, to resolve unknown channels

    # TODO: If stations have more than 3 channels (with the same two letters) notify about them!

    # TODO: Make sure every "component" has at least two characters: instrument and channel

    # TODO: Ask for a name
    mulplt_def = 'MULPLT.DEF'
    generate_mulplt_def('MULPLT.DEF', selected_stations, enforce_unique=True)
    print(f'Stations list saved as {mulplt_def}')


def configure():
    print('Running archive_scan.py configuration script!')
    configure_unix()
