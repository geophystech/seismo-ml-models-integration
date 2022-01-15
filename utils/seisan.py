import os
import sys
import re
from os.path import isfile
from obspy import UTCDateTime


def parse_seisan_params(path, params=None):
    """
    Reads SEISAN.DEF and parses main environment parameters (does not parse stations).
    If params is None, then will return a dictionary with parsed data.
    """
    if not params:
        d_params = {}
    pattern = 'ARC_ARCHIVE'
    l_pattern = len(pattern)
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[:l_pattern] == pattern:
                archive_path = line.split()
                if len(archive_path) == 2:
                    if not params:
                        d_params['archives'] = archive_path[1].strip()
                    else:
                        params['main', 'archives'] = archive_path[1].strip()
                break
    if not params:
        return d_params


def order_group(group, channel_order):
    # Determine correct channel order group
    order = None
    for channels in channel_order:
        if all(x in group.keys() for x in channels):
            order = channels
            break
    if not order:
        return None

    paths = [group[x] for x in order]
    # Check if all archives actually exist
    for x in paths:
        if not isfile(x):
            return None
    return paths


def convert_station_group_to_dictionary(group):
    """
    Takes station group in format: [[station, component, network, location, start, end], [..], ...]
    and returns its dictionary representation:
    {
     station: <station>,
     components: [component1, .., componentN],
     network: <network>,
     location: <location>,
     start: <start>,
     end: <end>
    }
    """
    components = [x[1] for x in group]
    station = group[0]
    return {
        'station': station[0],
        'components': components,
        'network':  station[2],
        'location': station[3],
        'start': station[4],
        'end': station[5]
    }


def archive_to_path(archive, date, archives_path):
    """
    Converts archive entry - dictionary of elements:
       {station,
       components,
       network,
       location,
       start,
       end]
    to dictionary of file names: {"N": "data/20160610AZTRO/20160610000000.AZ.TRO.HHN.mseed",}
    Path example: /seismo/archive/IM/LNSK/LNSK.IM.00.EHE.2016.100
                  <archive_dir>/<location>/<station>/<station>.<location>.<code>.<channel>.<year>.<julday>
    """
    # Fix path
    if archives_path[-1] != '/':
        archives_path += '/'

    # Get julian day and year
    julday = date.julday
    year = date.year

    if julday // 10 == 0:
        julday = '00' + f'{julday}'
    elif julday // 100 == 0:
        julday = '0' + f'{julday}'

    d_result = {}

    # Metadata
    station = archive['station']
    loc_code = archive['location']
    net_code = archive['network']
    components = archive['components']
    start = None
    end = None

    for component in components:
        # Find channel type
        channel_type = component[-1]

        # Path to archive
        path = archives_path + '{}/{}/{}.{}.{}.{}.{}.{}'.format(net_code, station,
                                                                station, net_code,
                                                                loc_code, component,
                                                                year, julday)

        d_result[channel_type] = path

    d_station = {
        'station': station,
        'components': components,
        'network': net_code,
        'location': loc_code,
        'start': start,
        'end': end
    }

    return {
        'paths': d_result,
        'station': d_station
    }


def get_archives(seisan, mulplt, archives, params):
    """
    Returns lists of lists of archive file names to predict on. Also saves stations information to
    params['main', 'stations'].
    :return:
    """
    mulplt_parsed = parse_multplt(mulplt)
    d_stations = parse_seisan_def(seisan, multplt_data=mulplt_parsed)

    params['main', 'stations'] = d_stations

    start = params['main', 'start']
    end = params['main', 'end']

    c_date = start
    paths = []
    while c_date < end:
        paths.extend([archive_to_path(group, c_date, archives) for group in params['main', 'stations']])
        c_date += 24*60*60

    # Order channels and convert them into nested lists
    archives_paths = []
    for group in paths:
        l_ordered = order_group(group['paths'], params[group['station']['station'], 'channel-order'])
        if not l_ordered:
            continue
        archives_paths.append({
            'paths': l_ordered,
            'station': group['station']
        })

    return archives_paths


def parse_multplt(path):
    """
    Parses multplt.def file and returns list of lists like: [station, channel type (e.g. SH), channel (E, N or Z)].
    """
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        tag = "DEFAULT CHANNEL"

        for line in lines:
            if line[:len(tag)] == tag:
                # entry[0] - station, ..[1] - type, ..[2] - channel
                entry = line[len(tag):].split()
                data.append(entry)
    return data


def group_by(l, column, comp_margin=None):
    """
    Groups list entities by column values.
    """
    sorted_values = []
    result_list = []
    current_value = None

    for i in range(0, len(l)):
        x = l[i]
        if x[column][0:comp_margin] in sorted_values or x[column][0:comp_margin] == current_value:
            continue

        current_value = x[column][0:comp_margin]
        current_list = []
        for j in range(i, len(l)):
            y = l[j]
            if y[column][0:comp_margin] != current_value:
                continue

            current_list.append(y)

        sorted_values.append(current_value)
        result_list.append(current_list)

    return result_list


def process_archives_list(l):
    """
    Processes output of parse_seisan_def: combines into lists of three channeled entries.
    """
    lst = group_by(l, 0)
    result = []
    for x in lst:
        channel_group = group_by(x, 1, 2)
        for y in channel_group:
            location_group = group_by(y, 3)

            for z in location_group:
                result.append(z)
    return result


def parse_seisan_def(path, multplt_data=None):
    """
    Parses seisan.def file and returns grouped lists like:
    [station, channel, network_code, location_code, archive start date, archive end date (or None)].
    """
    data = []

    if multplt_data is not None:
        stations_channels = [x[0] + x[1] + x[2] for x in multplt_data]

    with open(path, "r") as f:
        lines = f.readlines()
        tag = "ARC_CHAN"

        for line in lines:
            if line[:len(tag)] == tag:

                station = line[40:45].strip()
                channel = line[45:48]
                code = line[48:50]
                location = line[50:52]
                start_date = line[53:69].strip()
                end_date = line[70:].strip()

                if multplt_data is not None:
                    if station + channel not in stations_channels:
                        continue

                parsed_line = [station, channel, code, location, start_date, end_date]
                data.append(parsed_line)

    d_stations = []
    data = process_archives_list(data)
    for x in data:
        d_stations.append(convert_station_group_to_dictionary(x))

    return d_stations


def date_str(year, month, day, hour=0, minute=0, second=0., microsecond=None):
    """
    Creates an ISO 8601 string.
    """
    # Get microsecond if not provided
    if microsecond is None:
        if type(second) is float:
            microsecond = int((second - int(second)) * 1000000)
        else:
            microsecond = 0

    # Convert types
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    microsecond = int(microsecond)

    # ISO 8601 template
    tmp = '{year}-{month}-{day}T{hour}:{minute}:{second}.{microsecond}'

    return tmp.format(year=year, month=month, day=day,
                      hour=hour, minute=minute, second=second, microsecond=microsecond)


def stretch_right(line, length, character=' ', trim=True):
    line = str(line)
    l_diff = length - len(line)
    if l_diff > 0:
        line = line + character*l_diff
    if l_diff < 0 and trim:
        line = line[:l_diff]
    return line


def stretch_left(line, length, character=' ', trim=True):
    line = str(line)
    l_diff = length - len(line)
    if l_diff > 0:
        line = character*l_diff + line
    if l_diff < 0 and trim:
        l_diff = -l_diff
        line = line[l_diff:]
    return line


def number_of_stations(group):
    """
    Culculates number of different stations in detection.
    """
    return len(set([x['station']['station'] for x in group]))


def hypocenter_line(group, datetime, params, location):
    """
    Returns first line of Nordic file.
    """
    line = ' '
    line += stretch_left(datetime.year, 4)
    line += ' '
    line += stretch_left(datetime.month, 2)
    line += stretch_left(datetime.day, 2)
    line += ' '  # fix origin time (normally blank)
    line += datetime.strftime('%H%M')
    line += ' '
    line += datetime.strftime(f'%S.%f')[:4]
    line += ' '
    line += location[0]
    line += ' '  # event id, whitespace means "presumed earthquake"

    latitude = ''
    longitude = ''
    line += stretch_left(latitude, 7)
    line += stretch_left(longitude, 8)

    depth = ''
    line += stretch_left(depth, 5)

    line += ' '  # depth indicator (could be 'F')
    line += ' '  # location indicator

    agency = 'SAK'
    line += stretch_right(agency, 3)  # agency

    n_stations = str(number_of_stations(group))
    line += stretch_left(n_stations, 3)

    rms = ''  # RMS of Time Residuals
    line += stretch_left(rms, 4)

    magnitude = ''
    line += stretch_left(magnitude, 4)
    magnitude_type = ' '  # L=ML, b=mb, B=mB, s=Ms, S=MS, W=MW, G=MbLg (not used by SEISAN), C=Mc
    line += magnitude_type
    line += stretch_right('', 3)

    line += stretch_left('', 4)  # magnitude no. 2
    line += stretch_left('', 1)  # magnitude no. 2 type
    line += stretch_left('', 3)  # magnitude no. 2 agency

    line += stretch_left('', 4)  # magnitude no. 3
    line += stretch_left('', 1)  # magnitude no. 3 type
    line += stretch_left('', 3)  # magnitude no. 3 agency

    line += '1'  # type of this line
    line += '\n'

    return line


def waveform_line(group, datetime, params, location, waveform):
    """
    Returns line with waveform filename.
    """
    line = ' '
    line += stretch_right(waveform, 78)  # name of file or archive reference, a-format
    line += '6'  # type of this line
    line += '\n'

    return line


def id_line(group, datetime, params, location):
    """
    Returns ID line.
    """
    line = ' '
    line += stretch_left('ACTION:', 7)  # help text for action indicator
    line += stretch_right('NEW', 3)  # last action done
    line += ' '

    now_datetime = UTCDateTime().strftime('%y-%m-%d %H:%M')
    line += stretch_right(now_datetime, 14)  # datetime of last action
    line += ' '

    line += stretch_left('OP:', 3)  # help text for operator
    operator = 'MLS'  # help text for operator
    line += stretch_right(operator, 4)
    line += ' '

    line += stretch_left('STATUS:', 7)  # help text for status
    line += stretch_left('', 14)  # status flags, not yet defined in Seisan
    line += ' '

    line += stretch_left('ID:', 3)  # help text for ID
    id = datetime.strftime('%Y%m%d%H%M%S')
    line += stretch_right(id, 14)  # event ID

    duplicate = ' '  # if initial ID was already taken, and had to create a different ID, then "d",
                     # otherwive " "
    line += duplicate
    line += ' '  # "L" - synced with origin time, " " - not synced

    line += stretch_left('I', 4)  # type of this line
    line += '\n'

    return line


def write_nordic_head(f, group, datetime, params, location, waveform):
    """
    Generates and writes Nordic file contents before wave detections table.
    """
    f.write(hypocenter_line(group, datetime, params, location))
    f.write(id_line(group, datetime, params, location))
    if waveform:
        f.write(waveform_line(group, datetime, params, location, waveform))


def detection_line(detection, datetime, params):
    """
    Generates a line for detections table entry.
    """
    line = ' '
    station = detection['station']['station']
    line += stretch_right(station, 5)

    instrument = detection['station']['components'][0][0]  # instrument Type S = SP, I = IP, L = LP, etc.
    line += instrument

    component = 'Z'  # component Z, N, E ,T, R, 1, 2
    line += component
    line += ' '

    quality = 'I'  # quality Indicator I, E, etc.
    line += quality

    phase = detection['type']  # phase ID: PN, PG, LG, P, S, etc.
    if phase in params['main', 'label-names']:
        line += stretch_right(params['main', 'label-names'][phase], 4)
    else:
        line += stretch_right(phase, 4)

    weight_indicator = ' '
    line += weight_indicator

    automatic = 'A'
    line += automatic

    first_motion = ' '  # first motion C, D
    line += first_motion
    line += ' '

    if datetime.day < detection['datetime'].day:
        hour = str(detection['datetime'].hour + 24)
    else:
        hour = detection['datetime'].strftime('%H')
    line += stretch_left(hour, 2, '0')
    line += stretch_left(detection['datetime'].strftime('%M'), 2, '0')
    line += ' '

    line += stretch_right(detection['datetime'].strftime('%S.%f')[:5], 5, ' ')
    line += ' '*5

    amplitude = ''
    line += stretch_right(amplitude, 7)
    line += ' '

    period_seconds = ''
    line += stretch_right(period_seconds, 4)
    line += ' '

    azimuth = ''
    line += stretch_right(azimuth, 5)
    line += ' '

    velocity = ''
    line += stretch_right(velocity, 4)
    incidence = ''  # angle of incidence
    line += stretch_right(incidence, 4)
    back_azimuth_residual = ''
    line += stretch_right(back_azimuth_residual, 3)
    travel_time_residual = ''
    line += stretch_right(travel_time_residual, 5)
    i2_weight = ''
    line += stretch_right(i2_weight, 2)
    distance_to_epicenter = ''
    line += stretch_right(distance_to_epicenter, 5)
    line += ' '

    azimuth_at_source = ''
    line += stretch_right(azimuth_at_source, 3)

    line += ' '  # type of this line
    line += '\n'

    return line


def write_phase_table(f, group, datetime, params, location):
    """
    Generates and writes Nordic file detections table.
    """
    # Write title line
    f.write(' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO AIN AR TRES W  DIS CAZ7\n')

    for record in group:
        f.write(detection_line(record, datetime, params))


def generate_event(group, datetime, params, waveform):
    """
    Generates s-file for a detection.
    """
    location = 'L'
    filename = datetime.strftime(f'%d-%H%M-%S{location}.S%Y%m')

    with open(filename, 'w') as f:
        write_nordic_head(f, group, datetime, params, location, waveform)
        write_phase_table(f, group, datetime, params, location)

    return filename


def unique_filename(name):
    """
    Returns unique file name based on "name" argument.
    """
    return name


def create_cbase_file(stations, params):
    """
    Creates file with list of archive channels to slice for -cbase option of wavetool command from
    Seisan. Returns path to generated file.
    -cbase file contains information about archive to slice from.
    File example:
    0        1         2         3         4
    123456789012345678901234567890123456789012
    NGLK SHZIM00        20140403  202112312359
    NGLK SHNIM00        20140403  202112312359
    NGLK SHEIM00        20140403  202112312359
    ARGI SHZIM00        20140403  201809062359
    ARGI SHNIM00        20140403  201809062359
    ARGI SHEIM00        20140403  201809062359

    First two lines are not present in actual file and just to represent a character index.
    Each line contains:
        1:5 - station
        6:8 - component
        9:10 - network
        11:12 - location
        13:29 - archive start date
        31:42 - archive end date in format YYYYMMDDhhmm
    Archive end date is mandatory for "wavetool", so if archive does not have an end date, it will be
    written as last minute of a current year: "YYYY12312359"
    Parameters:
    :param event - group of detections
    :param datetime - datetime of the group (usually, earliest detection time in the group)
    :param params - parameters of the application
    :param stations - list of stations continious archives to cut waveforms from
    """
    import datetime

    file_name = unique_filename('cbase.inp')
    with open(file_name, 'w') as f:
        for station in stations:
            for component in station['components']:
                line = stretch_right(station['station'], 5)
                line += stretch_right(component, 3)
                line += stretch_right(station['network'], 2)
                line += stretch_right(station['location'], 2)

                # get/generate archive start date
                start = station['start']
                if not start or not len(start):
                    start = '19600101'

                # get/generate archive end date
                end = station['end']
                if not end or not len(end):
                    end = datetime.datetime.now().strftime('%Y12312359')

                line += stretch_left(start, 17) + ' '
                line += stretch_right(end, 12)

                line += '\n'
                f.write(line)

    return file_name


def slice_waveforms_wavetool(event, datetime, params, stations):
    """
    Generates (slices) waveform miniSEED file for group of detections. Returns path to generated file.
    Uses wavetool program from Seisan.
    Parameters:
    :param event - group of detections
    :param datetime - datetime of the group (usually, earliest detection time in the group)
    :param params - parameters of the application
    :param stations - list of stations continious archives to cut waveforms from
    """
    cbase = create_cbase_file(stations, params)

    # Get first event time
    start_datetime = datetime - params['main', 'waveform-duration']/2
    s_start = start_datetime.strftime('%Y%m%d%H%M%S')

    import os
    wavetool_command = f'wavetool -format MSEED -start {s_start} -arc ' \
                       f'-duration {params["main", "waveform-duration"]}' \
                       f' -wav_out_file SEISAN -cbase {cbase}'

    if not params['main', 'silence-wavetool']:
        print(f'\n{wavetool_command}')

    wavetool_pipe = os.popen(wavetool_command)

    # Parse wavetool output
    import re
    wavetool_output = wavetool_pipe.read()
    if not params['main', 'silence-wavetool']:

        matches = re.findall(r'Error: .*',
                             wavetool_output)
        for x in matches:
            print(x)

        matches = re.findall(r'(?<=Number of archive channels defined).*',
                             wavetool_output)
        if len(matches):
            print('Number of archive channels defined: ', matches[0].strip())

        matches = re.findall(r'(?<=Total duration:).*',
                             wavetool_output)
        if len(matches):
            print('Total duration:', matches[0].strip())

    waveform_file = None
    matches = re.findall(r'(?<=Output waveform file name is).*\d{4}-\d{2}-\d{2}-\d{4}-\d{2}\w\..*',
                         wavetool_output)
    if len(matches):
        waveform_file = matches[0].strip('\x00 ')

    files_to_remove = [cbase, 'extract.mes', 'respfile_list.out']
    for x in files_to_remove:
        try:
            os.remove(x)
        except Exception:
            pass

    return waveform_file


def slice_waveforms_obspy(event, datetime, params, stations):
    """
    Generates (slices) waveform miniSEED file for group of detections. Returns path to generated file.
    Uses ObsPy library.
    Parameters:
    :param event - group of detections
    :param datetime - datetime of the group (usually, earliest detection time in the group)
    :param params - parameters of the application
    :param stations - list of stations continious archives to cut waveforms from
    """
    # Get first event time
    start_datetime = datetime - params['main', 'waveform-duration'] / 2
    end_datetime = start_datetime + params['main', 'waveform-duration']

    from os.path import isfile
    for station in stations:
        path = archive_to_path(station, start_datetime, params['main', 'archives'])


def slice_event_waveforms(event, datetime, params, stations):
    """
    Generates (slices) waveform miniSEED file for group of detections. Returns path to generated file.
    Parameters:
    :param event - group of detections
    :param datetime - datetime of the group (usually, earliest detection time in the group)
    :param params - parameters of the application
    :param stations - list of stations continious archives to cut waveforms from
    """
    if params['main', 'wavetool-waveforms']:
        return slice_waveforms_wavetool(event, datetime, params, stations)
    else:
        return slice_waveforms_obspy(event, datetime, params, stations)


def ask(question, default=None, validation=None):
    """
    Asks a question, expecting string input.
    :param question: Question to ask. Note, that ":" and default will be automatically appended to the question
    string.
    :param default: Default value if input is empty. If not set, will ask for to non-empty input.
    :param validation: Validation function: def valid_func(answer: str) -> bool.
    :return: bool
    """
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

def detection_station_list(event, params):
    """
    Returns station list with only stations with detections.
    """
    all_stations = params['main', 'stations']
    unique_stations = set([x['station']['station'] for x in event])
    unique_stations_list = []
    for station in all_stations:
        if station['station'] in unique_stations:
            unique_stations_list.append(station)

    return unique_stations_list


def register_event(s_file, waveform, datetime, database, params):
    import os
    import shutil
    # os.path.join(base, new)
    rea_path = params['main', 'rea']
    wav_path = params['main', 'wav']

    year = datetime.strftime('%Y')
    month = datetime.strftime('%m')

    rea_path = os.path.join(rea_path, database, year, month)
    wav_path = os.path.join(wav_path, database, year, month)

    # Check if path(s) are valid
    full_s_file_path = os.path.join(rea_path, s_file)
    if not os.path.isfile(s_file):
        print(f'Cannot register event: generated s-file {s_file} not found!', file=sys.stderr)
    if not os.path.isdir(rea_path):
        print(f'Cannot register event: directory {rea_path} is not found!', file=sys.stderr)
        return
    if os.path.isfile(full_s_file_path):
        print(f'Cannot register event: file {full_s_file_path} already exists!', file=sys.stderr)
        return
    full_waveform_path = None
    if waveform:
        full_waveform_path = os.path.join(wav_path, waveform)
        if not os.path.isfile(waveform):
            print(f'Cannot register event: waveform {waveform} not found!', file=sys.stderr)
        if not os.path.isdir(wav_path):
            print(f'Cannot register event: directory {wav_path} is not found!', file=sys.stderr)
            return
        if os.path.isfile(full_waveform_path):
            print(f'Cannot register event: file {full_waveform_path} already exists!', file=sys.stderr)
            return

    shutil.move(s_file, full_s_file_path)
    if full_waveform_path:
        shutil.move(waveform, full_waveform_path)

    print(f'Saved: {full_s_file_path} and {full_waveform_path}')


def ask_database_name(params):
    """
    Returns database name, based on user input or default value, based on --use-default-database argument.
    """
    if not params['main', 'use-default-database']:
        def database_validation(name):
            if len(name) > 5:
                print('Database name should be shorter than or exactly 5 characters long!')
                return False
            return True

        database = ask('Enter target database for events registering', default=params['main', 'database'],
                       validation=database_validation)

        # Make sure database name is 5 characters long!
        def string_filler(value, length=0, append=True, filler='_'):
            if len(value) < length:
                l_diff = length - len(value)
                if append:
                    return value + filler * l_diff
                else:
                    return filler * l_diff + value
            return value

        database = string_filler(database, 5)
    else:
        database = params['main', 'database']
    return database


def generate_events(events, params):
    """
    Generates s-files for detections.
    """
    groups_counter = 0
    for filename, groups in events.items():
        for group, datetime in groups:
            if len(group) >= params['main', 'detections-for-event']:
                groups_counter += 1

    print(f'\n\nPotential events detected: {groups_counter}')

    if not groups_counter:
        return

    b_events_generation = False
    if params['main', 'generate-s-files'] == 'yes':
        b_events_generation = True
    if params['main', 'generate-s-files'] == 'ask once':
        b_events_generation = ask_yes_no('Do you want to generate s-files for potential events?')

    b_waveforms_generation = False
    if params['main', 'generate-waveforms'] == 'yes':
        b_waveforms_generation = True
    if params['main', 'generate-waveforms'] == 'ask once':
        b_waveforms_generation = ask_yes_no('Do you want to extract waveforms for potential events?')

    stations_list = None
    if not params['main', 'detection-stations']:
        stations_list = params['main', 'stations']

    l_s_files = []
    l_waveforms = []
    saved_events = []
    for filename, groups in events.items():
        for group, datetime in groups:
            if len(group) >= params['main', 'detections-for-event']:

                c_saved_event = {}

                # Waveforms file generation
                if params['main', 'generate-s-files'] == 'ask each':
                    question = f'Do you want to generate s-file for potential event: ' \
                               f'{datetime.strftime("%Y%m%d%H%M%S")} ({len(group)} positives)?'
                    b_events_generation = ask_yes_no(question)
                if params['main', 'generate-waveforms'] == 'ask each':
                    question = f'Do you want to extract waveforms for potential event: ' \
                               f'{datetime.strftime("%Y%m%d%H%M%S")} ({len(group)} positives)?'
                    b_waveforms_generation = ask_yes_no(question)

                waveforms_name = None
                if b_waveforms_generation:
                    if params['main', 'detection-stations']:
                        stations_list = detection_station_list(group, params)
                    waveforms_name = slice_event_waveforms(group, datetime, params, stations_list)

                if waveforms_name:
                    l_waveforms.append(waveforms_name)
                    c_saved_event['waveform'] = waveforms_name

                if b_events_generation:
                    s_name = generate_event(group, datetime, params, waveforms_name)
                    l_s_files.append(s_name)
                    c_saved_event['s-file'] = s_name
                    c_saved_event['datetime'] = datetime
                    saved_events.append(c_saved_event)

    print('\nGenerated s-files:')
    for x in l_s_files:
        print(x)

    print('\nExtracted waveforms:')
    for x in l_waveforms:
        print(x)
    print()

    # Events registration in the database
    if params['main', 'register-events'] == 'no':
        return
    if not len(saved_events):
        return

    if not params['main', 'database']:
        print('Cannot register event(s): --database is not set and failed to autodetect!',
              file=sys.stderr)
        return
    if not params['main', 'rea']:
        print('Cannot register event(s): --rea is not set and failed to autodetect!',
              file=sys.stderr)
        return
    if not params['main', 'wav']:
        print('Cannot register event(s): --wav is not set and failed to autodetect!',
              file=sys.stderr)
        return

    b_register_event = True
    if params['main', 'register-events'] == 'yes':
        b_register_event = True
    if params['main', 'register-events'] == 'ask once':
        b_register_event = ask_yes_no('Do you want to register all generated events in the database?')

    if not b_register_event:
        return

    database = ask_database_name(params)
    print(f'Saving to the database {database}..')

    for d_event in saved_events:
        s_file = d_event['s-file']
        datetime = d_event['datetime']
        waveform = None
        if 'waveform' in d_event:
            waveform = d_event['waveform']

        id = datetime.strftime('%Y%m%d%H%M%S')

        if params['main', 'register-events'] == 'ask each':
            question = f'Do you want to register event: ' \
                       f'{id} (file: {s_file}) '
            if waveform:
                question += f'and waveform {waveform} '
            question += 'in the database?'

            b_register_event = ask_yes_no(question)

        if b_register_event:
            register_event(s_file, waveform, datetime, database, params)
            

def create_unique_file(path, mode):
    """
    Opens a file. Ensures that provided path leads to unique filename (by adding to a filename).
    :return file, path
    """
    # Split path into file names
    # Split path into extension and name
    n = 1
    if not os.path.isfile(path):
        return open(path, mode), path
    else:
        base_path, extension = os.path.splitext(path)
        while os.path.isfile(path):
            path = f'{base_path}_{n}{extension}'
            n += 1
        return open(path, mode), path


def generate_mulplt_def(path, stations, enforce_unique=False):
    """
    Generates MULPLT.DEF file based on the list of stations.
    Each station is a dictionary (see parse_seisan_def return).
    :param path: path to save generated file.
    :param stations: list of station dictionaries.
    :param enforce_unique: if True, will modify provided path, if it points to existing file.
                           if False, will rewrite the path.
    """
    prefix = 'DEFAULT CHANNEL '
    if enforce_unique:
        f, path = create_unique_file(path, 'w')
    else:
        f = open(path, 'w')

    with f:
        for x in stations:
            name = x['station']
            f.write(f'[{name}]\n')
            current_prefix = f'{prefix}{name}\t'
            for component in x['components']:
                f.write(f'{current_prefix}{component[:-1]} {component[-1]}\n')
            f.write('\n')
    return path


def get_s_files(date, rea, start=None, end=None):
    """
    Return list of all s-file paths for specified day, if start or end specified, then all s-files outside
        start-end span will be ignored.
    """
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
    s_path = os.path.join(rea, year, month)

    try:
        files = []
        for f in os.listdir(s_path):
            f_day = f[:2]
            if f_day == day:
                files.append(os.path.join(s_path, f))
    except FileNotFoundError:
        print(f'Skipping --false-positives for {year}-{month}-{day}: directory {s_path} not found!', file=sys.stderr)
        return None

    if not len(files):
        print(f'Skipping --false-positives for {year}-{month}-{day}: s-file not found!', file=sys.stderr)
        return None

    if start or end:
        filtered_files = []
        for f in files:
            # Extract event date
            f_basename = os.path.basename(f)

            f_hour = f_basename[3:5]
            f_minute = f_basename[5:7]
            f_second = f_basename[8:10]

            # Check if date-time is overflowed
            try:
                if int(f_hour) == 24:
                    f_hour = str(23)
                if int(f_minute) == 60:
                    f_minute = str(59)
                if int(f_second) == 60:
                    f_second = str(59)
            except ValueError as e:
                print(f'--false-positives skipping file "{f}": {e}', file=sys.stderr)
                continue

            # Compare dates
            utc_string = f'{year}-{month}-{day}T{f_hour}:{f_minute}:{f_second}'

            try:
                utc_datetime = UTCDateTime(utc_string)
            except ValueError as e:
                print(f'--false-positives skipping file "{f}": {e}', file=sys.stderr)
                continue

            if start and start > utc_datetime:
                continue
            if end and utc_datetime > end:
                continue
            filtered_files.append(f)
        files = filtered_files

    return files


def get_meta(lines):
    """
    Returns event metadata
    :param lines:
    :return:
    """
    head = lines[0]

    # Magnitude
    magnitude = head[55:59].strip()
    if len(magnitude):
        magnitude = float(magnitude)
    else:
        magnitude = None

    magnitude_type = head[59]

    # Locale
    loc = head[21]
    if loc != 'L':
        raise AttributeError(f'Unsupported locale type "{loc}"! Skipping..')

    # Depth
    depth = head[38:43].strip()  # in Km
    if len(depth):
        depth = float(depth)
    else:
        depth = None

    # Parse ID
    event_id = None
    q_id = re.compile(r'\bID:')
    for line in lines:

        f_iter = q_id.finditer(line)

        found = False
        for match in f_iter:
            span = match.span()

            if span == (57, 60):
                found = True
                break

        if not found:
            continue

        event_id = line[span[1] : span[1] + 14]
        break

    return magnitude, magnitude_type, loc, depth, event_id


def parse_detections(detections, year, month, day, event_id, magnitude):
    """
    Parses s-file station detections table and returns list of event objects.
    """
    events = []
    for i, line in enumerate(detections):

        if not len(line.strip()):
            continue

        # Read detection parameters
        try:
            station = line[1:6].strip()
            instrument = line[6]
            phase = line[10:14].strip()
            hour = int(line[18:20].strip())
            minute = int(line[20:22].strip())
            second = float(line[22:28].strip())
        except ValueError as e:
            continue

        # Filter events

        # Sometimes it's just more than 60 seconds in a minute with SEISAN..
        if second >= 60.:

            minute_add = second // 60
            second = (second % 60)

            minute += minute_add
            minute = int(minute)

        if minute >= 60:

            if hour != 23:
                minute = 0
                hour += 1
            else:
                minute = 59

            minute = int(minute)
            hour = int(hour)

        if hour >= 24:
            continue

        datetime = UTCDateTime(date_str(year, month, day, hour, minute, second))

        event = {
            'event_id': event_id,
            'station': station,
            'datetime': datetime,
            'phase': phase,
            'instrument': instrument
        }
        events.append(event)

    return events


def get_events_from_s_file(path, start=None, end=None):
    """
    Return all events (true positives) from specified s-file, if start or end specified, then all events outside
        start-end span will be ignored.
    Events returned as a list in which each pick represented as dictionary.
    Each dictionary fields represent a property of a pick, e.g.:
        'daytime', 'station', 'phase'
    """
    # Read file if possible
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return []
    except FileNotFoundError:
        return []

    # Parse event properties
    magnitude, magnitude_type, loc, depth, event_id = get_meta(lines)

    if event_id:
        year = int(event_id[:4])
        month = int(event_id[4:6])
        day = int(event_id[6:8])
    else:
        year = None
        month = None
        day = None

    # Find station detections table
    table_head = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO AIN AR TRES W  DIS CAZ7'
    detections_table = n_detections_table = None
    for i, l in enumerate(lines):

        if l[:len(table_head)] == table_head:
            detections_table = lines[i + 1:]
            n_detections_table = i + 1 + 1  # + 1 - because number should start with 1
    # Parse detections
    events = parse_detections(detections_table, year, month, day, event_id, magnitude)

    return events


def get_events(date, params, start=None, end=None):
    """
    Return all events (true positives) in specified day, if start or end specified, then all events outside
        start-end span will be ignored.
    Events returned as a list in which each event represented as dictionary.
    Each dictionary fields represent a property of an event, e.g.:
        'daytime', 'station', 'phase'
    """
    if not params['main', 'rea']:
        # TODO: this error should be moved to the beginning of the script, no exceptions at the very end of the script!
        raise AttributeError('"rea" option/parameter is not set! "rea" is required for database events reading!')

    database = ask_database_name(params)
    rea = os.path.join(params['main', 'rea'], database)
    files = get_s_files(date, rea, start, end)

    events = []
    for path in files:
        events.extend(get_events_from_s_file(path))

    return events
