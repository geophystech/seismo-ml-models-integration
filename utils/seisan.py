import os
from os.path import isfile
from obspy import UTCDateTime


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


def get_archives(seisan, mulplt, archives, params):
    """
    Returns lists of lists of archive file names to predict on. Also saves stations information to
    params['main', 'stations'].
    :return:
    """
    mulplt_parsed = parse_multplt(mulplt)
    seisan_parsed = parse_seisan_def(seisan, multplt_data=mulplt_parsed)

    d_stations = []
    for x in seisan_parsed:
        d_stations.append(convert_station_group_to_dictionary(x))

    params['main', 'stations'] = d_stations

    start = params['main', 'start']
    end = params['main', 'end']

    c_date = start
    paths = []
    while c_date < end:
        paths.extend([archive_to_path(group, c_date, archives) for group in seisan_parsed])
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

    return process_archives_list(data)


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


def archive_to_path(arch, date, archives_path):
    """
    Converts archive entry (array of elements: [[station, channel, code, location, start_date, end_date],]
    to dictionary of this format: {"N": "data/20160610AZTRO/20160610000000.AZ.TRO.HHN.mseed",}
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
    chans = {}
    station = None
    loc_code = None
    net_code = None
    components = []
    start = None
    end = None

    for x in arch:
        # Find channel type
        ch_type = x[1][-1]

        chans[ch_type] = x[1]
        components.append(x[1])

        # Get metadata
        if not station:
            station = x[0]
        if not loc_code:
            loc_code = x[3]
        if not net_code:
            net_code = x[2]
        if not start:
            start = x[4]
        if not end:
            end = x[5]

        # Path to archive
        path = archives_path + '{}/{}/{}.{}.{}.{}.{}.{}'.format(x[2], x[0], x[0], x[2], x[3], x[1], year, julday)

        d_result[ch_type] = path

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

    latitude = '0.0'
    longitude = '0.0'
    line += stretch_left(latitude, 7)
    line += stretch_left(longitude, 8)

    depth = '0.0'
    line += stretch_left(depth, 5)

    line += 'F'  # depth indicator
    line += ' '  # location indicator

    agency = 'SAK'
    line += agency[:3]  # agency

    number_of_stations = str(0)[:3]  # calculate!
    line += stretch_left(number_of_stations, 3)

    rms = '0.0'  # RMS of Time Residuals
    line += stretch_left(rms, 4)

    magnitude = '0.0'
    line += stretch_left(magnitude, 4)
    magnitude_type = 'L'  # L=ML, b=mb, B=mB, s=Ms, S=MS, W=MW, G=MbLg (not used by SEISAN), C=Mc
    line += magnitude_type
    line += agency[:3]

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
    operator = 'mlsp'  # help text for operator
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

    instrument = 'E'  # instrument Type S = SP, I = IP, L = LP, etc.
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

    # Get all files before calling wavetool
    from os import listdir
    from os.path import isfile, join
    files_before = [f for f in listdir('.') if isfile(join('.', f))]

    os.system(f'wavetool -format MSEED -start {s_start} -arc -duration {params["main", "waveform-duration"]}'
              f' -wav_out_file SEISAN -cbase {cbase}')
    os.remove(cbase)

    # Find new file
    files_after = [f for f in listdir('.') if isfile(join('.', f))]
    new_files = []

    for file_a in files_after:
        if file_a not in files_before:
            new_files.append(file_a)

    if len(new_files) == 0:
        return None
    if len(new_files) == 1:
        return new_files[0]

    # Try to filter out Seisan-like file name
    filtered_files = []
    for x in new_files:
        file_name = x.split('/')[-1]

        if start_datetime.strftime('%Y-%m-%d-%H%M-%S') in file_name:
            filtered_files.append(x)

    if len(new_files) == 1:
        return new_files[0]
    return None


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
    pass


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


def ask_yes_no(question, repeat=False):
    """
    Asks a question with answer YES/NO. Returns True if YES, False otherwise.
    :param question - question to ask
    :param repeat - if True, will repeat a question until either positive or negative answer given,
        if False - any non-positive answer is treated as negative (no need to input exact negative answer, e.g.
        "NO" or "N", etc.)
    """
    print(question + ' [Y/N]: ', end='')
    answer = input()

    while True:
        answer = answer.strip().lower()
        if answer in ['y', 'yes']:
            return True
        if not repeat:
            return False
        if answer in ['n', 'no']:
            return False


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


def generate_events(events, params):
    """
    Generates s-files for detections.
    """
    groups_counter = 0
    for filename, groups in events.items():
        for group, datetime in groups:
            if len(group) >= params['main', 'detections-for-event']:
                groups_counter += len(groups)

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
    for filename, groups in events.items():
        for group, datetime in groups:
            if len(group) >= params['main', 'detections-for-event']:

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

                if b_events_generation:
                    s_name = generate_event(group, datetime, params, waveforms_name)
                    l_s_files.append(s_name)

    print('\nGenerated s-files:')
    for x in l_s_files:
        print(x)

    print('\nExtracted waveforms:')
    for x in l_waveforms:
        print(x)