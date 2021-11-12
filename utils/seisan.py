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


def get_archives(seisan, mulplt, archives, params):
    """
    Returns lists of lists of archive file names to predict on.
    :return:
    """
    mulplt_parsed = parse_multplt(mulplt)
    seisan_parsed = parse_seisan_def(seisan, multplt_data=mulplt_parsed)

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
        l_ordered = order_group(group['paths'], params[group['station'], 'channel-order'])
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
                entry = line[len(tag):].split()
                station = line[40:45].strip()
                channel = line[45:48]
                code = line[48:50]
                location = line[50:52]
                start_date = entry[2] if len(entry) >= 3 else None
                end_date = entry[3] if len(entry) >= 4 else None

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

    for x in arch:
        # Find channel type
        ch_type = x[1][-1]

        chans[ch_type] = x[1]

        # Get metadata
        if not station:
            station = x[0]
        if not loc_code:
            loc_code = x[3]
        if not net_code:
            net_code = x[2]

        # Path to archive
        path = archives_path + '{}/{}/{}.{}.{}.{}.{}.{}'.format(x[2], x[0], x[0], x[2], x[3], x[1], year, julday)

        d_result[ch_type] = path

    return {
        'paths': d_result,
        'station': station
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


def waveform_line(group, datetime, params, location):
    """
    Returns line with waveform filename.
    """
    line = ' '
    waveform_filename = '2021-04-01-1235-35S.IMGG__023'
    line += stretch_right(waveform_filename, 78)  # name of file or archive reference, a-format
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


def write_nordic_head(f, group, datetime, params, location):
    """
    Generates and writes Nordic file contents before wave detections table.
    """
    f.write(hypocenter_line(group, datetime, params, location))
    f.write(id_line(group, datetime, params, location))
    f.write(waveform_line(group, datetime, params, location))


def detection_line(detection, datetime, params):
    """
    Generates a line for detections table entry.
    """
    line = ' '
    station = detection['station']
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


def generate_event(group, datetime, params):
    """
    Generates s-file for a detection.
    """
    location = 'L'
    filename = datetime.strftime(f'%d-%H%M-%S{location}.S%Y%m')

    with open(filename, 'w') as f:
        write_nordic_head(f, group, datetime, params, location)
        write_phase_table(f, group, datetime, params, location)


def generate_events(events, params):
    """
    Generates s-files for detections.
    """
    groups_counter = 0
    for filename, groups in events.items():
        for group, datetime in groups:
            if len(group) > params['main', 'detections-for-event']:
                groups_counter += len(groups)

    # TODO: print message (this many groups found)
    # TODO: unless save-s-files == 'always' or 'never', (if save-s-files == 'ask' or None)
    #       ask for premission to save found events as s-files

    print(f'\nEvents detected: {groups_counter}')

    for filename, groups in events.items():
        for group, datetime in groups:
            if len(group) > params['main', 'detections-for-event']:
                generate_event(group, datetime, params)
