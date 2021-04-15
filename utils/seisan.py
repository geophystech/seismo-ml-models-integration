from obspy.core.utcdatetime import UTCDateTime


def parse_multplt(path):
    """
    Parses multplt.def file and returns list of lists like: [station, channel type (e.g. SH), channel (E, N or Z)].
    """
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        tag = "#DEFAULT CHANNEL"

        for line in lines:
            if line[:len(tag)] == tag:
                # entry[0] - station, ..[1] - type, ..[2] - channel
                entry = line[len(tag):].split()
                data.append(entry)
    return data


def group_by(l, column, comp_margin = None):
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
                if len(z) == 3:
                    result.append(z)
    return result


def parse_seisan_def(path, multplt_data = None, allowed_channels = None):
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
                is_channel_allowed = False
                for ch in allowed_channels:
                    if ch == channel[:len(ch)]:
                        is_channel_allowed = True

                if is_channel_allowed:
                    data.append(parsed_line)

    return process_archives_list(data)


def date_str(year, month, day, hour=0, minute=0, second=0., microsecond = None):
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


def archive_to_path(arch, date, archives_path, allowed_channel_dims):
    """
    Converts archive entry (array of elements: [[station, channel, code, location, start_date, end_date],]
    to dictionary of this format: {"N": "data/20160610AZTRO/20160610000000.AZ.TRO.HHN.mseed",}
    Path example: /seismo/archive/IM/LNSK/LNSK.IM.00.EHE.2016.100
                  <archive_dir>/<location>/<station>/<station>.<location>.<code>.<channel>.<year>.<julday>
    """
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

        if ch_type not in allowed_channel_dims:
            continue

        chans[ch_type] = x[1]

        # Get metadata
        if not station:
            station = x[0]
        if not loc_code:
            loc_code = x[3]
        if not net_code:
            net_code = x[2]

        # Path to archive
        path = archives_path + '/{}/{}/{}.{}.{}.{}.{}.{}'.format(x[2], x[0], x[0], x[2], x[3], x[1], year, julday)

        d_result[ch_type] = path

    d_result['meta'] = {'channels': chans,
                        'station': station,
                        'location_code': loc_code,
                        'network_code': net_code}

    return d_result
