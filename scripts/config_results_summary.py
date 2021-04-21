import os
import re
import argparse
from find_station_magnitude import parse_s_file


def print_s(path, stations):
    """
    Prints out s-file stations arrival data.
    """
    # Read file
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()

    th = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO AIN AR TRES W  DIS CAZ7'
    is_table = False
    for l in lines[1:]:

        if is_table:
            station = l[1:6].strip()

            if len(station) and station in stations:
                print(l[1:], end = '')

        if not is_table and l[:len(th)] == th:
            print('GROUND TRUTH')
            print(l[1:], end = '')
            is_table = True


def parse_scan(path):
    """
    Read archive_scan.py output file and returns list out output entries.
    """
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()

    parsed = []
    for l in lines:

        l_split = l.split()

        if l_split[0] not in ['P', 'S']:
            continue

        parsed.append({'station': l_split[2], 'line': l})
        
    return parsed


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument('s_file',
            help = 'Path to event s-file.')
    parser.add_argument('--out', '-o', default = 'config_run_summary.txt',
            help = 'Output file path.')
    parser.add_argument('--input', '-i', default = './',
            help = 'Path to input data (archive_scan.py output).')
    parser.add_argument('--regex', '--re', default = r'\.txt$',
            help = 'Input files RegEx')

    args = parser.parse_args()
    args = vars(args)

    s_parsed = parse_s_file(args['s_file'])

    # Print S-file
    stations = s_parsed['stations']
    print(end = '\n')
    print_s(args['s_file'], stations)

    # Parse input files
    if args['input'][-1] != '/':
        args['input'] += '/'

    files = os.listdir(args['input'])

    q = re.compile(args['regex'])

    for f in files:
        
        if not len(q.findall(f)):
            continue

        i_parsed = parse_scan(args['input'] + f)

        if len(i_parsed):
            print('\n\n', f)

        for x in i_parsed:

            if x['station'] in stations:
                print(x['line'], end = '')
