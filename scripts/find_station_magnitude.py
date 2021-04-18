import os
import os.path as pt
import argparse
import re
import sys

# TODO: script should go through specified directory and return path to every s-file which has atleast
#   one station from specified MULTPLT.DEF
#   and magnitude >= specifiei\

# Argument parsing 
parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)

parser.add_argument('s_dir',
        help = 'S-files year directory or month directory path.')
parser.add_argument('mulplt',
        help = 'Path to MULPLT.DEF file with list of stations.')
parser.add_argument('--magnitude', '--mag', '-m', default = 3., type = float,
        help = 'Minimal event magnitude allowed')
parser.add_argument('--out', '-o', default = 'stations.txt',
        help = 'Output path for s-files picking.')

args = parser.parse_args()
args = vars(args)

# Append '/' to path
if args['s_dir'][-1] != '/':
    args['s_dir'] += '/'

# Read MULPLT.DEF
stations = []
try:
    with open(args['mulplt'], 'r') as f:
        lines = f.readlines()
        tag = '#DEFAULT CHANNEL'

        for line in lines:

            if line[:len(tag)] == tag:
                entry = line[len(tag):].split()
                
                if entry[0] not in stations:
                    stations.append(entry[0])
except FileNotFoundError:
    print(f'MULPLT.DEF file: "{args["mulplt"]}" not found!')
    sys.exit(0)

# TODO: check if args['out'] exists and create if not.

# Parse s-files directory
def parse_s_file(f_name):
    """
    Parses s-file and returns dictionary with 'magnitude' and 'stations' list.
    """
    lines = []
    with open(f_name, 'r') as f:
        lines = f.readlines()

    if not len(lines):
        return

    # Magnitude parse
    magnitude = float(lines[0][55:59])
    magnitude_type = lines[0][59]

    if magnitude_type != 'L':
        print(f'In file "{f_name}": unsupported magnitude type "{magnitude_type}"! Skipping..')
        return

    # Stations parse
    th = ' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU VELO AIN AR TRES W  DIS CAZ7'
    stations = []

    is_table = False
    for l in lines[1:]:

        if is_table:
            s = l[1:6].strip()

            if len(s) and s not in stations:
                stations.append(s)

        if not is_table and l[:len(th)] == th:
            is_table = True

    return {'magnitude': magnitude, 'stations': stations} 


def parse_s_dir(s_dir, out, stations, magnitude):
    """
    Parses s_files directory and copies them in out directory if stations and magnitude check passed
    """
    files = os.listdir(s_dir)

    if not len(files):
        return
    
    print(f'PARSE: {s_dir}')

    for f in files:
        
        # Check if file name looks like s-file name
        q = re.compile(r'^\d{2}-\d{4}-\d{2}[L,R,D]\.S\d{6}$')
        if not len(q.findall(f)):
            continue

        # Parse file
        s_dict = parse_s_file(s_dir + f)

        # And check if it passes conditions
        if s_dict['magnitude'] < magnitude:
            continue
        
        l_stations = []
        for s in s_dict['stations']:
            
            if s in stations:
                l_stations.append(s)

        if not len(l_stations):
            continue

        # Output
        l_stations.sort()

        with open(out, 'a') as o:
            o.write(f'{s_dir + f} {s_dict["magnitude"]} {len(l_stations)}: {", ".join(l_stations)}\n')


def is_s_dir(path):
    """
    Returns True if provided path could be s-files monthly catalog. Returns False otherwise.
    """
    base = pt.basename(path)
    
    if not pt.isdir(path):
        return False

    q = re.compile(r'^\d{2}$')
    if not len(q.findall(f)):
        return False

    return True

# Check if main s_dir is monthly or annual
try:
    files = os.listdir(args['s_dir'])
except FileNotFoundError:
    print(f'Connot find s_dir: "{args["s_dir"]}"!')
    sys.exit(0)
except NotADirectoryError:
    print(f's_dir: "{args["s_dir"]}" is not a directory!')
    sys.exit(0)

if not len(files):
    print(f's_dir: "{args["s_dir"]}" is empty!')
    sys.exit(0)

is_annual = True
for f in files:
    
    path = args['s_dir'] + f + '/'
    if not is_s_dir(path):
        is_annual = False
        break

# If month catalog
if not is_annual:
    parse_s_dir(args['s_dir'], args['out'], stations, args['magnitude'])
# If annual catalog
else:
    for f in files:
        
        path = args['s_dir'] + f + '/'
        if is_s_dir(path):
            parse_s_dir(path, args['out'], stations, args['magnitude'])

