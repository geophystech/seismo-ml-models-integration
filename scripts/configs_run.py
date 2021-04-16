import argparse
import os
import pathlib

# Argument parsing
# ../archive_scan.py ../test/ --reg config_*.ini --args "--start_time 14-04-2020 --end_time 15-04-2020"
parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)

parser.add_argument('run', help = 'Command to run. Better specified in quotes: "<run>"')
parser.add_argument('configs_dir', help = 'Path to configs directory.')
parser.add_argument('--reg', '--re', help = 'Regex for config files selection. Default: *.ini.')
parser.add_argument('--args', '-a', help = 'Main script arguments. Specify in quotes: --args "<args>".')

args = parser.parse_args()
args = vars(args)

print(f'WORK DIR: {pathlib.Path().absolute()}')

files = os.listdir(args['configs_dir'])
print(f'FILSE: {files}')

