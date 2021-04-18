import argparse
import os
import re

# Argument parsing
parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)

parser.add_argument('run', default = 'python archive_scan.py',
        help = 'Command to run. Better specified in quotes: "<run>"')
parser.add_argument('config_dir', 
        help = 'Path to configs directory.')
parser.add_argument('--reg', '--re', default = '\.ini$', 
        help = 'Regex for config files selection. Default: \\.ini$')
parser.add_argument('--args', '-a', default = '',
        help = 'Main script arguments. Specify in quotes: --args "<args>".')

args = parser.parse_args()
args = vars(args)

# Config dir
config_dir = args['config_dir']
if config_dir[-1] != '/':
    config_dir += '/'

# Regex config files
files = os.listdir(config_dir)

pattern = re.compile(args['reg'])
configs = []
rel_configs = []

for f in files:
    
    if len(pattern.findall(f)):
        configs.append(f'{config_dir}{f}')
        rel_configs.append(f)

# Form os commands
commands = []

for i, c in enumerate(configs):

    commands.append(f'{args["run"]} --config {c} {args["args"]} --output_file out_{rel_configs[i]}.txt')

# Run
print('COMMANDS: ', *commands, sep = '\n')

for run in commands:

    print(f'\n\nCOMMAND: {run}')
    os.system(run)
