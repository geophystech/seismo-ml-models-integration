from os.path import isfile, isdir
from utils.params import Params
import sys


def archive_scan_defaults(params):
    # Default env values
    default_seisan = ['data/SEISAN.DEF']
    default_mulplt = ['data/MULPLT.DEF']
    default_archives = ['data/archives/']

    # Check and apply
    if not params['main', 'input']:
        if not params['main', 'seisan']:
            for x in default_seisan:
                if not isfile(x):
                    continue
                params['main', 'seisan'] = x
                break
        if not params['main', 'mulplt-def']:
            for x in default_mulplt:
                if not isfile(x):
                    continue
                params['main', 'mulplt-def'] = x
                break
        if not params['main', 'archives']:
            for x in default_archives:
                if not isdir(x):
                    continue
                params['main', 'archives'] = x
                break


def archive_scan(args):
    # Read config
    params = None
    if type(args['main']['config']) is str:
        args['main']['config'] = [args['main']['config']]
    for x in args['main']['config']:
        if not isfile(x):
            continue
        params = Params(path=x, config=args, default_dictionary='config')
        break
    if not params:
        print('Config file not found, using only default values and command line arguments!', file=sys.stderr)
        params = Params(path=None, config=args, default_dictionary='config')

    # Apply default values if not set
    archive_scan_defaults(params)

    return params