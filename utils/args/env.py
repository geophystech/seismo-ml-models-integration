import os
from ..seisan import parse_seisan_params

def parse_unix(params):
    """
    Parses environment variables in UNIX systems and passes them to params, if not set earlier
    (through config file or command line arguments).
    """
    seisan_top = os.environ.get('SEISAN_TOP')
    default_database = os.environ.get('DEF_BASE')

    if not params.key_exists(('main', 'database')):
        params['main', 'database'] = default_database

    if not seisan_top:
        return

    seisan_path = os.path.join(seisan_top, 'DAT/SEISAN.DEF')
    mulplt_path = os.path.join(seisan_top, 'DAT/MULPLT.DEF')
    rea_path = os.path.join(seisan_top, 'REA')
    wav_path = os.path.join(seisan_top, 'WAV')

    parse_seisan_params(seisan_path, params)

    if not params.key_exists(('main', 'seisan')):
        params['main', 'seisan'] = seisan_path
    if not params.key_exists(('main', 'mulplt-def')):
        params['main', 'mulplt-def'] = mulplt_path
    if not params.key_exists(('main', 'rea')):
        params['main', 'rea'] = rea_path
    if not params.key_exists(('main', 'wav')):
        params['main', 'wav'] = wav_path


def archive_scan(params):
    parse_unix(params)