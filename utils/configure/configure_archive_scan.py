"""
This module primary function is configure() which will try to generate a new configuration file for
the archive_scan.py script run.
"""
import os


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


def ask(question, default=None, validation=None):
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


def configure_unix():
    seisan_top = os.environ.get('SEISAN_TOP')
    seisan_top = ask('Enter path to top seisan directory (which contains DAT, REA, WAV, ect.)',
                     seisan_top)

    default_database = os.environ.get('DEF_BASE')
    default_database = ask('Enter default database name (5 or shorter characters)',
                           default_database)

    # Parse stations
    from ..seisan import parse_seisan_def
    seisan = os.path.join()


def configure():
    print('Running archive_scan.py configuration script!')
    configure_unix()
