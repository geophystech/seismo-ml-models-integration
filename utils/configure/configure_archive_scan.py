"""
This module primary function is configure() which will try to generate a new configuration file for
the archive_scan.py script run.
"""
import os


def ask(question, default=None, validation=None):
    if default:
        print(question, f' [{default}]: ', sep='', end='')
        while True:
            answer = input().strip()
            if validation and not validation(answer):
                continue
            if answer:
                return answer
            return default
    if not default:
        print(question, ': ', sep='', end='')
        while True:
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


def configure():
    print('Running archive_scan.py configuration script!')
    configure_unix()
