import sys
import os


def print_time_batch(data, file=None):
    """
    Prints out data for --time-batch option.
    :param data:
    :param file: File to print into, stdout by default
    :return:
    """
    if not file:
        file = sys.stdout
    print('\nBatch prediction times:')
    for x in data:
        name = os.path.basename(x['archives']['paths'][-1])
        station = x['archives']['station']
        if station:
            print(f'{station["station"]}: {name}')
        else:
            print(f'{name}')
        for batch in x['batches']:
            time = float(batch['time'])
            print(f'---batch: {batch["id"]}, time: {time:.6} seconds')


def print_time_archive(data, file=None):
    """
    Prints out data for --time-archive option.
    :param data:
    :param file: File to print into, stdout by default
    :return:
    """
    if not file:
        file = sys.stdout
    pass
    print('\nArchive prediction times:')
    for x in data:
        name = os.path.basename(x['archives']['paths'][-1])
        station = x['archives']['station']
        if station:
            print(f'{station["station"]}: ', end='')
        time = float(x['time'])
        print(f'{name}, {time:.6} seconds')
