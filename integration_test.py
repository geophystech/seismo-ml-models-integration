"""
This script takes in file with outputs and compares it with s-file(s) with set thresholds.
"""
from argparse import ArgumentParser
import re


def parse_s_file(path):
    """
    Parses a single path to s-file. Returns dictionary of lists: {station: [events]}
    """
    # Check if this is a text file

    with open(path, 'r') as f:
        content = f.readlines()

    event_date_pattern = re.compile(r'^\s*\d{4}-\d{2}-\d{2}-\d{4}-\d{2}[\w\.]*')
    detection_pattern = re.compile(r'^\s*[\w\s]{3,5}[A-Za-z\s\d]{12}'
                                   r'((\s[0-9]{1})|([0-9]{2}))((\s[0-9]{1})|([0-9]{2}))'
                                   r'\s{1,2}[0-9]{1,2}\.[0-9]{1,2}.*\d{1,6}\s*$')

    events = {}
    for line in content:

        date_match = event_date_pattern.match(line)
        detection_match = detection_pattern.match(line)

        if date_match:
            pass  # parse the date and store it as current date
        elif detection_match:
            pass  # parse the detection line and add it to events list

    return events


def parse_s_dir(path):
    """
    Parses a single path to s-directory. Returns dictionary of lists: {station: [events]}
    """
    # Get list of files
    files = []

    # Parse files
    events = {}
    for f in files:

        parsed_events = read_single_s_file(f)

        for station, data in parsed_events.items():
            if station not in events:
                events[station] = data
            else:
                events[station].extend(data)

    return events


def parse_s_path(path):
    """
    Parses a single path to either s-file or s-directory. Returns dictionary of lists: {station: [events]}
    """
    # Is path - directory
    if False:
        return parse_f_dir(path)
    else:
        return parse_s_file(path)


if __name__ == '__main__':

    # Read command-line arguments (config file, path to s-file(s)), path to prediction(s)
    parser = ArgumentParser()
    parser.add_argument('-p', '--predictions', help='Path(s) predictions',
                        type=str, action='append')
    parser.add_argument('-e', '--events', help='Path(s) to s-files or directory(s) with s-files',
                        type=str, action='append')
    parser.add_argument('-r', '--range', help='Range (in seconds) within actual even in which '
                                              'prediction counts as true, default: 6 seconds',
                        type=float, default=6.)
    args = parser.parse_args()

    # Load config file same way as in archive_scan.py (but without command line arguments)
    threshold = 0.95

    # Read s-file(s)
    events = {}
    for path in args.events:
        parsed_events = parse_s_path(path)

        for station, data in parsed_events.items():
            if station not in events:
                events[station] = data
            else:
                events[station].extend(data)

    # Sort event by date

    # Read prediction(s)

    # Compare results

