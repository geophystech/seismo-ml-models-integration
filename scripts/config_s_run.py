import os

# TODO: run scripts based on config files and on s file names
#   basically: parse s-files catalog <year>/<month>/<file name with day and hour and minute>
#   and get --start_date and --end_date arguments from that.
#   By default - get --start_date and --end_date arguments from +-1 hour around event time,
#   but make it possible to set time padding via argument.
#   Just duplicate most of the code from config_run.py and build on top of that.
#   ADD HELP MESSAGE COMMAND DESCRIPTION (now it's only arguments description, not command) for every script, including archive_scan.py.
#   This script should save results in a single directory, unlike s-files. Just encode dates in names.
#   ALSO!!! To archive_scan.py: add some message with starting ending dates to output file. 
#   this will make output more clear, it will also allow for better merge for multiple script runs into one output file
#   ALSO make it generate it's own MULPLT.DEF based on stations in s-file, and on their description in SESISMO.DEF (only 3 channel allowed)
