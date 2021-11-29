import sys
from obspy import UTCDateTime
from utils.params import applied_function

# Default weights for models
default_weights = {'favor': 'weights/w_model_performer_with_spec.hd5',
                   'cnn': 'weights/w_model_cnn_spec.hd5',
                   'gpd': 'weights/w_gpd_scsn_2000_2017.h5'}

# Following functions will be applied to parameters to set default values, convert data types, etc.
def apply_default_model_name(model_name, params, key):
    station_params = params[key]
    if station_params['favor']:
        return 'favor'
    if station_params['cnn']:
        return 'cnn'
    if station_params['gpd']:
        return 'gpd'
    if station_params['model']:
        return 'custom-model'
    if params['main', 'model-name']:
        return None
    raise AttributeError('Model is not specified, and default model did not apply! If you see this message, '
                         'this is a bug!')


@applied_function(defaults=default_weights)
def apply_default_weights(weight, params, key, defaults):
    params = params[key]
    if weight:
        return weight
    if params['model-name'] is None:
        return None
    if params['model-name'] not in defaults:
        return None
    return defaults[params['model-name']]

# Type converters
def type_converter(value, _, __, f_type):
    if value is None:
        return None
    if type(value) is f_type:
        return value
    return f_type(value)

int_converter = applied_function(f_type=int)(type_converter)
float_converter = applied_function(f_type=float)(type_converter)

# String processing
def string_trimmer(value, _, __):
    if type(value) is not str:
        return value
    return value.strip()

def string_filler(value, _, __, length=0, append=True, filler='_'):
    if type(value) is not str:
        return value
    if len(filler) != 1:
        print('WARNING: In string_filler length of "filler" parameter is not equal to 1! '
              'Skipping function!', file=sys.stderr)
        return value
    if len(value) < length:
        l_diff = length - len(value)
        if append:
            return value + filler*l_diff
        else:
            return filler*l_diff + value
    return value
def bool_converter(value, _, __):
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is str:
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            return None
    return bool(value)

def favor_default(value, params, key):
    params = params[key]
    if not params['cnn'] and not params['gpd'] and not params['model']:
        return True

def utc_datetime_converter(date, _, __):
    """
    Parse parameter from dictionary to UTCDateTime type.
    """
    if date is None:
        return None
    if not date:
        return date
    try:
        return UTCDateTime(date)
    except Exception as e:
        return None

def start_date_default(value, _, __):
    if value is None:
        return None
    if not value:
        date = UTCDateTime()
        return UTCDateTime(f'{date.year}-{date.month}-{date.day}')
    return value

def end_date_default(value, params, key):
    params = params[key]
    if value is None:
        return None
    if not value:
        date = UTCDateTime()
        date = UTCDateTime(f'{date.year}-{date.month}-{date.day}')
        if params['start']:
            date = params['start']
        return date + 24 * 60 * 60 - 0.000001
    return value

def trace_size_converter(value, params, key):
    """
    Converts trace size from seconds to samples
    """
    params = params[key]
    if value is None:
        return None
    if not value:
        return value
    if not params['frequency']:
        raise AttributeError('No frequency specified for trace-size argument!')
    return int(float(value) * params['frequency'])

def threshold_converter(value, _, __):
    if value is None:
        return None

    # Set label variables
    positive_labels = {'p': 0, 's': 1}

    # Parse and validate thresholds
    threshold_labels = {}
    global_threshold = False
    if type(value) is str:

        split_thresholds = value.split(',')

        if len(split_thresholds) == 1:
            value = float(value)
            global_threshold = True
        else:
            for split in split_thresholds:

                label_threshold = split.split(':')
                if len(label_threshold) != 2:
                    print('ERROR: Wrong --threshold format. Hint:'
                          ' --threshold "p: 0.95, s: 0.9901"', file=sys.stderr)
                    sys.exit(2)

                threshold_labels[label_threshold[0].strip()] = float(label_threshold[1])
    else:
        value = float(value)
        global_threshold = True

    if global_threshold:
        for label in positive_labels:
            threshold_labels[label] = value
    else:
        positive_labels_error = False
        if len(positive_labels) != len(threshold_labels):
            positive_labels_error = True

        for label in positive_labels:
            if label not in threshold_labels:
                positive_labels_error = True

        if positive_labels_error:
            sys.stderr.write('ERROR: --threshold values do not match positive_labels.'
                             f' positive_labels contents: {[k for k in positive_labels.keys()]}')
            sys.exit(2)

    return threshold_labels

def channel_order_converter(value, _, __):
    if value is None:
        return value
    return [x.split(',') for x in value.strip(',;').split(';')]


# ----------------------------------------------------------------------------------------
# Applied functions mappings

def archive_scan():
    """
    Returns converted to a dictionary of functions to apply to parameters of archive_scan.py
    """
    # Dictionary of default values setter, type converters and other applied functions
    d_applied_functions = {
        'favor': [bool_converter, favor_default],
        'cnn': [bool_converter],
        'gpd': [bool_converter],
        'model-name': [apply_default_model_name],
        'weights': [apply_default_weights],
        'features-number': [int_converter],
        'waveform-duration': [float_converter],
        'start': [utc_datetime_converter, start_date_default],
        'end': [utc_datetime_converter, end_date_default],
        'threshold': [threshold_converter],
        'batch-size': [int_converter],
        'frequency': [float_converter],
        'trace-size': [float_converter, trace_size_converter],
        'shift': [int_converter],
        'generate-s-files': [string_trimmer],
        'detections-for-event': [int_converter],
        'generate-waveforms': [string_trimmer],
        'no-filter': [bool_converter],
        'no-detrend': [bool_converter],
        'trace-normalization': [bool_converter],
        'wavetool-waveforms': [bool_converter],
        'detection-stations': [bool_converter],
        'plot-positives': [bool_converter],
        'silence-wavetool': [bool_converter],
        'plot-positives-original': [bool_converter],
        'print-scores': [bool_converter],
        'print-precision': [int_converter],
        'combine-events-range': [float_converter],
        'time': [bool_converter],
        'cpu': [bool_converter],
        'print-files': [bool_converter],
        'channel-order': [channel_order_converter],
    }

    return d_applied_functions