import numpy as np
import matplotlib.pyplot as plt
import obspy


def check_data_type(data):
    """
    Returns string representation (name) of a data type. Internally used by other module functions.
    :param data:
    :return:
    """
    max_channels = 16
    types_hint = 'Try using one of this data types:\n' \
                 '--- [obspy.Stream, ...] - list of obspy.Stream(s) (each stream should contain only 1 trace)\n' \
                 '--- [obspy.Trace, ...] - list of obspy.Trace(s)\n' \
                 '--- NumPy array with shape (n_samples, n_channels)\n' \
                 '--- [array, ...] - list of NumPy array, each array represents single channel (shape (n_samples))\n'
    if type(data) is list:
        x_type = 'none'
        if len(data) > max_channels:
            raise AttributeError('Failed to understand data type!\n' + types_hint)
        for x in data:
            if type(x) is obspy.Stream and (x_type == 'none' or x_type == 'stream'):
                if len(x) > 1:
                    raise AttributeError('Failed to understand data type!\n' + types_hint)
                x_type = 'stream'
            elif type(x) is obspy.Trace and (x_type == 'none' or x_type == 'trace'):
                x_type = 'trace'
            elif type(x) is np.ndarray and (x_type == 'none' or x_type == 'numpy'):
                if len(x.shape) > 1:
                    raise AttributeError('Failed to understand data type!\n' + types_hint)
                x_type = 'numpy'
            else:
                raise AttributeError('Functions check_data_type failed to understand data type!\n' + types_hint)
        return f'list_{x_type}'
    if type(data) is np.ndarray and len(data.shape) == 2 and data.shape[1] <= max_channels:
        return 'numpy'
    raise AttributeError('Failed to understand data type!\n' + types_hint)


def convert_data_to_np(data):
    d_type = check_data_type(data)

    if d_type == 'numpy':
        return data
    if d_type == 'list_numpy':
        return np.array(data)
    if d_type == 'list_stream' or d_type == 'list_trace':
        if d_type == 'list_stream':
            data = [channel[0].data for channel in data]
        else:
            data = [channel.data for channel in data]
        n_channels = len(data)
        min_length = min([x.shape[0] for x in data])
        converted_data = np.zeros((min_length, n_channels))
        for i, channel_data in enumerate(data):
            converted_data[:, i] = channel_data[:min_length]
        return converted_data


def plot_wave(data, basename):
    try:
        data = convert_data_to_np(data)
    except AttributeError as e:
        print('plot_wave failed with error:\n', e, sep='')
        return

    n_samples = data.shape[0]
    n_channels = data.shape[1]

    # Determine plot horizontal length
    min_n_samples = 500
    max_n_samples = 10000
    if n_samples > max_n_samples: n_samples = max_n_samples
    if n_samples < min_n_samples: n_samples = min_n_samples

    k_scale = 0.004
    x_size = n_samples * k_scale

    fig = plt.figure(figsize=(x_size, 4.), dpi=300)

    axes = fig.subplots(n_channels, 1, sharex=True)

    # Plot
    for i in range(n_channels):
        axes[i].plot(data[:, i], linewidth=1., color='#000')

    plt.savefig(f'{basename}.jpg')
    plt.close()
