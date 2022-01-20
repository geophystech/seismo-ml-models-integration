import h5py as h5


def add_dimension(data):
    """
    Prepends dimension to a data shape. Used for writing data batches to .h5 file.
    :param data:
    :return:
    """
    shape = (1,) + data.shape
    return data.reshape(shape)


def write_batch(path, dataset, batch, string=False):
    """
    Writes batch to h5 file
    :param path: Path to h5 file
    :param dataset: Name of the dataset
    :param batch: Data
    :param string: True - if data should be VL string
    """
    with h5.File(path, 'a') as file:

        first = True
        if dataset in file.keys():
            first = False

        if not first:
            file[dataset].resize((file[dataset].shape[0] + batch.shape[0]), axis=0)
            file[dataset][-batch.shape[0]:] = batch
        else:
            if not string:
                maxshape = list(batch.shape)
                maxshape[0] = None
                maxshape = tuple(maxshape)
                file.create_dataset(dataset, data=batch, maxshape=maxshape, chunks=True)
            else:
                dt = h5.string_dtype(encoding='utf-8')
                file.create_dataset(dataset, data=batch, maxshape=(None,), chunks=True, dtype=dt)
