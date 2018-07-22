import gzip
from pathlib import Path

import numpy as np

data_path = Path(__file__).parent / '..' / 'data'

train_images_file = data_path / 'train-images-idx3-ubyte.gz'
train_labels_file = data_path / 'train-labels-idx1-ubyte.gz'
test_images_file = data_path / 't10k-images-idx3-ubyte.gz'
test_labels_file = data_path / 't10k-labels-idx1-ubyte.gz'


def gz_to_npz(file):
    return Path(str(file)[:-3] + '.npz')


train_images_file_array = gz_to_npz(train_images_file)
train_labels_file_array = gz_to_npz(train_labels_file)
test_images_file_array = gz_to_npz(test_images_file)
test_labels_file_array = gz_to_npz(test_labels_file)


def read_int(f, size=1):
    return int.from_bytes(f.read1(size), 'big', signed=False)


def read_images(file, magic_number):
    print('Read images', str(file))
    with gzip.open(str(file)) as f:
        assert magic_number == read_int(f, 4)

        n_images = read_int(f, 4)
        n_rows = read_int(f, 4)
        n_cols = read_int(f, 4)

        images = []
        for n in range(n_images):
            data = np.reshape(np.frombuffer(f.read1(n_rows*n_cols), dtype=np.ubyte, count=n_rows*n_cols), (n_rows, n_cols))
            images.append(data)

    return images


def read_labels(file, magic_number, n_images):
    print('Read labels', str(file))
    with gzip.open(str(file)) as f:
        assert magic_number == read_int(f, 4)
        assert n_images == read_int(f, 4)

        labels = []

        for n in range(n_images):
            labels.append(read_int(f))

    return labels


def get_data():
    if not test_images_file_array.exists():
        print('Pre-extracted data does not exist... Creating data....')
        train_images = read_images(train_images_file, 2051)
        train_labels = read_labels(train_labels_file, 2049, len(train_images))
        test_images = read_images(test_images_file, 2051)
        test_labels = read_labels(test_labels_file, 2049, len(test_images))

        np.savez_compressed(str(train_images_file_array), data=train_images)
        np.savez_compressed(str(train_labels_file_array), data=train_labels)
        np.savez_compressed(str(test_images_file_array), data=test_images)
        np.savez_compressed(str(test_labels_file_array), data=test_labels)

    return np.load(train_images_file_array)['data'], \
        np.load(train_labels_file_array)['data'], \
        np.load(test_images_file_array)['data'], \
        np.load(test_labels_file_array)['data']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_img, train_lbl, test_img, test_lbl = get_data()

    plt.imshow(train_img[1], cmap='Greys')
    plt.title('Number: ' + str(train_lbl[1]))
    plt.show()
