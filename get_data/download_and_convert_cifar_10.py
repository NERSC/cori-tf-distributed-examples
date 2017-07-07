import os
from os.path import join
from subprocess import call
import numpy as np
from sklearn import model_selection
import h5py

def extract_cifar_data():
    user_scratch_path = os.environ["SCRATCH"]
    data_base_dir = join(user_scratch_path, "cori-tf-distributed-examples-data", "cifar10")


    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)

    tarpath = join(data_base_dir,"cifar-10-python.tar.gz")

    print("Downloading...")
    if not os.path.exists(tarpath):
        call(
            "wget --output-document %s http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"% tarpath,
            shell=True
        )
        print("Downloading done.\n")
    else:
        print("Dataset already downloaded. Did not download twice.\n")

    print("Extracting...")
    raw_cifar_directory = "cifar-10-batches-py"
    raw_data_path = join(data_base_dir, raw_cifar_directory)
    if not os.path.exists(raw_data_path):
        call(
            "cd %s; tar -zxvf %s" % (data_base_dir, tarpath),
            shell=True
        )
        print("Extracting successfully done to %s" % raw_data_path)
    else:
        print("Dataset already extracted. Did not extract twice.\n")

    cifar_directory = join(raw_data_path, "cifar_hdf5")
    train_filename = os.path.join(cifar_directory, 'train.h5')
    test_filename = os.path.join(cifar_directory, 'test.h5')
    if not os.path.exists(cifar_directory):

        def unpickle(file):
            try:
                import cPickle as pickle
            except:
                import pickle
            fo = open(file, 'rb')
            dict_ = pickle.load(fo)
            fo.close()
            return dict_

        def shuffle_data(data, labels):
            data, _, labels, _ = model_selection.train_test_split(
                data, labels, test_size=0.0, random_state=42
            )
            return data, labels

        def load_data(train_batches):
            data = []
            labels = []
            for data_batch_i in train_batches:
                d = unpickle(
                    os.path.join(raw_data_path, data_batch_i)
                )
                data.append(d['data'])
                labels.append(np.array(d['labels']))
            # Merge training batches on their first dimension
            data = np.concatenate(data)
            labels = np.concatenate(labels)
            length = len(labels)

            data, labels = shuffle_data(data, labels)
            return data.reshape(length, 3, 32, 32), labels

        X, y = load_data(
            ["data_batch_{}".format(i) for i in range(1, 6)]
        )

        Xt, yt = load_data(["test_batch"])

        print("INFO: each dataset's element are of shape 3*32*32:")
        print('"print(X.shape)" --> "{}"\n'.format(X.shape))

        print("Data is fully loaded, now truly converting.")

        os.makedirs(cifar_directory)


        comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
        # Train
        with h5py.File(train_filename, 'w') as f:
            f.create_dataset('data', data=X, **comp_kwargs)
            f.create_dataset('label', data=y.astype(np.int_), **comp_kwargs)
        # Test
        with h5py.File(test_filename, 'w') as f:
            f.create_dataset('data', data=Xt, **comp_kwargs)
            f.create_dataset('label', data=yt.astype(np.int_), **comp_kwargs)

        print('Conversion successfully done to "{}".\n'.format(cifar_directory))
    else:
        print("Conversion was already done. Did not convert twice.\n")
    return train_filename, test_filename
if __name__ == "__main__":
    extract_cifar_data()