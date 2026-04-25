'''
Concrete IO class for stage 2 dataset
train.csv / test.csv
Each line: label, feature1, feature2, ..., feature784
'''

from Base_Classes import dataset
import numpy as np
import os


class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    normalize = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _load_file(self, file_name):
        print(f'loading data from {file_name}...')
        X = []
        y = []

        full_path = os.path.join(self.dataset_source_folder_path, file_name)
        with open(full_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                elements = [int(i) for i in line.split(',')]
                y.append(elements[0])      # first column = label
                X.append(elements[1:])     # remaining 784 columns = features

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        if self.normalize:
            X = X / 255.0

        return {'X': X, 'y': y}

    def load_train(self):
        return self._load_file(self.train_file_name)

    def load_test(self):
        return self._load_file(self.test_file_name)