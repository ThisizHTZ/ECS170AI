from code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    dataset_name = 'ORL'  # 'MNIST' / 'CIFAR' / 'ORL'
    dataset_source_folder_path = './'
    train_data = None
    test_data = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _load_pickle_data(self):
        print(f'loading data for {self.dataset_name}...')

        with open(self.dataset_source_folder_path + self.dataset_name, 'rb') as f:
            data = pickle.load(f)

        return data

    def _process_instances(self, instances):
        X = []
        y = []

        for instance in instances:
            image_matrix = np.array(instance['image'], dtype=np.float32)
            image_label = instance['label']

            # normalization
            image_matrix = image_matrix / 255.0

            # MNIST: (28, 28) -> (1, 28, 28)
            if self.dataset_name == 'MNIST':
                image_matrix = np.expand_dims(image_matrix, axis=0)

            # CIFAR: (32, 32, 3) -> (3, 32, 32)
            elif self.dataset_name == 'CIFAR':
                image_matrix = image_matrix.transpose(2, 0, 1)

            # ORL: 原始是 (112, 92, 3)，但三个通道一样，取一个通道就够
            elif self.dataset_name == 'ORL':
                if len(image_matrix.shape) == 3:
                    image_matrix = image_matrix[:, :, 0]
                image_matrix = np.expand_dims(image_matrix, axis=0)

                # ORL 标签是 1~40，CrossEntropyLoss 更适合 0~39
                image_label = image_label - 1

            X.append(image_matrix)
            y.append(image_label)

        return {
            'X': np.array(X, dtype=np.float32),
            'y': np.array(y, dtype=np.int64)
        }

    def load_train(self):
        data = self._load_pickle_data()
        self.train_data = self._process_instances(data['train'])
        return self.train_data

    def load_test(self):
        data = self._load_pickle_data()
        self.test_data = self._process_instances(data['test'])
        return self.test_data