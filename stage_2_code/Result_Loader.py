'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

'''
Concrete ResultModule for loading stage 2 results
'''

from code.base_class.result import result
import pickle


class Result_Loader(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def load(self):
        print('loading results...')
        full_path = self.result_destination_folder_path + self.result_destination_file_name
        with open(full_path, 'rb') as f:
            self.data = pickle.load(f)