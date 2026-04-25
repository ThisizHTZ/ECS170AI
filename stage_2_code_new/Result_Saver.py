'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

'''
Concrete ResultModule for saving stage 2 results
'''

from Base_Classes import result
import pickle
import os

class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)

        full_path = os.path.join(self.result_destination_folder_path, self.result_destination_file_name)
        with open(full_path, 'wb') as f:
            pickle.dump(self.data, f)