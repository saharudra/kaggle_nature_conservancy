""""
1. Create map of className and classNumber
2. A list of tuples of training data (index, name, abs_path, classNumber)
3. create tuple( (array(image array), dtype=float32), class ).
4. Take a list of these tuples in TupleDataset (Maintain TupleDataset semantics)
"""

import os
import numpy as np

class_name_number_map = {'ALB': 0, 'BET': 1, 'DOL': 2, 'LAG': 3, 'NoF': 4, 'OTHER': 5, 'SHARK': 6, 'YFT': 7}
train_directory = os.getcwd() + '/../../HalfLifeBlueShift/train/train/'
test_directory = os.getcwd() + '/../../HalfLifeBlueShift/test'


def train_fetch_file(dictionary, train_directory):
    train_file_handle = open('training_file.csv', 'wb')
    fish_directory_list = [x[0] for x in os.walk(train_directory)]
    # print(fish_directory_list)
    index = 0
    for directory_i in range(1, len(fish_directory_list)):
        current_fish_type = fish_directory_list[directory_i].split('/')[-1]
        current_range = len(os.listdir(fish_directory_list[directory_i] + '/'))
        for fish_i in os.listdir(fish_directory_list[directory_i] + '/'):
            abs_path = fish_directory_list[directory_i] + '/' + fish_i
            current_array = [str(index), fish_i, abs_path, str(class_name_number_map[current_fish_type])]
            train_file_handle.write(",".join(current_array))
            train_file_handle.write("\n")
            index += 1
    train_file_handle.close()


def test_fetch_file(dictionary, test_directory):
    train_file_handle = open('test_file.csv', 'wb')
    for (dirpath, dirnames, filenames) in os.walk(test_directory):
        for file in filenames:
            f = dirpath + "/" + file + "," + file

            train_file_handle.write(f)
            train_file_handle.write("\n")
    train_file_handle.close()
    # fish_directory_list = [x[0] for x in os.walk(test_directory)]
    # # print(fish_directory_list)
    # index = 0
    # for directory_i in range(1, len(fish_directory_list)):
    #     current_fish_type = fish_directory_list[directory_i].split('/')[-1]
    #     current_range = len(os.listdir(fish_directory_list[directory_i] + '/'))
    #     for fish_i in os.listdir(fish_directory_list[directory_i] + '/'):
    #         abs_path = fish_directory_list[directory_i] + '/' + fish_i
    #         current_array = [str(index), fish_i, abs_path, str(class_name_number_map[current_fish_type])]
    #         train_file_handle.write(",".join(current_array))
    #         train_file_handle.write("\n")
    #         index += 1
    # train_file_handle.close()

if __name__ == '__main__':
    # train_fetch_file(class_name_number_map, train_directory)
    test_fetch_file(class_name_number_map, test_directory)
