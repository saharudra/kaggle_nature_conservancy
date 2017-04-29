import numpy as np
import cv2
from chainer.datasets.tuple_dataset import TupleDataset
import csv
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

""""
1. Create map of className and classNumber
2. A list of tuples of training data (index, name, abs_path, classNumber)
3. create tuple(array(image_array, dtype=float32), class ).
4. Take a list of these tuples in TupleDataset (Maintain TupleDataset semantics)
"""

training_file_name = "training_file.csv"
test_file_name = "test_file.csv"


def get_im_cv2(path):
    img = cv2.imread(path)
    img = rgb2gray(img)
    # resized = np.reshape(img, (np.product(img.shape),))
    resized = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    # plt.imshow(resized)
    # plt.show()
    return resized


def create_train_image_tuple(image_path):
    image = get_im_cv2(image_path)
    image = np.asarray(image, dtype=float)
    resized = np.reshape(image, (np.product(image.shape),))
    return resized


def create_test_image_tuple(image_path):
    image = get_im_cv2(image_path)
    image = np.asarray(image, dtype=float)
    resized = np.reshape(image, (np.product(image.shape),))
    return resized


def make_train_tuple_dataset(image_tuples):
    images = []
    labels = []
    for image_tuple in image_tuples:
        # print(image_tuple)
        images.append(create_train_image_tuple(image_tuple[2]))
        labels.append(int(image_tuple[3]))
    return TupleDataset(images, labels)


def make_test_tuple_dataset(image_tuples):
    images = []
    image_names = []
    for image_tuple in image_tuples:
        images.append(create_test_image_tuple(image_tuple[0]))
        image_names.append(image_tuple[1])
    return TupleDataset(images, image_names)


def read_from_csv(file_name):
    image_info_tuples = []
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            image_info_tuples.append(line)
    return image_info_tuples
