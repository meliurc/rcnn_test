import os
import pandas as pd

image_file_path = "./images/VOCdevkit/VOC2007/JPEGImages"
image_class_info_file_path = './images/VOCdevkit/VOC2007/ImageSets/Main'


def gen_dataset_file(image_class_info_file_path, image_file_path, origin_file, dataset_file_name):
    origin_file_path = os.path.join(image_class_info_file_path, origin_file)
    df = pd.read_csv(origin_file_path, sep=' ', names=['file_name', 'class'], dtype={"file_name": str, 'class': str})

    df['class'].fillna(0, inplace=True)
    df['class'].replace('-1', 0, inplace=True)
    df['file_name'] = df['file_name'].map(lambda x : os.path.join(image_file_path, x))
    df.to_csv(dataset_file_name, sep=' ', header=False, index=False)


if __name__ == '__main__':
    object_class = "dog"
    train_file = object_class + "_train.txt"
    trainval_file = object_class + "_trainval.txt"
    test_file = object_class + "_val.txt"

    gen_dataset_file(image_class_info_file_path, image_file_path, train_file, train_file)
    gen_dataset_file(image_class_info_file_path, image_file_path, trainval_file, trainval_file)
    gen_dataset_file(image_class_info_file_path, image_file_path, test_file, test_file)
