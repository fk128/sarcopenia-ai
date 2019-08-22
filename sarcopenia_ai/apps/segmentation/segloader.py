import glob
import os

import imageio
import numpy as np
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator
from midatasets.preprocessing import normalise_zero_one, normalise_one_one
from sklearn.model_selection import train_test_split

from sarcopenia_ai.core.data_loader import BaseDataLoader


def get_augmentation_sequence():
    sometimes = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential([
        iaa.Sometimes(0.01, iaa.OneOf([iaa.CoarseDropout((0.01, 0.15), size_percent=(0.02, 0.08)),
                                       iaa.CoarseSaltAndPepper(p=0.2, size_percent=0.01),
                                       iaa.CoarseSalt(p=0.2, size_percent=0.02)
                                       ])),
        iaa.Sometimes(0.2,
                      iaa.LinearContrast((0.25, 0.8))),
        iaa.Sometimes(0.2,
                      iaa.Add((-20, 20))),
    ])
    seq2 = iaa.Sequential([

        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01), cval=0)),
        iaa.Sometimes(0.2, iaa.ElasticTransformation(alpha=(15, 25), sigma=6, cval=0))

    ])

    return seq, seq2


def preprocess_test_image(image):
    image = normalise_one_one(image, -250, 250)
    return image


def load_txt(filepath):
    with open(filepath, 'r') as f:
        return f.read().splitlines()


class TrainerData(BaseDataLoader):

    def __init__(self, config):
        super(TrainerData, self).__init__(config)

    def get_num(self):
        path = self.config.train_datapath
        if type(path) is list:
            files = []
            for p in path:
                files += glob.glob(os.path.join(p, '*.npz'))
        else:
            files = glob.glob(os.path.join(path, '*.npz'))
        return len(files)

    def split_data(self, train_idx=None, val_idx=None, train_filenames=None, val_filenames=None):

        if train_filenames:
            train_filenames = load_txt(train_filenames)
        if val_filenames:
            val_filenames = load_txt(val_filenames)

        images, labelmaps = self.load_and_preprocess(self.config.train_datapath, train_filenames)

        if train_idx is not None:

            self.x_train = images[train_idx]
            self.y_train = labelmaps[train_idx]
            self.x_val = images[val_idx]
            self.y_val = labelmaps[val_idx]

        else:
            print('random split')
            self.x_train, self.x_val, self.y_train, self.y_val = \
                train_test_split(images, labelmaps, test_size=0.01, random_state=42)

        self.x_val, self.y_val = self.load_and_preprocess(self.config.val_datapath, val_filenames)
        self.validation_steps = len(self.x_val) // self.config.batch_size
        self.steps_per_epoch = len(self.x_train) // self.config.batch_size
        print(self.steps_per_epoch, self.validation_steps)

        self.train_generator = self.create_generator(self.x_train, self.y_train, self.config.batch_size,
                                                     do_augment=self.config.do_augment)
        self.validation_generator = self.create_generator(self.x_val, self.y_val, self.config.batch_size,
                                                          do_augment=False)

    @staticmethod
    def one_hot_encode_labelmap(lmap, num_classes=None, is_list=True):

        if num_classes is None:
            classes = np.unique(lmap)
            num_classes = len(classes)
        else:
            classes = list(range(num_classes))

        if is_list:
            return [(lmap == l).astype(np.uint8) for l in classes]
        else:
            out = np.zeros(lmap.shape + (num_classes,))
            for i, l in enumerate(classes):
                out[:, :, i] = (lmap == l).astype(np.uint8)
            return out

    def load_and_preprocess(self, path, filenames=None):

        files = []
        for p in path:
            files += glob.glob(os.path.join(p, '*.npz'))

        images, labelmaps = [], []

        if filenames:
            filenames = set(filenames)

        for file in files:

            name = file.split('/')[-1].replace('.npz', '')

            if filenames and name not in filenames:
                continue
            data = np.load(file)
            image = np.expand_dims(data['image'], 0)
            labelmap = np.expand_dims(data['labelmap'], 0)
            images.append(image)
            labelmaps.append(labelmap)
        images = np.concatenate(images, axis=0)
        labelmaps = np.concatenate(labelmaps, axis=0)

        images = np.expand_dims(images, 3)
        labelmaps = np.expand_dims(labelmaps, 3)

        images = normalise_one_one(images, -250, 250)

        return images, labelmaps

    def create_generator(self, x_train, y_train, batch_size, do_augment=False, save=False):
        SEED = 42

        if save:
            output_path = os.path.join(self.config.model_path, 'input_generator_output')
            os.makedirs(output_path, exist_ok=True)

        itr = 0
        if do_augment:
            gen_args = dict(width_shift_range=0.05,
                            height_shift_range=0.05,
                            # rotation_range=10,
                            horizontal_flip=True,
                            vertical_flip=True,
                            # shear_range=0.05,
                            zoom_range=0.07,
                            fill_mode='constant',
                            cval=0)
            data_generator = ImageDataGenerator(**gen_args).flow(x_train, x_train, batch_size, seed=SEED)
            mask_generator = ImageDataGenerator(**gen_args).flow(y_train, y_train, batch_size, seed=SEED)
        else:

            data_generator = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
            ).flow(x_train, x_train, batch_size, seed=SEED)
            mask_generator = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
            ).flow(y_train, y_train, batch_size, seed=SEED)

        while True:
            x_batch, _ = data_generator.next()
            y_batch, _ = mask_generator.next()

            if save:
                itr += 1
                for i, (x, y) in enumerate(zip(x_batch, y_batch)):
                    out = (255 * np.hstack([normalise_zero_one(x), normalise_zero_one(y)])).astype(np.uint8)
                    imageio.imwrite(os.path.join(output_path, str(itr) + '_' + str(i) + '_out.jpg'), out)

            y_batch = [(y_batch == l).astype(np.uint8) for l in range(self.config.num_labels)]
            y_batch[0] = (y_batch[1] + y_batch[2] + y_batch[3]) > 0
            yield x_batch, y_batch
