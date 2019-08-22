import os
import numpy as np
from collections import defaultdict

class BaseDataLoader:

    def __init__(self, config):
        self.config = config
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.train_idx = None
        self.val_idx = None
        self.train_generator = None
        self.validation_generator = None
        self.class_weight = None
        self.cross_val_data = defaultdict(dict)

    def get_training_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_validation_data(self):
        return self.x_val, self.y_val


    def update_crossval_data(self, fold_idx):

        self.cross_val_data[fold_idx]['train'] = self.train_idx
        self.cross_val_data[fold_idx]['val'] = self.val_idx

    def save_train_val_split(self, do_cross_val=False):
        try:
            if not do_cross_val:
                np.savez_compressed(os.path.join(self.config.model_path, 'train_val_split.npz'),
                                    train_idx=self.train_idx,
                                    val_idx=self.val_idx)
            else:
                np.savez_compressed(os.path.join(self.config.model_path, 'cross_val_split.npz'),
                                    cross_val_data=self.cross_val_data)
        except:
            print('error saving train val split.')

    def load_train_val_split(self, do_cross_val=False):
        try:
            if not do_cross_val:
                data = np.load(os.path.join(self.config.model_path, 'train_val_split.npz'))
                self.train_idx = data['train_idx']
                self.val_idx = data['val_idx']
            else:
                data = np.load(os.path.join(self.config.model_path, 'cross_val_split.npz'))
                self.cross_val_data = data['cross_val_data']
        except:
            print('error saving train val split.')
