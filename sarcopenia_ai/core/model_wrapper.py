import glob
import logging
import math
import os
import pickle
import re
import subprocess

import keras
import tensorflow as tf
from keras import backend as K
from keras.backend import set_session
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.models import model_from_json, Model
from keras.optimizers import Adam
from keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.client import device_lib

from sarcopenia_ai.core.layers import GroupNormalization
from .callbacks import LRFinder

logger = logging.getLogger(__name__)


def get_number_of_gpus_from_system():
    n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    return n


def allocate_tf_gpu_devices(cuda_devices):
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name.split(':')[-1] for x in local_device_protos if x.device_type == 'GPU']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if cuda_devices != '':
        config.gpu_options.visible_device_list = cuda_devices
        set_session(tf.Session(config=config))
    else:
        set_session(tf.Session(config=config))
        cuda_devices = ','.join(get_available_gpus())

    num_gpus = len(cuda_devices.split(','))
    return num_gpus, cuda_devices


class ModelMultiGPU(Model):
    def __init__(self, model, num_gpus):
        parallel_model = multi_gpu_model(model, num_gpus)
        self.__dict__.update(parallel_model.__dict__)
        self._model = model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._model, attrname)

        return super(ModelMultiGPU, self).__getattribute__(attrname)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 5.0

    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print('learning rate ', lrate)
    return lrate


class BaseModelWrapper:
    def __init__(self, model_dir, name=None, data_loader=None, config=None, is_multi_gpu=False):

        self.model_dir = model_dir
        self.config = config
        self.model = None
        self.parallel_model = None
        self.name = self._get_model_name(name)
        self.start_epoch = 0
        self.is_multi_gpu = is_multi_gpu
        self.num_gpus = 1
        self.class_mode = 'binary'
        self.data_loader = data_loader
        self.custom_objects = {'GroupNormalization': GroupNormalization}
        self.callbacks = []
        self.model_input_shape = (None, None, 1)
        self.max_queue_size = 10
        self.workers = 1
        self.depth_multiplier = 0.5
        self.epochs = 50
        self.labels = []
        self.steps_per_epoch = None
        self.learning_rate = 0.05
        if config is not None:
            self.__dict__.update({k: v for k, v in vars(config).items() if k in self.__dict__.keys()})
        self.compile_args = {
            'loss': self.class_mode + '_crossentropy',
            'optimizer': Adam(lr=self.learning_rate),
            'metrics': ['accuracy']
        }

        # for compatibility
        self.input_shape = self.model_input_shape

        if self.workers > 1:
            self.use_multiprocessing = True
        else:
            self.use_multiprocessing = False

        print(self.__dict__)

    def _get_model_name(self, name):
        if name is None:
            # get name from architecture file if exists
            name = [f for f in os.listdir(self.model_dir) if f.endswith('.json')]

            if name:
                name = name[0].replace('.json', '')
                return name
            else:
                return 'model'
        else:
            return name

    def save_architecture(self):
        model_path = os.path.join(self.model_dir, self.name + '.json')

        if self.custom_objects != {}:
            custom_objects_path =  os.path.join(self.model_dir, self.name +'.custom_objects')
            with open(custom_objects_path, 'wb') as file:
                pickle.dump(self.custom_objects, file)

        if not os.path.exists(model_path):
            logger.info('Saving model architecture to file.')

            json_string = self.model.to_json()
            with open(model_path, 'w') as outfile:
                outfile.write(json_string)
            logger.info('Model architecture saved to json.')
        else:
            logger.info('Model architecture file exists: not overwriting.')

    def load_architecture(self):
        logger.info('Attempting to load model architecture from file.')

        custom_objects_path = os.path.join(self.model_dir, self.name + '.custom_objects')
        if os.path.exists(custom_objects_path):
            with open(custom_objects_path, 'rb') as file:
                self.custom_objects.update(pickle.load(file))

        model_path = os.path.join(self.model_dir, self.name + '.json')
        if os.path.exists(model_path):
            with open(model_path, 'r') as json_file:
                json_string = json_file.read()
                self.model = model_from_json(json_string, custom_objects=self.custom_objects)
                logger.info('Model architecture loaded from file.')
        else:
            logger.info('Model architecture file does not exist.')

    def save(self):
        if self.model is None:
            raise Exception("Model does not exist.")

        logger.info("Saving weights...")
        model_path = os.path.join(self.model_dir, self.name + '.h5')
        self.model.save_weights(model_path)
        logger.info("Weights saved.")

    def load_weights(self):
        if self.model is None:
            raise Exception("Model not defined.")
        model_path = os.path.join(self.model_dir, self.name + '.h5')
        checkpoint_paths = glob.glob(os.path.join(self.model_dir, self.name + "*-checkpoint.h5"))
        model_list = []
        if os.path.exists(model_path):
            model_list.append((os.path.getmtime(model_path), model_path))

        if len(checkpoint_paths) > 0:
            checkpoint_list = []
            for cmodel_path in checkpoint_paths:
                checkpoint_list.append((os.path.getmtime(cmodel_path), cmodel_path))
            checkpoint_list.sort(reverse=True)
            model_list.extend(checkpoint_list)

            # get last epoch number
            match = re.findall('at_epoch_(\d+)', checkpoint_list[0][1])
            if match:
                self.start_epoch = int(match[0])

        model_list.sort(reverse=True)

        if model_list != []:

            success = False
            for i in range(len(model_list)):
                model_path = model_list[i][1]
                logger.info("Loading model weights  {} ...\n".format(model_path))
                try:
                    # self.model = load_model(model_path, custom_objects=self.custom_objects)
                    if self.is_multi_gpu:
                        with tf.device('/cpu:0'):
                            self.model.load_weights(model_path)
                    else:
                        self.model.load_weights(model_path)
                    success = True
                except Exception as e:
                    logger.warning('Failed to load model weights at {}'.format(model_path))
                    logger.warning(str(e))

                if success:
                    logger.info("Model weights loaded.")
                    break

            if not success:
                logger.warning("Failed to load any model weights.")

        else:
            logger.info("No saved model weights found.")

    def setup_model(self):
        self.build_multi_gpu_model()

    def build_model(self):
        self.load_architecture()
        if self.model is None:
            logger.info('Building model from definition.')
            self.model = self.define_model()
            self.save_architecture()

        self.load_weights()

    def build_multi_gpu_model(self):
        if self.is_multi_gpu:
            logger.info('Building multi-gpu model')
            with tf.device('/cpu:0'):
                self.build_model()
            self.parallel_model = ModelMultiGPU(self.model, self.num_gpus)
        else:
            self.build_model()

    def compile(self, compile_args={}):
        self.compile_args.update(compile_args)
        self.model.compile(**self.compile_args)

        if self.is_multi_gpu:
            self.parallel_model.compile(**self.compile_args)

    def get_model(self):
        if self.is_multi_gpu:
            return self.parallel_model
        else:
            return self.model

    def freeze_model(self, layer_idx=-1):
        self.frozen_model = Model(self.model.input, self.model.output, name=self.model.name + '_frozen')
        for layer in self.frozen_model.layers:
            layer.trainable = True
        for layer in self.frozen_model.layers[:layer_idx]:
            layer.trainable = False
        self.frozen_model.compile(**self.compile_args)
        return self.frozen_model

    def define_model(self):
        raise NotImplementedError

    def get_callbacks(self):
        return self.get_default_callbacks() + self.callbacks

    def get_default_callbacks(self):

        callbacks_list = []

        # checkpoint
        checkpoint = ModelCheckpoint(os.path.join(self.model_dir,
                                                  self.name + "_at_epoch_{epoch:02d}-checkpoint.h5"),
                                     save_weights_only=True)
        callbacks_list.append(checkpoint)

        # learning rate scheduler
        lrate_callback = LearningRateScheduler(step_decay)
        callbacks_list.append(lrate_callback)

        # logger
        log_path = os.path.join(self.model_dir, self.name + "_log.csv")
        csvlogger = CSVLogger(log_path, separator=',', append=True)
        callbacks_list.append(csvlogger)

        return callbacks_list

    def train_generator(self, train_model=None, epochs=None):

        try:
            if train_model:
                pass
            elif self.parallel_model:
                train_model = self.parallel_model
            else:
                train_model = self.model

            if epochs:
                pass
            else:
                epochs = self.epochs

            train_model.fit_generator(self.data_loader.train_generator,
                                      epochs=epochs,
                                      steps_per_epoch=self.data_loader.steps_per_epoch,
                                      validation_data=self.data_loader.validation_generator,
                                      validation_steps=self.data_loader.validation_steps,
                                      callbacks=self.get_callbacks(),
                                      initial_epoch=self.start_epoch,
                                      max_queue_size=self.max_queue_size,
                                      workers=self.workers,
                                      use_multiprocessing=self.use_multiprocessing,
                                      class_weight=self.data_loader.class_weight
                                      )

        except KeyboardInterrupt:
            pass

    def train_frozen(self):

        frozen_model = self.freeze_model(layer_idx=-2)
        print(frozen_model.summary())
        self.train_generator(train_model=frozen_model, epochs=10)
        idx = 0
        for i, layer in enumerate(self.model.layers):
            if 'decoder' in layer.name:
                idx = i
                break
        frozen_model = self.freeze_model(layer_idx=idx)
        print(frozen_model.summary())
        self.train_generator(train_model=frozen_model, epochs=10)

        self.train_generator()


    def find_lr(self, max_iterations=5000, base_lr=10e-8, max_lr=10, alpha=0.98):

        lr_finder_callback = LRFinder(max_iterations=max_iterations,
                                      base_lr=base_lr,
                                      max_lr=max_lr,
                                      alpha=alpha,
                                      log_path=self.model_dir)

        if self.parallel_model:
            train_model = self.parallel_model
        else:
            train_model = self.model

        train_model.fit_generator(self.data_loader.train_generator,
                                  callbacks=[lr_finder_callback],
                                  epochs=self.epochs,
                                  steps_per_epoch=self.steps_per_epoch,
                                  max_queue_size=self.max_queue_size,
                                  workers=self.workers,
                                  use_multiprocessing=self.use_multiprocessing,
                                  class_weight=self.data_loader.class_weight
                                  )

