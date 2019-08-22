import os
from distutils.util import strtobool

import keras.backend as K
from sklearn.model_selection import KFold

from sarcopenia_ai.apps.slice_detection.callbacks import PreviewOutput
from sarcopenia_ai.apps.slice_detection.dataloader import TrainerData
from sarcopenia_ai.core.input_parser import InputParser
from sarcopenia_ai.core.model_wrapper import allocate_tf_gpu_devices
from .models import get_model


def parse_inputs():
    parser = InputParser()

    parser.add_argument('--restart', type=strtobool, default=False,
                        help='restart training by deleting all associated files')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='number of distinct images to samples from in a given batch')
    parser.add_argument('--img_batch_size', type=int, default=3,
                        help='number of samples obtained from a single image in a given batch')
    parser.add_argument('--input_shape', type=int, default=None, nargs=3,
                        help='image input shape')
    parser.add_argument('--model_input_shape', type=int, default=[None, None, 1], nargs=3,
                        help='model input shape')
    parser.add_argument('--model_name', type=str, default='UNet', help='name of model')
    parser.add_argument('--dataset_path', type=str, default=None, help='location of dataset .npz')
    parser.add_argument('--n_splits', type=int, default=3,
                        help='number of splits for cross validation')
    parser.add_argument('--random_state', type=int, default=42, help='random seed')
    parser.add_argument('--ds_factor', type=int, default=2, help='output downsampling factor')
    parser.add_argument('--input_spacing', type=int, default=1, help='spacing of input image')
    parser.add_argument('--num_val', type=int, default=20,
                        help='number of validation samples during training')
    parser.add_argument('--do_crossval', type=strtobool, default=False, help='do cross validation')
    parser.add_argument('--flatten_output', type=strtobool, default=False,
                        help='1D output if true; otherwise, the output is 2D')
    parser.add_argument('--use_cache', type=strtobool, default=True,
                        help='cache input image pre-processing')
    parser.add_argument('--cache_path', type=str, default=None,

                        help='path to store the pre-processed images. If None, then model_path is used')
    parser.add_argument('--mode', type=str, default='heatmap',
                        help='labelmap as heatmap or regression', choices=['heatmap', 'reg'])
    parser.add_argument('--image_type', type=str, default='frontal', choices=['frontal', 'sagittal'])
    parser.add_argument('--cuda_devices', type=str, default='')
    parser.add_argument('--model_path', type=str, default='/tmp/slice_detection_1/')
    parser.add_argument('--sigma', type=float, default=3)
    parser.add_argument('--sampling_rate', type=float, default=0.5,
                        help='rate to sample from crops that contain the slice')
    parser.add_argument('--do_augment', type=strtobool, default=True, help='enable augmentation')
    parser.add_argument('--preview_generator_output', type=strtobool, default=False,
                        help='preview generator output')
    parser.add_argument('--preview_training_output', type=strtobool, default=False,
                        help='preview intermediate training output')
    parser.add_argument('--preview_validation_steps', type=int, default=2)
    parser.add_argument('--regression_dual_output', type=strtobool, default=False,
                        help='enable dual output for regression')
    parser.add_argument('--do_checkpoint', type=strtobool, default=False,
                        help='enable model checkpoint saving')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    args = parser.parse()

    return args


def distance(y_true, y_pred):
    x_true = K.flatten(K.argmax(y_true, axis=1))
    valid = K.cast(K.sum(y_true, axis=(1, 2)) > 0.5, 'float32')

    x_pred = K.flatten(K.argmax(y_pred, axis=1))
    d = K.cast(x_true - x_pred, 'float32')
    return valid * d * d


def cross_validate(base_model, args):
    trainer_data = TrainerData(args)
    kf = KFold(n_splits=args.n_splits, random_state=args.random_state, shuffle=True)

    for idx, (train_index, val_index) in enumerate(kf.split(list(range(trainer_data.num_samples)))):
        print('cross validation step {} of {}'.format(idx + 1, args.n_splits))

        trainer_data.split_data(train_index, val_index)
        trainer_data.update_crossval_data(idx)
        trainer_data.save_train_val_split(True)

        if args.preview_generator_output:
            trainer_data.preview_generator_output()

        # Setup model
        model_name = args.model_name + '_cv_' + str(idx + 1) + '_of_' + str(args.n_splits)
        model_wrapper = base_model(model_dir=args.model_path,
                                   name=model_name,
                                   config=args,
                                   data_loader=trainer_data)

        if args.preview_training_output:
            model_wrapper.callbacks.append(PreviewOutput(trainer_data, args.preview_validation_steps, args))

        print(model_wrapper.model.summary())

        try:
            model_wrapper.train_generator()

        except KeyboardInterrupt:
            pass

        model_wrapper.save()


def main():
    args = parse_inputs()

    print(args)

    args.num_gpus, args.cuda_devices = allocate_tf_gpu_devices(args.cuda_devices)
    args.is_multi_gpu = args.num_gpus > 1

    # Handle restarting and resuming training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    base_model = get_model(args.model_name)

    if args.do_crossval:
        cross_validate(base_model, args)
    else:
        trainer_data = TrainerData(args)
        trainer_data.split_data()

        if args.preview_generator_output:
            trainer_data.preview_generator_output()

        # Setup model
        model_wrapper = base_model(model_dir=args.model_path,
                                   name=args.model_name,
                                   config=args,
                                   data_loader=trainer_data)

        model_wrapper.compile({'metrics': ['accuracy', distance]})

        if args.preview_training_output:
            model_wrapper.callbacks.append(
                PreviewOutput(trainer_data, validation_steps=args.preview_validation_steps, config=args))

        print(model_wrapper.model.summary())

        try:
            model_wrapper.train_generator()

        except KeyboardInterrupt:
            pass

        model_wrapper.save()


if __name__ == '__main__':
    main()
