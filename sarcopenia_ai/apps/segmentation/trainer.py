import os

from .segloader import TrainerData
from sarcopenia_ai.core.input_parser import InputParser
from sarcopenia_ai.core.model_wrapper import allocate_tf_gpu_devices
from .models import get_available_models
from distutils.util import strtobool


def create_parser():
    parser = InputParser()

    parser.add_argument('--restart', type=strtobool, default=False,
                        help='restart training by deleting all associated files')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='number of distinct images to samples from in a given batch')
    parser.add_argument('--img_batch_size', type=int, default=3,
                        help='number of samples obtained from a single image in a given batch')
    parser.add_argument('--input_shape', type=int, default=None, nargs=3,
                        help='image input shape')
    parser.add_argument('--model_input_shape', type=int, default=[None, None, 1], nargs=3,
                        help='model input shape')
    parser.add_argument('--model_architecture', type=str, default='UNet2D', choices=get_available_models().keys(),
                        help='name of model')
    parser.add_argument('--model_name', type=str, default='UNet', help='name of model')
    parser.add_argument('--dataset_path', type=str, default=None, help='location of dataset .npz')
    parser.add_argument('--train_filenames', type=str, default=None, help='')
    parser.add_argument('--val_filenames', type=str, default=None, help='')
    parser.add_argument('--random_state', type=int, default=42, help='random seed')
    parser.add_argument('--input_spacing', type=int, default=1, help='spacing of input image')
    parser.add_argument('--num_val', type=int, default=20,
                        help='number of validation samples during training')
    parser.add_argument('--use_cache', type=bool, default=True, help='cache input image pre-processing')
    parser.add_argument('--cache_path', type=str, default=None,

                        help='path to store the pre-processed images. If None, then model_path is used')
    parser.add_argument('--cuda_devices', type=str, default='0,1')
    parser.add_argument('--model_path', type=str, default='/tmp/segmentation/')
    parser.add_argument('--sigma', type=float, default=3)
    parser.add_argument('--sampling_rate', type=float, default=0.5,
                        help='rate to sample from crops that contain the slice')
    parser.add_argument('--do_augment', type=strtobool, default=True, help='enable augmentation')
    parser.add_argument('--preview_generator_output', type=strtobool, default=False,
                        help='preview generator output')
    parser.add_argument('--preview_training_output', type=strtobool, default=False,
                        help='preview intermediate training output')
    parser.add_argument('--do_checkpoint', type=strtobool, default=False,
                        help='enable model checkpoint saving')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--steps_per_epoch', type=int, default=None)
    parser.add_argument('--train_datapath', type=str, default='', nargs='+')
    parser.add_argument('--val_datapath', type=str, default='', nargs='+')
    parser.add_argument('--num_labels', type=int, default=1)

    return parser


def run(args):
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

    trainer_data = TrainerData(args)

    trainer_data.split_data(train_filenames=args.train_filenames, val_filenames=args.val_filenames)

    Model = get_available_models()[args.model_architecture]
    args.model_name = args.model_architecture

    # Setup model
    model_wrapper = Model(model_dir=args.model_path,
                           name=args.model_name,
                           config=args,
                           data_loader=trainer_data)

    model_wrapper.compile()

    print(model_wrapper.model.summary())

    try:
        model_wrapper.train_generator()

    except KeyboardInterrupt:
        pass

    model_wrapper.save()


def main():
    parser = create_parser()
    args = parser.parse()
    run(args)


if __name__ == '__main__':
    main()
