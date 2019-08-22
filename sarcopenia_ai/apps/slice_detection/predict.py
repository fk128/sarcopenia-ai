import SimpleITK as sitk
import cv2

from sarcopenia_ai.apps.slice_detection.utils import place_line_on_img, decode_slice_detection_prediction, \
    preprocess_sitk_image_for_slice_detection, adjust_detected_position_spacing
from sarcopenia_ai.core.input_parser import InputParser
from sarcopenia_ai.core.model_wrapper import BaseModelWrapper, allocate_tf_gpu_devices
from sarcopenia_ai.io import load_image
from sarcopenia_ai.preprocessing.preprocessing import to256


def parse_inputs():
    parser = InputParser()

    parser.add_argument('--batch_size', type=int, default=3
                        , help='number of distinct images to samples from in a given batch')
    parser.add_argument('--ds_factor', type=int, default=2, help='output downsampling factor')
    parser.add_argument('--input_spacing', type=int, default=1, help='spacing of input image')
    parser.add_argument('--cuda_devices', type=str, default='0,1')
    parser.add_argument('--model_path', type=str, default='/data/slice_detection/')
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('--image', type=str, default=[], nargs='+')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prob_threshold', type=float, default=0.1)

    args = parser.parse()

    return args


def load_model(model_path):
    # Setup model
    model_wrapper = BaseModelWrapper(model_path,
                                     name=None,
                                     config=None,
                                     data_loader=None,
                                     is_multi_gpu=False
                                     )
    model_wrapper.setup_model()

    print(model_wrapper.model.summary())

    return model_wrapper.model


def main():
    args = parse_inputs()

    # GPU allocation options
    args.num_gpus, args.cuda_devices = allocate_tf_gpu_devices(args.cuda_devices)
    args.is_multi_gpu = args.num_gpus > 1

    print(args)

    # Setup model
    model_wrapper = BaseModelWrapper(args.model_path)
    model_wrapper.setup_model()

    for image_path in args.image:
        print(image_path)
        sitk_image, image_name = load_image(image_path)

        image2d = preprocess_sitk_image_for_slice_detection(sitk_image)

        spacing = sitk_image.GetSpacing()
        print('direction', sitk_image.GetDirection())
        print('spacing', spacing)
        print('image shape', image2d.shape)

        preds = model_wrapper.model.predict(image2d)

        pred_z, prob = decode_slice_detection_prediction(preds)
        slice_z = adjust_detected_position_spacing(pred_z, spacing)

        if prob > args.prob_threshold:
            print('Slice detected at position {} with confidence {}'.format(slice_z, prob))
            image = sitk.GetArrayFromImage(sitk_image)
            print(image.shape)
            slice_image = image[slice_z, :, :]
            image2d = place_line_on_img(image2d[0], pred_z, pred_z, r=1)

            cv2.imwrite(f'/{args.output_path}/{image_name}_slice.jpg', to256(slice_image))

            cv2.imwrite(f'/{args.output_path}/{image_name}_frontal.jpg', to256(image2d))
        else:
            print('slice not detected')


if __name__ == '__main__':
    main()
