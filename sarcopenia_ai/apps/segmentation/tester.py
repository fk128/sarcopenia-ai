import glob
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import dice
from skimage.color import label2rgb

from sarcopenia_ai.apps.segmentation.segloader import preprocess_test_image, load_txt
from sarcopenia_ai.core import BaseModelWrapper
from sarcopenia_ai.core.input_parser import InputParser


def main():
    parser = InputParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--cuda_devices', type=str, default='0')
    parser.add_argument('--images_path', type=str, default='')
    parser.add_argument('--images_list', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='/tmp/model/')
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse()

    # Set up model
    model_wrapper = BaseModelWrapper(args.model_path,
                                     name=None,
                                     config=None,
                                     data_loader=None,
                                     is_multi_gpu=False
                                     )
    model_wrapper.setup_model()

    print(model_wrapper.model.summary())

    num_labels = len(model_wrapper.model.output_names)
    print('num_labels', num_labels)

    try:

        path = args.output

        os.makedirs(path, exist_ok=True)

        images_list = set(load_txt(args.images_list))
        files = glob.glob(os.path.join(args.images_path, '**/*.npz'))

        files = [f for f in files if f.split('/')[-1].replace('.npz', '') in images_list]

        df = pd.DataFrame(columns=[str(s) for s in range(1, num_labels, 1)])
        df_areas = pd.DataFrame(
            columns=['pred_' + str(s) for s in range(1, num_labels, 1)] + [str(s) for s in range(1, num_labels, 1)])
        for file in files:
            data = np.load(file)
            image = np.expand_dims(np.expand_dims(data['image'], 0), 3)
            lmap = data['labelmap']
            name = ''.join(os.path.basename(file).replace('.npz', ''))
            print(name)
            # labelmap = np.expand_dims(data['labelmap'], 0)
            image = preprocess_test_image(image)

            pred_seg = model_wrapper.model.predict(image)

            np.savez_compressed(os.path.join(path, name + '_probmap.npz'), probmap=pred_seg)

            if type(pred_seg) is not list:
                pred_seg = [pred_seg]

            labels = np.unique(lmap)
            labels = labels[labels != 0]
            if len(pred_seg) == 1:
                lmap = lmap > 0
            # else:
            #     lmap = lmap[np.newaxis,:]
            #     lmap = [lmap > 0] + [lmap == l for l in lmap]
            #     lmap =np.concatenate(lmap, axis=0)

            seg = np.zeros_like(pred_seg[0])
            seg_c = np.zeros_like(pred_seg[0])
            seg_c[pred_seg[0] > args.threshold] = 1
            for i, l in enumerate(labels):
                seg[pred_seg[i + 1] > args.threshold] = l
            pred_seg = seg

            # pred_seg = np.expand_dims(pred_seg, axis=3)

            df.loc[name, ['1','2','3']] = dice(seg[0, :, :, 0], lmap, labels)
            df.loc[name, '0'] = dice(seg_c[0, :, :, 0], lmap > 0, (0,1))[1]
            # pred_seg = blend2d(np.squeeze(image), np.squeeze(pred_seg), 0.25)
            pred_areas = []
            gt_areas = []
            print(labels)
            for l in labels:
                pred_areas.append(np.sum(pred_seg[0, :, :, 0] == l))
                gt_areas.append(np.sum(lmap == l))
            df_areas.loc[name, :] = pred_areas + gt_areas
            print(image.shape)
            print(pred_seg.shape)

            cpred_seg = label2rgb(pred_seg[0, :, :, 0], image=np.dstack([image[0, :, :, :]] * 3), bg_label=0, alpha=0.5)
            cpred_seg_all = label2rgb(seg_c[0, :, :, 0], image=np.dstack([image[0, :, :, :]] * 3), bg_label=0,
                                      alpha=0.5)
            diff_seg = label2rgb(pred_seg[0, :, :, 0] - lmap, image=np.dstack([image[0, :, :, :]] * 3), bg_label=0,
                                 alpha=0.5)
            lmap = label2rgb(lmap, image=np.dstack([image[0, :, :, :]] * 3), bg_label=0, alpha=0.5)
            # segb = blend2d(np.squeeze(image), np.squeeze(seg), 0.25)
            # img = np.concatenate(((segb * 255).astype(np.uint8), (pred_seg * 255).astype(np.uint8)))

            imageio.imwrite(os.path.join(path, name + '_pred.jpg'),
                            np.hstack([cpred_seg_all, cpred_seg, lmap, diff_seg]))

        df = df.rename(columns={'1': "Erector Spinae", '3': "Psoas", '2': " Rectus Abdominus"})
        df.to_csv(os.path.join(path, 'dice_scores.csv'))
        df_areas.to_csv(os.path.join(path, '_areas.csv'))
        sns.boxplot(data=df)
        plt.title('Test set:  {} images'.format(len(df)))
        plt.ylabel('Dice')
        plt.savefig(os.path.join(path, 'dice_score.jpg'))

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
