import os

import imageio
import numpy as np
from keras.callbacks import Callback
from scipy.ndimage import zoom

from sarcopenia_ai.apps.slice_detection.utils import place_line_on_img, predict_reg, predict_slice
from sarcopenia_ai.preprocessing.preprocessing import overlay_heatmap_on_image, to256, gray2rgb


class PreviewOutput(Callback):
    def __init__(self, data, validation_steps, config):
        super().__init__()
        self.data = data
        self.config = config
        self.validation_steps = validation_steps
        self.output_dir = os.path.join(self.config.model_path, 'intermediate')
        os.makedirs(self.output_dir, exist_ok=True)

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}

        ds = self.config.ds_factor

        if self.config.mode == 'heatmap':
            for i, (image, y, name, spacing) in enumerate(zip(self.data.x_val, self.data.y_val,
                                                              self.data.names_val, self.data.spacings_val)):

                height = image.shape[0]
                if i > self.validation_steps:
                    break
                slice_thickness = spacing[2]
                pred_y, prob, pred_map, img = predict_slice(self.model, image, ds=ds)
                pred_map = np.expand_dims(zoom(np.squeeze(pred_map), ds), 2)

                img = img[:, :height, :, :]
                pred_map = pred_map[:height, :]

                img = place_line_on_img(img[0], y, pred_y, r=1)
                img = to256(img)
                #         imageio.imwrite(os.path.join(out_path,str(i)+'_test.jpg'),img)
                print(pred_map.shape)
                if pred_map.shape[1] == 1:  # case that the output is 1D
                    pred_map = np.expand_dims(np.concatenate([pred_map] * img.shape[1], axis=1), 2)
                img = overlay_heatmap_on_image(img, pred_map)
                imageio.imwrite(os.path.join(self.output_dir, str(i) + '_' + name + '_map_e' + str(epoch) + '.jpg'),
                                np.clip(img, 0, 255).astype(np.uint8))
                # img = place_line_on_img(np.hstack([X[:, :, np.newaxis], X_s[:, :, np.newaxis]]), y, m, r=1)
                # imageio.imwrite(os.path.join(out_path, str(i) + '_' + str(int(max_pred * 100)) + '_otest.jpg'), img)
        else:
            for i, (image, y, name, spacing) in enumerate(zip(self.data.x_train, self.data.y_train,
                                                              self.data.names_train, self.data.spacings_train)):
                if i > self.validation_steps:
                    break
                height = image.shape[0]
                img = image.copy()
                pred_y, prob = predict_reg(self.model, image, y)
                img = place_line_on_img(gray2rgb(img), y, pred_y, r=1)
                img = to256(img)
                imageio.imwrite(os.path.join(self.output_dir, str(i) + '_' + name + '_map_e' + str(epoch) + '.jpg'),
                                np.clip(img, 0, 255).astype(np.uint8))

        return

    def on_train_end(self, _):
        pass
