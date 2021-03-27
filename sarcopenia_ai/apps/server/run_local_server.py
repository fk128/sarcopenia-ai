import os
import uuid

import SimpleITK as sitk
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, flash, request, redirect, render_template
from flask import jsonify
from flask import send_from_directory
from flask_materialize import Material
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename

from sarcopenia_ai.apps.segmentation.segloader import preprocess_test_image
from sarcopenia_ai.apps.server import settings
from sarcopenia_ai.apps.slice_detection.predict import parse_inputs, to256
from sarcopenia_ai.apps.slice_detection.utils import decode_slice_detection_prediction, \
    preprocess_sitk_image_for_slice_detection, adjust_detected_position_spacing, place_line_on_img
from sarcopenia_ai.core.model_wrapper import BaseModelWrapper
from sarcopenia_ai.io import load_image
from sarcopenia_ai.preprocessing.preprocessing import blend2d
from sarcopenia_ai.utils import compute_muscle_area, compute_muscle_attenuation

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
graph = tf.get_default_graph()


def load_models():
    set_session(sess)
    model_wrapper = BaseModelWrapper(settings.SLICE_DETECTION_MODEL_PATH)
    model_wrapper.setup_model()
    global slice_detection_model
    slice_detection_model = model_wrapper.model
    slice_detection_model._make_predict_function()

    global segmentation_model
    model_wrapper = BaseModelWrapper(settings.SEGMENTATION_MODEL_PATH)
    model_wrapper.setup_model()
    segmentation_model = model_wrapper.model
    segmentation_model._make_predict_function()
    global graph
    graph = tf.get_default_graph()


def create_app(args):
    app = Flask(__name__)
    Material(app)
    app.config['args'] = args
    app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
    print('Passed item: ', app.config['args'])
    return app


args = parse_inputs()

app = create_app(args)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if request.files.get("image"):
            file = request.files["image"]
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            return jsonify(process_file(path, file.filename))


def process_file(image_path, filename, prob_threshold=0.1):
    global segmentation_model
    global slice_detection_model
    print(image_path)

    pred_id = uuid.uuid4().hex
    results = {"success": False, "prediction": {'id': pred_id}}

    try:
        sitk_image, _ = load_image(image_path)
    except:
        return 1

    image2d, image2d_preview = preprocess_sitk_image_for_slice_detection(sitk_image)
    image3d = sitk.GetArrayFromImage(sitk_image)

    spacing = sitk_image.GetSpacing()
    size = list(sitk_image.GetSize())

    with graph.as_default():
        set_session(sess)
        preds = slice_detection_model.predict(image2d)

    pred_z, prob = decode_slice_detection_prediction(preds)
    slice_z = adjust_detected_position_spacing(pred_z, spacing)

    slice_detected = prob > prob_threshold

    if slice_detected:
        results["prediction"]["slice_z"] = slice_z

        slice_image = image3d[-slice_z, :, :]

        with graph.as_default():
            set_session(sess)
            seg_image = segmentation_model.predict(preprocess_test_image(slice_image[np.newaxis, :, :, np.newaxis]))
            seg_image = seg_image[0]

        out_seg_image = np.flipud(blend2d(np.squeeze(preprocess_test_image(slice_image)),
                                          np.squeeze(seg_image > 0.5), 0.5))
        image2d_preview = place_line_on_img(image2d_preview, pred_z, pred_z, r=1)

        cv2.imwrite(f'{settings.UPLOAD_FOLDER}/{filename}_slice-{pred_id}.jpg',
                    to256(np.squeeze(preprocess_test_image(slice_image))))
        cv2.imwrite(f'{settings.UPLOAD_FOLDER}/{filename}_frontal-{pred_id}.jpg', to256(np.squeeze(image2d_preview)))
        cv2.imwrite(f'{settings.UPLOAD_FOLDER}/{filename}_seg-{pred_id}.jpg', to256(np.squeeze(out_seg_image)))

        results["prediction"]["muscle_attenuation"] = '{0:.2f} HU'.format(
            compute_muscle_attenuation(slice_image, np.squeeze(seg_image > 0.5)))
        results["prediction"]["muscle_area"] = '{0:.2f}'.format(
            compute_muscle_area(np.squeeze(seg_image > 0.5), spacing, units='cm'))
        results["prediction"]["slice_prob"] = '{0:.2f}%'.format(100 * prob)

        results["success"] = True

    if results["success"]:
        result_str = 'Slice detected at position {0} of {1} with {2:.2f}% confidence '.format(slice_z, size[2],
                                                                                              100 * prob)
    else:
        result_str = 'Slice not detected'

    results["prediction"]["str"] = result_str
    return results


def allowed_file(filename):
    for ext in settings.ALLOWED_EXTENSIONS:
        if filename.lower().endswith(ext):
            return True
    return False


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    print('upload')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(dest)
            file.save(dest)
            results = process_file(dest, filename)
            print('done')
            os.remove(dest)
            print(results)
            pred_id = results['prediction']['id']
            if results["success"]:
                return render_template('template.html',
                                       filename=f'{filename}_slice-{pred_id}.jpg',
                                       success=True,
                                       frontal_image=f'{filename}_frontal-{pred_id}.jpg',
                                       seg_image=f'{filename}_seg-{pred_id}.jpg',
                                       **results['prediction']

                                       )
            else:
                return render_template('template.html',
                                       filename=None,
                                       success=False,
                                       frontal_image=f'{filename}_frontal-{pred_id}.jpg',
                                       seg_image=None,
                                       **results['prediction'])
    return render_template('template.html', filename=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0')
