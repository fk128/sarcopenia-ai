import os
from argparse import ArgumentParser

import flask
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from sarcopenia_ai.apps.slice_detection.utils import preprocess_sitk_image_for_slice_detection, \
    decode_slice_detection_prediction, adjust_detected_position_spacing
from sarcopenia_ai.core.model_wrapper import BaseModelWrapper
from sarcopenia_ai.io import load_image

sess = tf.Session()
graph = tf.get_default_graph()

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/data'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def parse_inputs():
    parser = ArgumentParser()

    parser.add_argument('--cuda_devices', type=str, default='')
    parser.add_argument('--model_path', type=str, default='/tmp/slice_detection_1/')
    parser.add_argument('--sigma', type=float, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--image_path', type=str, default=None)

    args = parser.parse_args()

    return args


def load_model(model_path):
    set_session(sess)
    model_wrapper = BaseModelWrapper(model_path)
    model_wrapper.setup_model()
    global model
    model = model_wrapper.model
    model._make_predict_function()
    global graph
    graph = tf.get_default_graph()


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    global graph
    global sess

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            file = flask.request.files["image"]
            print(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)

            sitk_image, _ = load_image(path)

            image = preprocess_sitk_image_for_slice_detection(sitk_image)

            spacing = sitk_image.GetSpacing()
            print(image.shape)

            with graph.as_default():
                set_session(sess)
                preds = model.predict(image)

            pred_z, prob = decode_slice_detection_prediction(preds)
            slice_z = adjust_detected_position_spacing(pred_z, spacing)

            print('Slice detected at position {} with confidence {}'.format(slice_z, prob))

            data["predictions"] = [{'filename': file.filename, 'slice_z': slice_z, 'probability': prob}]

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


def main():
    args = parse_inputs()
    print(args)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Setup model
    load_model(args.model_path)

    app.run(host='0.0.0.0')

    # predict_and_evaluate(model_wrapper, args)


if __name__ == '__main__':
    main()
