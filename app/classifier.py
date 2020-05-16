import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

IMAGE_RES = 224

def read_img_from_file(file_object):
    image = tf.image.decode_image(file_object.read())
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return tf.expand_dims(image, 0)

class Classifier():
    def __init__(self, model_path):
        self.model_path = model_path
        self.mod = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    def predict_from_file(self, image_file):
        image = read_img_from_file(image_file)
        pred =  self.mod.predict(image, batch_size=1)[0]
        ix = np.argmax(pred)
        label = 'cat' if (ix == 0) else 'dog'
        conf = (float)(pred[ix])
        return {"label": label, "confidence": conf}
        