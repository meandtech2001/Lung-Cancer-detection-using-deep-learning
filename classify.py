from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from PIL import Image


def get_model():
    cnn = load_model('cnn_model.h5')
    return cnn


def predict(image_data):
    loaded_model = get_model()
    img = img_to_array(image_data)
    np_image = transform.resize(img, (256, 256, 3))
    image4 = np.expand_dims(np_image, axis=0)
    result__ = loaded_model.predict(image4)
    def Bicubic(rgb):

        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(GrayImg)
    plt.title(" Segmentation Image " ,fontweight ="bold")
    plt.show()
    # img = img.save('static/segmentation.jpg')
    return result__

