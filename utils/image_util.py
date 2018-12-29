from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import cv2 as cv


def prepare_image(image):
    """
    图像数据的预处理
    :param image:图像的字节数据
    :return:
    """
    image = Image.open(io.BytesIO(image))
    mode = image.mode
    image = img_to_array(image)
    image = image.astype(np.uint8)

    if mode == "RGB":
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        gray = image

    # 直方图均衡化
    cv.equalizeHist(gray, gray)
    return gray
