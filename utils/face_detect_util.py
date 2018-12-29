import cv2 as cv
import numpy as np

frontal_face_cascade = cv.CascadeClassifier(r'etc/haarcascade_frontalface_alt2.xml')
profile_face_cascade = cv.CascadeClassifier(r'etc/haarcascade_profileface.xml')


def frontal_face_detect(img):
    """
    正脸检测
    :param img:
    :return: 正脸的坐标信息
    """
    frontal_face_rects = frontal_face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                                               flags=cv.CASCADE_SCALE_IMAGE)

    if len(frontal_face_rects) == 0:
        frontal_face_rects = []

    return frontal_face_rects


def profile_face_detect(img):
    """
    侧脸检测
    :param img:
    :return:侧脸的位置信息
    """
    left_profile_face_rects = profile_face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
                                                                    minSize=(30, 30),
                                                                    flags=cv.CASCADE_SCALE_IMAGE)

    flipped_img = cv.flip(img, 1)
    right_profile_face_rects = profile_face_cascade.detectMultiScale(flipped_img, scaleFactor=1.3, minNeighbors=4,
                                                                     minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    # 将左右脸的识别结果合并
    if len(left_profile_face_rects) != 0 and len(right_profile_face_rects) == 0:
        return left_profile_face_rects

    elif len(left_profile_face_rects) == 0 and len(right_profile_face_rects) != 0:
        height, width = img.shape
        # 图片翻转之后，坐标信息（x,y）相应的翻转回来
        right_profile_face_rects[:, 0:2] = [height, width] - right_profile_face_rects[:, 0:2]
        return right_profile_face_rects

    elif len(left_profile_face_rects) != 0 and len(right_profile_face_rects) != 0:
        return np.concatenate((left_profile_face_rects, right_profile_face_rects), axis=0)

    else:
        return []


def get_faces_info(img):
    """
    坐标信息作相应的转化
    :param img:
    :return: 人脸的位置信息
    """
    height, width = img.shape
    frontal_faces = frontal_face_detect(img)
    if len(frontal_faces) != 0:
        frontal_faces = frontal_faces / [width, height, width, height]
        frontal_faces = [{'face_info': face, 'is_frontal': 1} for face in frontal_faces.tolist()]

    profile_faces = profile_face_detect(img)
    if len(profile_faces) != 0:
        profile_faces = profile_faces / [width, height, width, height]
        profile_faces = [{'face_info': face, 'is_frontal': 0} for face in profile_faces.tolist()]

    frontal_faces.extend(profile_faces)

    return frontal_faces
