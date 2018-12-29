from django.shortcuts import render, HttpResponse
from keras.preprocessing.image import img_to_array
from utils import face_detect_util, image_util
import json
import logging


# Create your views here.

def index(request):
    return render(request, 'index.html')


def face_detect(request):
    resp = {"success": False}
    if request.method == "POST":
        if request.FILES['img']:
            try:
                img = request.FILES['img'].read()
                img = image_util.prepare_image(img)

                faces_info = face_detect_util.get_faces_info(img)

                resp['success'] = True
                resp['faces_info'] = faces_info
            except Exception as e:
                resp['error_msg'] = str(e)

    return HttpResponse(json.dumps(resp), content_type="application/json")
