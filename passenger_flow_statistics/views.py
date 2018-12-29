from django.shortcuts import HttpResponse
from PIL import Image
import io
import cv2
import numpy as np

from passenger_flow_statistics.yolov3.yolo import YOLO

from passenger_flow_statistics.tools import generate_detections as gdet
from passenger_flow_statistics.deep_sort.detection import Detection
from passenger_flow_statistics.deep_sort import preprocessing
import passenger_flow_statistics.tools.distance_util as util
import redis
from passenger_flow_statistics.box import Box, handle
import uuid
import tensorflow as tf
from ai import settings
import json

# Create your views here.


max_cosine_distance = settings.MAX_COSINE_DISTANCE
nms_max_overlap = settings.NMS_MAX_OVERLAP
yolo = YOLO()

deep_sort_model_filename = settings.DEEP_SORT_MODEL_FILENAME
encoder = gdet.create_box_encoder(deep_sort_model_filename, batch_size=1)

pool = redis.ConnectionPool(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
r = redis.StrictRedis(connection_pool=pool)

graph = tf.get_default_graph()


def flow_statistics(request):
    resp = {"success": False}
    result = {}
    if request.method == "POST":
        if request.FILES['img']:
            img = request.FILES['img'].read()
            camera_id = request.POST['camera_id']
            image = Image.open(io.BytesIO(img))
            img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            image = Image.fromarray(img)

            with graph.as_default():
                boxes = yolo.detect_image(image)

                features = encoder(img, boxes)

            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # 获取基准库
            if r.get(camera_id) is None:
                base_boxes = [Box(str(uuid.uuid4()), det.feature.tolist(), [det.tlwh.tolist()]) for det in
                              detections]

                # 返回人数
                result['left'] = len([box for box in base_boxes if box.is_left()])
                result['right'] = len([box for box in base_boxes if box.is_right()])
                result['Total'] = len(base_boxes)

                resp['result'] = result
                resp['success'] = True

                r.set(camera_id, json.dumps(base_boxes, default=lambda o: o.__dict__))
                r.set(str(camera_id) + "_total", len(base_boxes))

            else:
                b = json.loads(r.get(camera_id), object_hook=handle)
                bases = np.array([box.feature for box in b])
                targets = np.array([box.feature for box in detections])

                matrix = util.similarity(bases, targets)

                mask = matrix < max_cosine_distance
                # 判断更新
                update_row = np.argmin(matrix, axis=0)
                update_boxes = {}
                for c_idx, r_idx in enumerate(update_row):
                    if matrix[r_idx, c_idx] < max_cosine_distance:
                        if update_boxes.get(r_idx):
                            if matrix[r_idx, update_boxes.get(r_idx)] > matrix[r_idx, c_idx]:
                                update_boxes[r_idx] = c_idx
                        else:
                            update_boxes[r_idx] = c_idx

                for r_idx, c_idx in update_boxes.items():
                    base_box = b[r_idx]
                    detection = detections[c_idx]
                    # 更新特征向量 更新坐标
                    base_box.update(detection.feature.tolist(), detection.tlwh.tolist())

                # 判断是否删除（一行都>0.3删除）
                retain_index = np.sum(mask, axis=1) != 0
                retain_box = np.array(b)[retain_index].tolist()

                # 判断是否新加入 （一列都>0.3）
                join_index = np.sum(mask, axis=0) == 0
                join_box_feature = np.array(detections)[join_index].tolist()
                join_box = [Box(str(uuid.uuid4()), box.feature.tolist(), [box.tlwh.tolist()]) for box in
                            join_box_feature]

                retain_box.extend(join_box)
                base_boxes = retain_box

                total = int(r.get(str(camera_id) + "_total"))

                result['left'] = len([box for box in base_boxes if box.is_left()])
                result['right'] = len([box for box in base_boxes if box.is_right()])
                result['Total'] = total + len(join_box)

                resp['result'] = result
                resp['success'] = True

                r.set(str(camera_id) + "_total", total + len(join_box))
                r.set(camera_id, json.dumps(base_boxes, default=lambda o: o.__dict__))

    return HttpResponse(json.dumps(resp), content_type="application/json")
