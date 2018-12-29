from __future__ import division, print_function, absolute_import

import warnings
import cv2
import numpy as np
from PIL import Image
from passenger_flow_statistics.yolov3.yolo import YOLO

from passenger_flow_statistics.tools import generate_detections as gdet
from passenger_flow_statistics.deep_sort.detection import Detection
from passenger_flow_statistics.deep_sort import preprocessing
import passenger_flow_statistics.tools.distance_util as util
import redis
from passenger_flow_statistics.box import Box, State, handle
import uuid
import json


warnings.filterwarnings('ignore')


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    # nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    img = cv2.imread('img/3.jpeg')
    # cv2.imshow('img', img)
    image = Image.fromarray(img)
    boxes = yolo.detect_image(image)
    # print(boxes)
    # print("box_num",len(boxs))
    features = encoder(img, boxes)

    # score to 1.0 here).
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # for det in detections:
    #     bbox = det.to_tlbr()
    #     print(bbox)
    #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    # 获取基准库
    pool = redis.ConnectionPool(host='127.0.0.1', port=6379)
    r = redis.StrictRedis(connection_pool=pool)

    camera_id = "1111"
    if r.get(camera_id) is None:
        base_boxes = [Box(str(uuid.uuid4()), det.feature.tolist(), [det.tlwh.tolist()]) for det in
                      detections]

        r.set(camera_id, json.dumps(base_boxes, default=lambda o: o.__dict__))

        # 返回人数
        print("left:", len([box for box in base_boxes if box.is_left()]))
        print("right:", len([box for box in base_boxes if box.is_right()]))
        print("Total:", len(base_boxes))

    else:
        b = json.loads(r.get(camera_id), object_hook=handle)
        bases = np.array([box.feature for box in b])
        targets = np.array([box.feature for box in detections])

        matrix = util.similarity(bases, targets)
        # print(matrix)

        mask = matrix < max_cosine_distance
        # 判断更新
        # ss = np.min(matrix, axis=0) < 0.3
        # print(type(ss))
        update_row = np.argmin(matrix, axis=0)
        update_boxes = {}
        for c_idx, r_idx in enumerate(update_row):
            if matrix[r_idx, c_idx] < max_cosine_distance:
                if update_boxes.get(r_idx):
                    if matrix[r_idx, update_boxes.get(r_idx)] > matrix[r_idx, c_idx]:
                        update_boxes[r_idx] = c_idx
                else:
                    update_boxes[r_idx] = c_idx
        # print(update_boxes)

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

        print("left:", len([box for box in base_boxes if box.is_left()]))
        print("right:", len([box for box in base_boxes if box.is_right()]))
        print("Total:", len(base_boxes))

        r.set(camera_id, json.dumps(base_boxes, default=lambda o: o.__dict__))

        # for det in retain_box:
        #     det = Detection(np.array(det.bboxes[-1]), 1, np.array(det.feature))
        #     bbox = det.to_tlbr()
        #     # bbox = np.array(det.bboxes[-1])
        #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 255), 2)
        #
        # for det in join_box:
        #     det = Detection(np.array(det.bboxes[-1]), 1, np.array(det.feature))
        #     bbox = det.to_tlbr()
        #     # bbox = np.array(det.bboxes[-1])
        #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # print(len(retain_box))
        # print(len(join_box))
        # retain_box.extend(join_box)
        # base_boxes = retain_box
        # # print(base_boxes)
        #
        # print(len(base_boxes))


# 返回结果


if __name__ == '__main__':
    main(YOLO())
