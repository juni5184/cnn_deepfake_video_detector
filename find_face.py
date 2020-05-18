#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_face as vis_util
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def load_face_model():
    # Path to frozen detection graph. This is the actual face_model that is used for the object detection.
    PATH_TO_CKPT = './face_model/frozen_inference_graph_face.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './face_labelmap/face_labelmap.pbtxt'

    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # out = None
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    return sess, detection_graph, category_index


def detect_face (PATH_TO_VIDEO, number, count, f_sess, f_detection_graph, f_category_index) :
    code_start = time.time()

    num = number
    c = count
    video_name = PATH_TO_VIDEO.split('.')[0]
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    face_list = []
    frame_num = 1000
    label_list = []

    # print("얼굴 찾는 중..")
    while frame_num:
        frame_num -= 1
        ret, image = cap.read()
        if ret == 0:
            break
        # video_name = video.split('.mp4')[0]
        # if out is None:
        #     [h, w] = image.shape[:2]
        #     # 아래 이름 비디오 생성
        #     out = cv2.VideoWriter(video_name+"_out.mp4", 0, 25.0, (w, h))
        if (int(cap.get(1)) % num == c):
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # the array based representation of the image will be used later in order to prepare the
            # face_result image with boxes and labels on it.
            # Expand dimensions since the face_model expects images to have shape: [fix_keep, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = f_detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = f_detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the face_result image, together with the class label.
            scores = f_detection_graph.get_tensor_by_name('detection_scores:0')
            classes = f_detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = f_detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            # start_time = time.time()
            (boxes, scores, classes, num_detections) = f_sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # ========================================================
            # Visualization of the results of a detection.
            left, right, top, bottom, label_str = vis_util.visualize_boxes_and_labels_on_image_array(
                #          image_np,
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                f_category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            if left < 0 : left = 0
            elif right < 0 : right = 0
            elif top < 0: top = 0
            elif bottom < 0 : bottom = 0

            if label_str == "" :
                label_list.append("nothing")
            else :
                label_list.append(label_str)
            face_result = [left, right, top, bottom]
            # print(str(face_result))
            # print("face !!! : ", face_result)
            # if left != 0 and right != 0 and top != 0 and bottom != 0:
            #     print('얼굴 없음')
            face_list += [face_result]

            # 박스 확인용 이미지 출력
            # output_dir = './test_videos/face/' + video_name + '/'
            # if not os.path.exists('./test_videos/face/' + video_name):
            #     os.mkdir('./test_videos/face/' + video_name)
            #     print(str('./test_videos/face/' + video_name))
            # if not os.path.exists(output_dir):
            #     os.mkdir(output_dir)
            # # print(output_dir+model_name+"_"+str(face_num)+'.jpg')
            # cv2.imwrite(output_dir + "_" + str(int(cap.get(fix_keep))) + '.jpg', image)
            # cv2.imshow('Object detector', image)
            # # Press 'q' to quit
            # if cv2.waitKey(fix_keep) == ord('q'):
            #     break
        # 얼굴 찾는 비디오 생성
        # out.write(image)
    # cap.release()
    # out.release()
    load_face_model_time = time.time() - code_start

    return face_list, load_face_model_time, label_list
