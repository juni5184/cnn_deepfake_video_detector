from detect_function import detect_face
from find_face_part import f_p_load_models, detect_face_part
from find_fake import load_fake_models, detect_fake
import os
import time
import pandas
from utils import label_map_util

import tensorflow as tf
import cv2
import numpy as np
import threading


def load_face_model(PATH_TO_CKPT):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return sess, detection_graph


font = cv2.FONT_HERSHEY_SIMPLEX
code_start = time.time()
# 전체 모델 세션에 업로드
f_sess, f_detection_graph  = load_face_model('./face_model/frozen_inference_graph_face.pb')
eye_sess, eye_detection_graph = load_face_model('./face_part_model/eyes_v6.pb')
nose_sess, nose_detection_graph = load_face_model('./face_part_model/nose_ssd_inception.pb')
mouth_sess, mouth_detection_graph = load_face_model('./face_part_model/z_mouth_v6_ssd.pb')

# f_d_sess, f_d_detection_graph = load_face_model('./models/face_dotnoise_v11_ssd.pb')
# f_g_sess, f_g_detection_graph = load_face_model('./models/face_gridnoise_v7_ssd.pb')
# f_blur_sess, f_blur_detection_graph = load_face_model('./models/face_blur_faster.pb')
#
# e_d_sess, e_d_detection_graph = load_face_model('./models/eye_dotnoise_v1_ssd.pb')
# e_g_sess, e_g_detection_graph = load_face_model('models/eye_gridnoise_v2_ssd.pb')
#
# n_d_sess, n_d_detection_graph = load_face_model('./models/nose_dotnoise_v3_ssd_50k.pb')
# n_g_sess, n_g_detection_graph = load_face_model('./models/nose_gridnoise_v7_ssd.pb')
#
# m_d_sess, m_d_detection_graph = load_face_model('./models/mouth_dotnoise_v3_ssd.pb')
# m_g_sess, m_g_detection_graph = load_face_model('./models/mouth_gridnoise_v2_ssd.pb')

# folder_path = "G:\\DFDC_rnn\\videos\\test\\"
folder_path = "G:\\DFDC_rnn\\videos\\real_frames\\"
folder_list = os.listdir(folder_path)

eye =[]
nose =[]
mouth =[]
face =[]
start = time.time()
for img_item in folder_list:
    print(img_item)
    frame = cv2.imread(folder_path+img_item)

    result_rnn = []
    th = threading.Thread(target=detect_face,
                            args=(face, eye, nose, mouth, result_rnn, frame, f_sess, f_detection_graph,
                                eye_sess, eye_detection_graph,
                                nose_sess, nose_detection_graph,
                                mouth_sess, mouth_detection_graph))
                                # f_d_sess, f_d_detection_graph,
                                # f_g_sess, f_g_detection_graph,
                                # f_blur_sess, f_blur_detection_graph,
                                # e_d_sess, e_d_detection_graph,
                                # e_g_sess, e_g_detection_graph,
                                # n_d_sess, n_d_detection_graph,
                                # n_g_sess, n_g_detection_graph,
                                # m_d_sess, m_d_detection_graph,
                                # m_g_sess, m_g_detection_graph))
    th.start()
    th.join()

    # str_result = ""
    # if result_rnn == []:
    #     result_rnn = ['face 0%']
        # print('none', result_rnn)

    y0, dy = 50, 35
    for j, label in enumerate(result_rnn):
        y = y0 + j * dy
        txt_img = cv2.putText(frame, str(label), (50, y), font, 1, (0, 0, 255), 3)
    # cv2.imwrite("G:\\test\\" + folder_name + "\\txt_" + item, txt_img)
    print(len(face), len(eye), len(nose), len(mouth))
    cv2.imwrite( "G:\\DFDC_rnn\\videos\\real_frames_result\\" + img_item, frame)

print('final', len(face),  len(eye),len(nose), len(mouth))
# print('model process time ', time.time() - start)
# print('process time ', time.time() - code_start)
