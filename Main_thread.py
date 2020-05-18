from detect_function import detect_face
from detection_filter3 import detect_face_filter3
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

import dlib


def load_face_model(PATH_TO_CKPT):
    print(PATH_TO_CKPT)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return sess, detection_graph


code_start = time.time()
# 전체 모델 세션에 업로드
f_sess, f_detection_graph  = load_face_model('./face_model/frozen_inference_graph_face.pb')
eye_sess, eye_detection_graph = load_face_model('./face_part_model/eyes_v6.pb')
nose_sess, nose_detection_graph = load_face_model('./face_part_model/nose_ssd_inception.pb')
mouth_sess, mouth_detection_graph = load_face_model('./face_part_model/z_mouth_v6_ssd.pb')

f_d_sess, f_d_detection_graph = load_face_model('./models/face_dotnoise_v11_ssd.pb')
f_g_sess, f_g_detection_graph = load_face_model('./models/face_gridnoise_v7_ssd.pb')
f_blur_sess, f_blur_detection_graph = load_face_model('./models/face_blur_faster.pb')

e_d_sess, e_d_detection_graph = load_face_model('./models/eye_dotnoise_v1_ssd.pb')
e_g_sess, e_g_detection_graph = load_face_model('models/eye_gridnoise_v2_ssd.pb')
# e_glasses_sess, e_glasses_detection_graph = load_face_model('keep/eye_glasses_v2_ssd.pb')

n_d_sess, n_d_detection_graph = load_face_model('./models/nose_dotnoise_v3_ssd_50k.pb')
n_g_sess, n_g_detection_graph = load_face_model('./models/nose_gridnoise_v7_ssd.pb')
# n_bridge_sess, n_bridge_detection_graph = load_face_model('keep/nose_bridge_v3_ssd.pb')

m_d_sess, m_d_detection_graph = load_face_model('./models/mouth_dotnoise_v3_ssd.pb')
m_g_sess, m_g_detection_graph = load_face_model('./models/mouth_gridnoise_v2_ssd.pb')
# m_noteeth_sess, m_noteeth_detection_graph = load_face_model('keep/mouth_noteeth_v1_ssd.pb')


e_b_sess, e_b_detection_graph = load_face_model('./models_3/eye_bridge_v4_ssd.pb')
e_t_l_sess, e_t_l_detection_graph = load_face_model('./models_3/eye_temple_left_v2_ssd.pb')
e_t_r_sess, e_t_r_detection_graph = load_face_model('./models_3/eye_temple_right_v2_ssd.pb')
n_c_sess, n_c_detection_graph = load_face_model('./models_3/nose_cheekline_v2_ssd.pb')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# =============================================================================================
# real
# for i in range (36) :
#     # 기존에 분류를 진행해놓은 폴더
#     if i < 10 :
#         folder_path = "G:\\Facebook_Dataset_video\\dfdc_train_part_0" + str(i) + "\\dfdc_train_0" + str(i) + "_real\\"
#     else :
#         folder_path = "G:\\Facebook_Dataset_video\\dfdc_train_part_" + str(i) +"\\dfdc_train_" + str(i) +"_real\\"
    # folder_path = "G:\\Facebook_Dataset_video\\dfdc_train_part_04\\dfdc_train_04_real\\"
# fake
video_path = 'G:\\DFDC_rnn\\Fake_videos\\'
video_list = os.listdir(video_path)
# print(video_list)
for video in video_list :
    folder_path = video_path + video + "\\"

    folder_list = os.listdir(folder_path)

    # txt_output_dir = "G:\\DFDC_rnn\\rnn_txt\\final\\real_txt_" + str(i) + "\\"
    # if not os.path.exists(txt_output_dir):
    #     os.mkdir(txt_output_dir)

    # 영상 라벨 - real 인지 fake 인지 입력
    # fake = 0 / real  = 1
    # final_label = 1
    final_label = 0

    # ===========================================================================
    # num = input("영상에서 몇 프레임마다 detect 할지 입력 : ")
    # count = input("몇번째 프레임을 확인할지 입력 : ")
    num = 30
    count = 15
    start = time.time()
    label_list = []
    for video_item in folder_list:
        label_list = "video_name/frame_num/label+percent/video_tag\n"

        # print(video_item)
        PATH_TO_VIDEO = folder_path + video_item
        print(video_item)

        video_name = PATH_TO_VIDEO.split('.')[0]
        vn = video_name.split('\\')[4]
        cap = cv2.VideoCapture(PATH_TO_VIDEO)

        frame_num = 300

        while frame_num:
            result_rnn = []
            result_face =[]
            result_rnn_3 =[]

            ret, frame = cap.read()
            # face_result = list()
            if ret == 0:
                break
            if (int(cap.get(1)) % num == count):
                img_num = int(cap.get(1))
                th = threading.Thread(target=detect_face,
                                      args=(result_rnn, result_face, frame, f_sess, f_detection_graph,
                                            eye_sess, eye_detection_graph,
                                            nose_sess, nose_detection_graph,
                                            mouth_sess, mouth_detection_graph,
                                            f_d_sess, f_d_detection_graph,
                                            f_g_sess, f_g_detection_graph,
                                            f_blur_sess, f_blur_detection_graph,
                                            e_d_sess, e_d_detection_graph,
                                            e_g_sess, e_g_detection_graph,
                                            n_d_sess, n_d_detection_graph,
                                            n_g_sess, n_g_detection_graph,
                                            m_d_sess, m_d_detection_graph,
                                            m_g_sess, m_g_detection_graph
                                            ))

                th3 = threading.Thread(target=detect_face_filter3,
                                       args=(result_rnn_3, frame,
                                             detector, predictor,
                                             f_sess, f_detection_graph,
                                             e_b_sess, e_b_detection_graph,
                                             e_t_l_sess, e_t_l_detection_graph,
                                             e_t_r_sess, e_t_r_detection_graph,
                                             n_c_sess, n_c_detection_graph))
                th.start()
                th3.start()
                th.join()
                th3.join()

                str_result = ""
                str_result_3 = ""
                if result_rnn is not None:
                    for i in range(len(result_rnn)):
                        str_result += (result_rnn[i])
                if result_rnn_3 is not None:
                    for i in range(len(result_rnn_3)):
                        str_result_3 += (result_rnn_3[i])

                # print(result_face)
                if result_face == [''] or result_face == []:
                    result_face.clear()
                    result_face.append('unknown')
                    # fake = 0 real = 1
                    # print(vn,'/', img_num,'/', str_result,'/',str(final_label))

                print((str(vn) + '/' + str(img_num) + '/'+str(result_face[0])+
                               '*'+ str(str_result) + '/' + str(final_label)) )
                print((str(vn) + '/' + str(img_num) + '/' + str(str_result_3) + '/' + str(final_label)))
                # label_list += (str(vn) + '/' + str(img_num) + '/'+str(result_face[0])+
                #                '*'+ str(str_result) + '/' + str(final_label)) + "\n"
                # print(label_list)
                cv2.imshow('Object detector', frame)
                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break
            frame_num -= 1


        # txt = open(txt_output_dir + vn + '.txt', 'w', encoding='utf-8', newline='')
        # txt.write(label_list)
        # txt.close()

print('model process time ', time.time() - start)
print('process time ', time.time() - code_start)
