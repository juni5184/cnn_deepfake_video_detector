# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import math
import time

# Import utilites
from utils import label_map_util
from utils import visualization_utils_face_part as vis_util
import filter_def

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

f_list = []
def f_p_load_models() :
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'face_part_model'
    LABELMAP_NAME = 'face_part_labelmap'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    model_list = os.listdir(CWD_PATH+"\\"+MODEL_NAME)
    labelmap_list = os.listdir(CWD_PATH+"\\"+LABELMAP_NAME)

    # =========================================================================
    # model 에 따라서 바뀌는 값들 배열 선언
    sess = {}
    detection_graph = {}
    categories = {}
    category_index = {}
    i = 0
    # 모델 불러오는 부분 => 반복 (여러 모델 불러오기)
    for model, labelmap in zip(model_list, labelmap_list):
        # print(str(len(model_list))+"개 중에 "+str(i+fix_keep)+"번째 모델 불러오는 중")
        # print(model+", "+labelmap)

        # Path to frozen detection graph .pb file, which contains the model that is used for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, model)
        # print("model_path : "+PATH_TO_CKPT)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, LABELMAP_NAME, labelmap)
        # print("label_path : "+PATH_TO_LABELS)

        # Number of classes the object detector can identify
        # max_num_classes 를 지정해주는 값
        NUM_CLASSES = 1

        # Load the label map.
        # Label maps map indices to category names
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories[i] = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index[i] = label_map_util.create_category_index(categories[i])

        # Load the Tensorflow model into memory.
        # Tensorflow 모델을 메모리에 로드
        # detection_graph 배열로 만들어서 이름 바꿔주면서 image 넣기
        detection_graph[i] = tf.Graph()
        with detection_graph[i].as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess[i] = tf.Session(graph=detection_graph[i])
        i = i + 1
    # print("모델 불러오는데 걸리는 시간 : "+str(round(time.time() - code_start,5)))
    # ==========================================================================

    return sess, detection_graph, category_index


def detect_face_part(face_list, video, number, count, sess, detection_graph, category_index) :
    code_start = time.time()

    # Name of the directory containing the object detection module we're using
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    model_list = os.listdir(CWD_PATH + "\\" + 'face_part_model')

    # Open video file
    f_list = face_list
    eye_list = []
    nose_list = []
    mouth_list = []

    video_name = video.split('.')[1]
    # print("nameaaaaaa : "+video_name)
    cap = cv2.VideoCapture(video)

    face_num = 0
    num = number
    c = count
    label_list = []
    # out = None
    while (cap.isOpened()):
        # Acquire frame and expand frame dimensions to have shape: [fix_keep, None, None, 3]
        # 프레임을 획득하고 모양을 갖도록 프레임 크기를 확장합니다. [fix_keep, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        # 즉, 열의 각 항목에 픽셀 RGB 값이있는 단일 열 배열
        ret, frame = cap.read()
        if ret == 0:
            break
        # 영상 출력
        # if out is None:
        #     [h, w] = frame.shape[:2]
        #     out = cv2.VideoWriter("."+video_name + "_out.mp4", 0, 25.0, (w, h))
        if (int(cap.get(1)) % num == c ):

            label_percent = ""
            # 모델 개수만큼 반복
            for j in range(len(sess)):
                # print(j)
                if f_list is None : break
                # 얼굴 좌표 찾기
                # im_width, im_height, c= frame.shape
                # (xmin, xmax, ymin, ymax) = f_list[face_num]
                # (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                #                               ymin * im_height, ymax * im_height)
                # 소수점 올림, 내림 /  ceil 올림, floor 내림
                (left, right, top, bottom) = f_list[face_num]
                left = math.floor(left)
                right = math.ceil(right)
                top = math.floor(top)
                bottom = math.ceil(bottom)
                height = bottom - top
                width = right - left
                # print(left, right, top, bottom)

                # left > 0인 경우 얼굴 좌표가 있는 것
                # 얼굴이 있는 경우 모델 통과
                if left != 0 and height>0 and width>0:
                    model_name = model_list[j].split('.')[0]
                    # image = frame[int(top - 50):int(bottom + 50), int(left - 50):int(right + 50)]
                    # 각 부위별로 크롭하고 필터를 거침
                    image = frame[int(top):int(bottom), int(left):int(right)]
                    # image = frame[int(top - int(round(height / 6))):int(bottom+int(round(height / 6))),
                    #          int(left - int(round(width / 6))):int(right + int(round(width / 6)))]
                    if 'eye' in model_name:
                        #     f_image = filter_def.eyebrow_doubleline(image)
                        f_image = filter_def.eye(image)
                    elif 'nose' in model_name:
                        f_image = filter_def.nose(image)
                    elif 'mouth' in model_name:
                        # f_image = filter_def.mouth_h_b(image)
                        f_image = filter_def.mouth(image)
                    else:
                        f_image = image

                    # 크롭, 필터까지 다 거친 이미지 배열
                    image_np_expanded = np.expand_dims(f_image, axis=0)

                    image_tensor = detection_graph[j].get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    # 각 상자는 특정 물체가 감지된 이미지의 일부를 나타냅니다.
                    boxes = detection_graph[j].get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # 각 점수는 각 객체에 대한 신뢰 수준을 나타냅니다.
                    # Score is shown on the result image, together with the class label.
                    # 점수는 클래스 라벨과 함께 결과 이미지에 표시됩니다.
                    scores = detection_graph[j].get_tensor_by_name('detection_scores:0')
                    classes = detection_graph[j].get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph[j].get_tensor_by_name('num_detections:0')

                    # Perform the actual detection by running the model with the image as input
                    # 이미지를 입력으로 하여 모델을 실행하여 실제 감지 수행
                    # 여기서 모델을 거친다
                    # inference_start = time.time()
                    (boxes, scores, classes, num_detections) = sess[j].run(
                        [boxes, scores, classes, num_detections],
                        # feed_dict = {image_tensor: frame_expanded})
                        feed_dict={image_tensor:image_np_expanded})

                    # Draw the results of the detection (aka 'visualize the results')
                    # 탐지 결과 그리기 (일명 '결과 시각화')
                    label_str, xmin, xmax, ymin, ymax = vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index[j],
                        use_normalized_coordinates=True,
                        line_thickness=4,
                        min_score_thresh=0.3)
                    # 모델 거친 결과 출력
                    # if label_str != "" : print(label_str)
                    # print(left, right, top, bottom)
                    # print(label_str)
                    label_percent += label_str

                    # label_list.append(label_str)

                    # if 'eye' in model_name:
                    #     face_result = face_list
                    if 'eye' in model_name:
                        eye_result = [xmin, xmax, ymin, ymax ]
                        eye_list += [eye_result]
                        # print(str('eye_result : ') +str(eye_result))
                    elif 'nose' in model_name:
                        nose_result = [xmin, xmax, ymin, ymax ]
                        nose_list += [nose_result]
                        # print(str('nose_result : ') + str(nose_result))
                    elif 'mouth' in model_name:
                        mouth_result = [xmin, xmax, ymin, ymax ]
                        mouth_list += [mouth_result]
                        # print(str('mouth_result : ') + str(mouth_result))
                    # elapsed_time = time.time() - inference_start
                    # print('inference time cost: {}'.format(elapsed_time))
            # print(label_percent)
            face_num += 1
            if face_num >= len(face_list) : break
            label_list.append(label_percent)
            # All the results have been drawn on the frame,
            # so it's time to display it.
            # 프레임에 그림을 그리고 이미지로 보여줌
            # cv2.imshow('Object detector', frame)
            # # Press 'q' to quit
            # if cv2.waitKey(1) == ord('q'):
            #     break

        # Clean up
        # 얼굴 찾는 비디오 생성
    #     out.write(frame)
    # out.release()
    # cap.release()
    # cv2.destroyAllWindows()
    load_model_time = time.time() - code_start
    # print()

    return load_model_time, eye_list, nose_list, mouth_list, label_list