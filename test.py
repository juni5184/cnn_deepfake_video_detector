import threading
import time
import tensorflow as tf
from utils import label_map_util

class Worker(threading.Thread):
    """클래스 생성시 threading.Thread를 상속받아 만들면 된다"""

    def __init__(self, args, name=""):
        """__init__ 메소드 안에서 threading.Thread를 init한다"""
        threading.Thread.__init__(self)
        self.name = name
        self.args = args
        print('init')
        PATH_TO_CKPT = './face_model/frozen_inference_graph_face.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = './face_labelmap/face_labelmap.pbtxt'

        NUM_CLASSES = 2

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
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


    def run(self):
        """start()시 실제로 실행되는 부분이다"""
        print("{} is start : {}".format(threading.currentThread().getName(), self.args[0]))
        # image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # # the array based representation of the image will be used later in order to prepare the
        # # face_result image with boxes and labels on it.
        # # Expand dimensions since the face_model expects images to have shape: [fix_keep, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        #
        # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # # Each box represents a part of the image where a particular object was detected.
        # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # # Each score represent how level of confidence for each of the objects.
        # # Score is shown on the face_result image, together with the class label.
        # scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # # Actual detection.
        # # start_time = time.time()
        # (boxes, scores, classes, num_detections) = f_sess.run(
        #     [boxes, scores, classes, num_detections],
        #     feed_dict={image_tensor: image_np_expanded})
        #
        # # ========================================================
        # # Visualization of the results of a detection.
        # left, right, top, bottom, label_str = vis_util.visualize_boxes_and_labels_on_image_array(
        #     #          image_np,
        #     image,
        #     np.squeeze(boxes),
        #     np.squeeze(classes).astype(np.int32),
        #     np.squeeze(scores),
        #     f_category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=4)
        # print(label_str)


def main():
    for i in range(10):
        # threading.Thread 대신, 클래스명으로 쓰레드 객체를 생성하면 된다
        msg = "hello"
        th = Worker(name="[th cls {}]".format(i), args=(msg,))
        th.start()  # run()에 구현한 부분이 실행된다

if __name__ == "__main__":
    main()