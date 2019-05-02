import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import csv
from numpy import *
import pandas


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

#directories used
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'test.avi'
targetdir='out_images1'

# current working dir
CWD_PATH = os.getcwd()

# Path requuired for the following
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

PATH_TO_OUTPUT_IMAGES = os.path.join(CWD_PATH,targetdir)
# no of classification classes
NUM_CLASSES = 9


# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
curr_frame_no = 1
a=[]
class1=[]
class2=[]
class3=[]
class4=[]
class5=[]
i=[]
i1=[]
i2=[]
no1=[]
cpt=1
composite_list=[]
while(video.isOpened()):
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # actual detection by running the model
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
    #converting the array to list
    no1=num[0].astype(np.int64)
    class1=scores*100
    class2=list(class1.astype(np.int64).flat)
    i=class2[0:no1]
    class3=list(classes.astype(np.int64).flat)
    i1=class3[0:no1]
    class4=boxes*10000
    class5=list(class4.astype(np.int64).flat)
    i2=filter(lambda a: a != 0, class5)
    composite_list = [i2[x:x+4] for x in range(0, len(i2),4)]
    #storing in dataframe
    df = pandas.DataFrame(data={"A_Frame_no": curr_frame_no, "Score": i ,"Classes": i1 , "No_of_objects" : no1 , "Boxes" :composite_list })
    #appending df to list
    a.append(df)
    #concating the list
    masterDF = pandas.concat(a, ignore_index=True)
    masterDF.to_csv("abc1.csv", sep=',',index=False)
    curr_frame_no += 1

    #visualizing the output

    vis_util.visualize_boxes_and_labels_on_image_array(
    frame,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.70)

    #cv2.imshow('Object detector', frame)
    cv2.imwrite(os.path.join(targetdir, "%i.jpg" %cpt), frame)   
    cpt += 1
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break



# Clean up
video.release()
cv2.destroyAllWindows()
