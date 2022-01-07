# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'BD.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5 import QLabel
from PyQt5.QtWidgets import QFileDialog, QDialog, QApplication,QLabel 
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
import os
import cv2
import random
import datetime
import numpy as np
import sys

import subprocess


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDialog, QApplication,QLabel 
from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5 import QtCore, QtGui, QtWidgets

processed_files_path = os.path.join(os.getcwd()+'/test')
results_files_path = os.path.join(os.getcwd()+'/results')





from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1504, 782)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.Title_groupBox = QtWidgets.QGroupBox(Dialog)
        self.Title_groupBox.setGeometry(QtCore.QRect(0, -10, 1501, 50))
        self.Title_groupBox.setStyleSheet("background-color: rgb(28, 149, 255);")
        self.Title_groupBox.setTitle("")
        self.Title_groupBox.setObjectName("Title_groupBox")
        self.Title_label = QtWidgets.QLabel(self.Title_groupBox)
        self.Title_label.setGeometry(QtCore.QRect(20, 10, 541, 31))
        font = QtGui.QFont()
        font.setFamily("FreeSans")
        font.setPointSize(18)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.Title_label.setFont(font)
        self.Title_label.setStyleSheet("color: rgb(255, 255, 255);")
        self.Title_label.setObjectName("Title_label")
        self.logo_label = QtWidgets.QLabel(self.Title_groupBox)
        self.logo_label.setGeometry(QtCore.QRect(1030, 10, 541, 31))
        font = QtGui.QFont()
        font.setFamily("FreeSans")
        font.setPointSize(18)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.logo_label.setFont(font)
        self.logo_label.setStyleSheet("color: rgb(255, 255, 255);")
        self.logo_label.setText("HC Robotics")
        self.logo_label.setObjectName("logo_label")
        self.UploadButton = QtWidgets.QPushButton(Dialog)
        self.UploadButton.setGeometry(QtCore.QRect(500, 80, 100, 35))
        font = QtGui.QFont()
        font.setFamily("FreeSans")
        font.setPointSize(13)
        self.UploadButton.setFont(font)
        self.UploadButton.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(28, 149, 255);")
        self.UploadButton.setObjectName("UploadButton")
        self.processed_img_label_2 = QtWidgets.QLabel(Dialog)
        self.processed_img_label_2.setGeometry(QtCore.QRect(730, 80, 150, 30))
        font = QtGui.QFont()
        font.setFamily("FreeSans")
        font.setPointSize(14)
        self.processed_img_label_2.setFont(font)
        self.processed_img_label_2.setStyleSheet("")
        self.processed_img_label_2.setObjectName("processed_img_label_2")
        self.processed_img_label_3 = QtWidgets.QLabel(Dialog)
        self.processed_img_label_3.setGeometry(QtCore.QRect(170, 80, 251, 30))
        font = QtGui.QFont()
        font.setFamily("FreeSans")
        font.setPointSize(14)
        self.processed_img_label_3.setFont(font)
        self.processed_img_label_3.setStyleSheet("")
        self.processed_img_label_3.setObjectName("processed_img_label_3")
        self.ProcessButton = QtWidgets.QPushButton(Dialog)
        self.ProcessButton.setGeometry(QtCore.QRect(950, 80, 100, 35))
        font = QtGui.QFont()
        font.setFamily("FreeSans")
        font.setPointSize(13)
        self.ProcessButton.setFont(font)
        self.ProcessButton.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(28, 149, 255);")
        self.ProcessButton.setObjectName("ProcessButton")
        self.UPLabel = QtWidgets.QLabel(Dialog)
        self.UPLabel.setGeometry(QtCore.QRect(0, 130, 1501, 551))
        self.UPLabel.setStyleSheet("border:1px solid grey;")
        self.UPLabel.setText("")
        self.UPLabel.setScaledContents(True)
        self.UPLabel.setObjectName("UPLabel")
        self.BackwardButton = QtWidgets.QPushButton(Dialog)
        self.BackwardButton.setGeometry(QtCore.QRect(640, 700, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.BackwardButton.setFont(font)
        self.BackwardButton.setIconSize(QtCore.QSize(38, 38))
        self.BackwardButton.setObjectName("BackwardButton")
        self.ForwardButton = QtWidgets.QPushButton(Dialog)
        self.ForwardButton.setGeometry(QtCore.QRect(880, 700, 50, 30))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.ForwardButton.setFont(font)
        self.ForwardButton.setIconSize(QtCore.QSize(38, 38))
        self.ForwardButton.setObjectName("ForwardButton")
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(1160, 90, 161, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.ForwardButton.setDisabled(True)
        self.BackwardButton.setDisabled(True)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Title_label.setText(_translate("Dialog", "Boats Detection Desktop Application"))
        self.UploadButton.setText(_translate("Dialog", "Select Folder"))
        self.processed_img_label_2.setText(_translate("Dialog", "Process Images"))
        self.processed_img_label_3.setText(_translate("Dialog", "Select a Folder to Process"))
        self.ProcessButton.setText(_translate("Dialog", "Process"))
        self.BackwardButton.setText(_translate("Dialog", "<<"))
        self.ForwardButton.setText(_translate("Dialog", ">>"))
class App(QDialog):
    def __init__(self):
        super().__init__()
        loadUi("UI/BD.ui",self)
        self.setWindowTitle("BGR-AI | HC Robotics")
        # self.setWindowIcon(QIcon('UI/images/logo.jpg'))
        # self.total_files_count.setText('0')
        # self.Processed_files_count.setText('0')

        self.logo_label.setPixmap(QPixmap('UI/images/ROBOTIC_LOGO.png'))
        self.logo_label.show()
        # self.progress.hide()

        # self.ProcessBTN.setDisabled(True)
        # self.ProcessFilesViewBTN.setDisabled(True)
        

        self.UploadButton.clicked.connect(self.pick_folder)
        
        
        
        
    def pick_folder(self):

        # call(Bgr_frame.py)
        dialog = QFileDialog()
        self.folder_path = dialog.getExistingDirectory(None, "Select Folder")
        # print(self.folder_path)
        
        for i in range(101):
  
            # slowing down the loop
            time.sleep(0.03)
  
            # setting value to progress bar
            self.progressBar.setValue(i)
  
        fname=os.listdir(processed_files_path)
        # print(fname[0])

        
        # pixmap1 = QPixmap(os.path.join(processed_files_path,fname[0]))
        # self.UPLabel.setPixmap(pixmap1)
        # self.show()
        
        self.ProcessButton.clicked.connect(self.ProcessBoats)
    
    def ProcessBoats(self):
        
        for i in range(101):
  
            # slowing down the loop
            time.sleep(0.03)
  
            # setting value to progress bar
            self.progressBar.setValue(i)
  
        fname=os.listdir(results_files_path)
        # print(fname[0])

        
        pixmap1 = QPixmap(os.path.join(results_files_path,fname[0]))
        self.UPLabel.setPixmap(pixmap1)
        self.show()



        forward_number=2
        self.ForwardButton.clicked.connect(lambda:self.forward(forward_number))


    # def doAction(self):
  
        # setting for loop to set value of progress bar
        

    def forward(self,image_number):
        
        # image_number=0
        fname=os.listdir(results_files_path)
        # pixmap1 = QPixmap(os.path.join(processed_files_path,fname[image_number+1]))
        if(image_number>len(os.listdir(results_files_path))):
            # print("no image found")
            self.ForwardButton.setDisabled(True)
            # break;

        else:
            self.ForwardButton.setDisabled(False)
            pixmap1 = QPixmap(os.path.join(results_files_path,fname[image_number-1]))
            self.UPLabel.setPixmap(pixmap1)
            # self.Image_label2.setText(fname[image_number-1])

            self.ForwardButton.clicked.connect(lambda:self.forward(image_number+1))
            self.BackwardButton.clicked.connect(lambda:self.backward(image_number-1))
            image_number=image_number+1


        return image_number

        
        
    def backward(self,image_number):
        # print("checking................")
        fname=os.listdir(results_files_path)
        # pixmap1 = QPixmap(os.path.join(processed_files_path,fname[image_number+1]))
        if(image_number<0):
            # print("no image found")
            self.BackwardButton.setDisabled(True)

            # break;

        else:
            self.BackwardButton.setDisabled(False)
            pixmap1 = QPixmap(os.path.join(results_files_path,fname[image_number-1]))
            self.UPLabel.setPixmap(pixmap1)
            # self.Image_label2.setText(fname[image_number-1])

            self.ForwardButton.clicked.connect(lambda:self.forward(image_number+1))
            self.BackwardButton.clicked.connect(lambda:self.backward(image_number-1))
            image_number=image_number-1
        return image_number




#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# def download_images():
#     base_url = "G:/Tensorflow/workspace/training_demo/test/"
#     filenames = ['658.jpg', '659.jpg']
#     image_paths = []
#     for filename in filenames:
#         image_path = os.path.join(base_url,filename)
# #                                         untar=False)
#         image_path = pathlib.Path(image_path)
#         image_paths.append(str(image_path))
#     return image_paths

IMG_PATHS = "G:/Tensorflow/workspace/training_demo/test/"

l=os.listdir(IMG_PATHS)
IMAGE_PATHS=[]
for fname in l:
    IMAGE_PATHS.append(IMG_PATHS+fname)


# IMAGE_PATHS = download_images()


# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
PATH_TO_MODEL_DIR = 'G:/Tensorflow/models/research/object_detection/'        #download_model(MODEL_NAME, MODEL_DATE)


LABEL_FILENAME = 'label_map.pbtxt'
PATH_TO_LABELS = 'G:/Tensorflow/workspace/training_demo/annotations/label_map.pbtxt'          #download_labels(LABEL_FILENAME)

# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = 'G:/Tensorflow/workspace/training_demo/exported-models/my_model' + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    
    return np.array(Image.open(path))

counter=0
for image_path in IMAGE_PATHS:
    image = cv2.imread(image_path)
    fname=image_path.split('/')[-1]
    # print(fname)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    # image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = input_tensor[:, :, :, :3] # <= add this line
    detections = detect_fn(input_tensor)


    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections

  # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

  # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=0.20,
          agnostic_mode=False)
    counter+=1
    print('Processing........ It will take few minutes please wait.')
  # DISPLAYS OUTPUT IMAGE
  # cv2.imshow('Object Detector', image_with_detections)
    cv2.imwrite("G:/Tensorflow/workspace/training_demo/results/"+fname,image_with_detections)
  # CLOSES WINDOW ONCE KEY IS PRESSED
  # cv2.waitKey(0)
  # # CLEANUP
  # cv2.destroyAllWindows()

        
 
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainwindow=App()
    widget=QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(1504)
    widget.setFixedHeight(782)
    widget.show()
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
   # Dialog.show()
    sys.exit(app.exec_())



# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     Dialog = QtWidgets.QDialog()
#     ui = Ui_Dialog()
#     ui.setupUi(Dialog)
#     Dialog.show()
#     sys.exit(app.exec_())

