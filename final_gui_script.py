import sys
from PyQt4 import QtGui, QtCore
import cv2 
import numpy as np
import pandas as pd
import itertools
import os
import matplotlib.pyplot as plt


import tensorflow as tf
import sys
import csv
from numpy import *


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(500, 500, 1500, 1300)
        self.setWindowTitle("Cross traffic event identification")
        self.setWindowIcon(QtGui.QIcon('mainlogo.png'))
        
        

        extractAction = QtGui.QAction("& click to annotate frame by frame ", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('leave the app')
        extractAction.triggered.connect(self.annotate)

        
        extractAction1 = QtGui.QAction("& click to generate the output video ", self)
        extractAction1.setShortcut("Ctrl+Z")
        extractAction1.setStatusTip('leave the app')
        extractAction1.triggered.connect(self.generate_video)


        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&Annotate')
        fileMenu.addAction(extractAction)
        fileMenu1 = mainMenu.addMenu('&Create output video')
        fileMenu1.addAction(extractAction1)

        self.home()



    def home(self):
        btn1 = QtGui.QPushButton("Input video", self)
        btn1.clicked.connect(self.play)
        btn1.resize(100, 80)
        btn1.move(100,100)


        btn2 = QtGui.QPushButton("Output video", self)
        btn2.clicked.connect(self.play1)
        btn2.resize(100, 80)
        btn2.move(100,300)



        btn3 = QtGui.QPushButton("Object's Pie Chart", self)
        btn3.clicked.connect(self.graph)
        btn3.resize(130, 50)
        btn3.move(400,100)



        btn4 = QtGui.QPushButton("Bar chart", self)
        btn4.clicked.connect(self.graph1)
        btn4.resize(130, 50)
        btn4.move(400,200)



        btn5 = QtGui.QPushButton("Output Bar chart", self)
        btn5.clicked.connect(self.graph2)
        btn5.resize(130, 50)
        btn5.move(400,300)

        btn6 = QtGui.QPushButton("Frame's pie chart", self)
        btn6.clicked.connect(self.graph3)
        btn6.resize(130, 50)
        btn6.move(400,400)

        btn = QtGui.QPushButton("Quit", self)
        btn.clicked.connect(self.close_application)
        btn.resize(100, 80)
        btn.move(100,500)

        p = QtGui.QPalette()
        brush = QtGui.QBrush(QtCore.Qt.white,QtGui.QPixmap('aa.png'))
        p.setBrush(QtGui.QPalette.Active,QtGui.QPalette.Window,brush)
        p.setBrush(QtGui.QPalette.Inactive,QtGui.QPalette.Window,brush)
        p.setBrush(QtGui.QPalette.Disabled,QtGui.QPalette.Window,brush)
        self.setPalette(p)


		
        self.le1 = QtGui.QLineEdit()

       
        self.show()



    def gettext(self):
      text, ok = QtGui.QInputDialog.getText(self, 'Text Input Dialog', 'Enter video name:')
		
      if ok:
         self.le1.setText(str(text))
         abc=text
         return abc


    def play(self):
        abc = self.gettext()
        cap = cv2.VideoCapture(str(abc)+'.avi')  
        while(cap.isOpened()): 
            ret, frame = cap.read() 
            if ret == True:  
                cv2.imshow('Frame', frame)  
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break
        cap.release()  
        cv2.destroyAllWindows()


    def play1(self):
        cap = cv2.VideoCapture('video.avi')  
        while(cap.isOpened()): 
            ret, frame = cap.read() 
            if ret == True:  
                cv2.imshow('Frame', frame)  
                if cv2.waitKey(60) & 0xFF == ord('q'): 
                    break
        cap.release()  
        cv2.destroyAllWindows()

    def graph(self):
        data=pd.read_csv("abc.csv")
        A=60
        score_60=data.loc[(data['Score'] >= A)]
        score_60_len=len(score_60)
        cz=len(data.index)
        dz=(score_60_len*100)/cz
        ez=(100-dz)
        values = [cz,score_60_len]
        size=[ez,dz]
        colors = ['red', 'grey']
        labels = ['total_no_of_objects_detected','objects_with_predection>70']
        explode = (0.1, 0)
        plt.pie(size, colors=colors,autopct='%1.1f%%', labels= values,explode=explode,counterclock=False, shadow=True)
        plt.title('objects detection accuracy analysis')
        plt.legend(labels,loc="upper left")
        plt.axis('equal')
        plt.show()



    def graph1(self):
        data=pd.read_csv("abc.csv")
        f=len(data.index)
        g=len(data[data['Classes'] == 1.0])
        h=len(data[data['Classes'] == 2.0])
        i=len(data[data['Classes'] == 3.0])
        j=len(data[data['Classes'] == 4.0])
        k=len(data[data['Classes'] == 5.0])
        l=len(data[data['Classes'] == 6.0])
        m=len(data[data['Classes'] == 7.0])
        n=len(data[data['Classes'] == 8.0])
        o=len(data[data['Classes'] == 9.0])
        x = ['total_objects','sign board','straight','left','right','side view car','side view bike','left u turn','straight right','straight left']
        y = [f,g,h,i,j,k,l,m,n,o]

        fig, ax = plt.subplots()    
        width = 0.75 # the width of the bars 
        ind = np.arange(len(y))  # the x locations for the groups
        ax.barh(ind, y, width,color="red")
        ax.set_yticks(ind+width/2)
        ax.set_yticklabels(x, minor=False)
        for i, v in enumerate(y):
            ax.text(v + 3, i + .25, str(v), color='grey', fontweight='bold')
        plt.title('Detected object analysis')
        plt.xlabel('count of objects')
        plt.ylabel('objects')      
        plt.show()


    def graph2(self):
        main_list=[(2.0,5.0), (2.0, 6.0), (2.0, 7.0),(2.0,8.0), (2.0, 9.0), (5.0, 2.0),(6.0,2.0), (7.0, 2.0), (8.0, 2.0),
        (9.0,2.0),(3.0,5.0) ,(3.0, 6.0), (3.0, 8.0),(5.0,3.0), (6.0, 3.0), (8.0, 3.0),(4.0,5.0), (4.0, 6.0), (4.0, 7.0),(4.0,9.0),
        (5.0,4.0), (6.0, 4.0), (7.0, 4.0),(9.0,4.0), (5.0, 6.0), (5.0, 7.0),(5.0,8.0), (5.0, 9.0), (6.0, 5.0),(7.0, 5.0),
        (8.0,5.0), (9.0, 5.0), (6.0, 7.0),(6.0,8.0), (6.0, 9.0), (7.0, 6.0),(8.0,6.0), (9.0, 6.0), (7.0, 8.0),(8.0,7.0),
        (8.0, 9.0),(9.0,8.0)]
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        list5=[]
        a=[]
        b=[]
        d=[]
        c=0
        resultList=[]
        max_frame_no=[]
        curr_frame_no=1
        final_frame_list=[]
        #reading a csv
        data=pd.read_csv("abc.csv")
        A=70
        #selecting frames with detected object accuracy >70
        score_70=data.loc[(data['Score'] >= A)]
        score_70_len=len(score_70)
        #selecting the frames with more than one object detected
        dataframe= score_70.loc[score_70.No_of_objects!= 1]
        #max no of the frame
        max_frame_no=dataframe['A_Frame_no'].max()

        while(curr_frame_no!=max_frame_no):
            #select all the rows of a particular frame no
            frame=dataframe.loc[(dataframe['A_Frame_no']==curr_frame_no)]
            count_row = frame.shape[0]
            if(count_row>1):
                #creating the combination of the objects detected
                list1 = list(itertools.combinations(frame['Classes'], 2))
                list2 = np.unique(list1, axis=0)
                for i in list2:
                    list3.append(tuple(i))
                list4=(main_list+list3)
                a=len(list4)
                list5=np.unique(list4,axis=0)
                b=len(list5)
                if(a!=b):
                    c=frame['A_Frame_no'].unique()      
                    d=c[0].astype(np.int64)
                    final_frame_list.append(d)
                    curr_frame_no=curr_frame_no+1
                    del list1[:]
                    del list3[:]
                    del list4[:]
                else:
                    curr_frame_no=curr_frame_no+1
            else:
                curr_frame_no=curr_frame_no+1



        #3's combination for cross traffic
        main_list1=[(2.0,3.0,4.0),(2.0,4.0,3.0),(3.0,2.0,4.0),(3.0,4.0,2.0),(4.0,3.0,2.0),(4.0,2.0,3.0),(2.0,3.0,5.0),(2.0,5.0,3.0),(3.0,2.0,5.0),(3.0,5.0,2.0),(5.0,3.0,2.0),(5.0,2.0,3.0),
        (2.0,3.0,6.0),(2.0,6.0,3.0),(3.0,2.0,6.0),(3.0,6.0,2.0),(6.0,3.0,2.0),(6.0,2.0,3.0),(2.0,3.0,8.0),(2.0,8.0,3.0),(3.0,2.0,8.0),(3.0,8.0,2.0),(8.0,3.0,2.0),(8.0,2.0,3.0),
        (2.0,4.0,5.0),(2.0,5.0,4.0),(4.0,2.0,5.0),(4.0,5.0,2.0),(5.0,4.0,2.0),(5.0,2.0,4.0),(2.0,4.0,6.0),(2.0,6.0,4.0),(4.0,2.0,6.0),(4.0,6.0,2.0),(6.0,4.0,2.0),(6.0,2.0,4.0),
        (2.0,4.0,7.0),(2.0,7.0,4.0),(4.0,2.0,7.0),(4.0,7.0,2.0),(7.0,4.0,2.0),(7.0,2.0,4.0),(2.0,4.0,9.0),(2.0,9.0,4.0),(4.0,2.0,9.0),(4.0,9.0,2.0),(9.0,4.0,2.0),(9.0,2.0,4.0),
        (5.0,6.0,7.0),(5.0,7.0,6.0),(6.0,5.0,7.0),(6.0,7.0,5.0),(7.0,5.0,6.0),(7.0,6.0,5.0),(5.0,6.0,8.0),(5.0,8.0,6.0),(6.0,5.0,8.0),(6.0,8.0,5.0),(8.0,5.0,6.0),(8.0,6.0,5.0),
        (5.0,6.0,9.0),(5.0,9.0,6.0),(6.0,5.0,9.0),(6.0,9.0,5.0),(9.0,5.0,6.0),(9.0,6.0,5.0),(6.0,7.0,8.0),(6.0,8.0,7.0),(7.0,6.0,8.0),(7.0,8.0,6.0),(8.0,6.0,7.0),(8.0,7.0,6.0),
        (6.0,7.0,9.0),(6.0,9.0,7.0),(7.0,6.0,9.0),(7.0,9.0,6.0),(9.0,6.0,7.0),(9.0,7.0,6.0),(7.0,8.0,9.0),(7.0,9.0,8.0),(8.0,7.0,9.0),(8.0,9.0,7.0),(9.0,8.0,7.0),(9.0,7.0,8.0),
        (5.0,5.0,3.0),(5.0,3.0,5.0),(3.0,5.0,5.0),(5.0,5.0,4.0),(5.0,4.0,5.0),(4.0,5.0,5.0),(5.0,5.0,7.0),(5.0,7.0,5.0),(7.0,5.0,5.0),(5.0,5.0,8.0),(5.0,8.0,5.0),(8.0,5.0,5.0),
        (5.0,5.0,9.0),(5.0,9.0,5.0),(9.0,5.0,5.0),(6.0,6.0,3.0),(6.0,3.0,6.0),(3.0,6.0,6.0),(6.0,6.0,4.0),(6.0,4.0,6.0),(4.0,6.0,6.0),(6.0,6.0,7.0),(6.0,7.0,6.0),(7.0,6.0,6.0),
        (6.0,6.0,8.0),(6.0,8.0,6.0),(8.0,6.0,6.0),(6.0,6.0,9.0),(6.0,9.0,6.0),(9.0,6.0,6.0),(2.0,2.0,5.0),(2.0,5.0,2.0),(5.0,2.0,2.0)]
        list11=[]
        list21=[]
        list31=[]
        list41=[]
        list51=[]
        a1=[]
        b1=[]
        d1=[]
        c1=0
        final_frame_list1=[]
        curr_frame_no1=1




        while(curr_frame_no!=max_frame_no):
            #select all the rows of a particular frame no
            frame=dataframe.loc[(dataframe['A_Frame_no']==curr_frame_no1)]
            count_row = frame.shape[0]
            if(count_row>1):
                #creating the combination of the objects detected
                list11 = list(itertools.combinations(frame['Classes'], 2))
                list21 = np.unique(list11, axis=0)
                for i in list21:
                    list31.append(tuple(i))
                list4=(main_list1+list31)
                a1=len(list41)
                list51=np.unique(list41,axis=0)
                b1=len(list51)
                if(a!=b):
                    c1=frame['A_Frame_no'].unique()      
                    d1=c1[0].astype(np.int64)
                    final_frame_list1.append(d1)
                    curr_frame_no1=curr_frame_no1+1
                    del list11[:]
                    del list31[:]
                    del list41[:]
                else:
                    curr_frame_no1=curr_frame_no1+1
            else:
                curr_frame_no1=curr_frame_no1+1




        #final list with cross traffic
        resultList=sorted(np.unique(final_frame_list+final_frame_list1))
        resultList_len=[]
        resultList_len=len(resultList)
        data1=data.loc[data['A_Frame_no'].isin(resultList)]
        f1=len(data1.index)
        g1=len(data1[data1['Classes'] == 1.0])
        h1=len(data1[data1['Classes'] == 2.0])
        i1=len(data1[data1['Classes'] == 3.0])
        j1=len(data1[data1['Classes'] == 4.0])
        k1=len(data1[data1['Classes'] == 5.0])
        l1=len(data1[data1['Classes'] == 6.0])
        m1=len(data1[data1['Classes'] == 7.0])
        n1=len(data1[data1['Classes'] == 8.0])
        o1=len(data1[data1['Classes'] == 9.0])
        x1 = ['total_objects','sign board','straight','left','right','side view car','side view bike','left u turn','straight right','straight left']
        y1 = [f1,g1,h1,i1,j1,k1,l1,m1,n1,o1]

        fig, ax = plt.subplots()    
        width = 0.75 # the width of the bars 
        ind1 = np.arange(len(y1))  # the x locations for the groups
        ax.barh(ind1, y1, width,color="red")
        ax.set_yticks(ind1+width/2)
        ax.set_yticklabels(x1, minor=False)
        for i, v in enumerate(y1):
            ax.text(v + 3, i + .25, str(v), color='grey', fontweight='bold')
        plt.title('Detected object analysis of cross traffic')
        plt.xlabel('count of objects')
        plt.ylabel('objects')      
        plt.show()

    def graph3(self):
        main_list=[(2.0,5.0), (2.0, 6.0), (2.0, 7.0),(2.0,8.0), (2.0, 9.0), (5.0, 2.0),(6.0,2.0), (7.0, 2.0), (8.0, 2.0),
        (9.0,2.0),(3.0,5.0) ,(3.0, 6.0), (3.0, 8.0),(5.0,3.0), (6.0, 3.0), (8.0, 3.0),(4.0,5.0), (4.0, 6.0), (4.0, 7.0),(4.0,9.0),
        (5.0,4.0), (6.0, 4.0), (7.0, 4.0),(9.0,4.0), (5.0, 6.0), (5.0, 7.0),(5.0,8.0), (5.0, 9.0), (6.0, 5.0),(7.0, 5.0),
        (8.0,5.0), (9.0, 5.0), (6.0, 7.0),(6.0,8.0), (6.0, 9.0), (7.0, 6.0),(8.0,6.0), (9.0, 6.0), (7.0, 8.0),(8.0,7.0),
        (8.0, 9.0),(9.0,8.0)]
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        list5=[]
        a=[]
        b=[]
        d=[]
        c=0
        resultList=[]
        max_frame_no=[]
        curr_frame_no=1
        final_frame_list=[]
        #reading a csv
        data=pd.read_csv("abc.csv")
        A=70
        #selecting frames with detected object accuracy >70
        score_70=data.loc[(data['Score'] >= A)]
        score_70_len=len(score_70)
        #selecting the frames with more than one object detected
        dataframe= score_70.loc[score_70.No_of_objects!= 1]
        #max no of the frame
        max_frame_no=dataframe['A_Frame_no'].max()

        while(curr_frame_no!=max_frame_no):
            #select all the rows of a particular frame no
            frame=dataframe.loc[(dataframe['A_Frame_no']==curr_frame_no)]
            count_row = frame.shape[0]
            if(count_row>1):
                #creating the combination of the objects detected
                list1 = list(itertools.combinations(frame['Classes'], 2))
                list2 = np.unique(list1, axis=0)
                for i in list2:
                    list3.append(tuple(i))
                list4=(main_list+list3)
                a=len(list4)
                list5=np.unique(list4,axis=0)
                b=len(list5)
                if(a!=b):
                    c=frame['A_Frame_no'].unique()      
                    d=c[0].astype(np.int64)
                    final_frame_list.append(d)
                    curr_frame_no=curr_frame_no+1
                    del list1[:]
                    del list3[:]
                    del list4[:]
                else:
                    curr_frame_no=curr_frame_no+1
            else:
                curr_frame_no=curr_frame_no+1



        #3's combination for cross traffic
        main_list1=[(2.0,3.0,4.0),(2.0,4.0,3.0),(3.0,2.0,4.0),(3.0,4.0,2.0),(4.0,3.0,2.0),(4.0,2.0,3.0),(2.0,3.0,5.0),(2.0,5.0,3.0),(3.0,2.0,5.0),(3.0,5.0,2.0),(5.0,3.0,2.0),(5.0,2.0,3.0),
        (2.0,3.0,6.0),(2.0,6.0,3.0),(3.0,2.0,6.0),(3.0,6.0,2.0),(6.0,3.0,2.0),(6.0,2.0,3.0),(2.0,3.0,8.0),(2.0,8.0,3.0),(3.0,2.0,8.0),(3.0,8.0,2.0),(8.0,3.0,2.0),(8.0,2.0,3.0),
        (2.0,4.0,5.0),(2.0,5.0,4.0),(4.0,2.0,5.0),(4.0,5.0,2.0),(5.0,4.0,2.0),(5.0,2.0,4.0),(2.0,4.0,6.0),(2.0,6.0,4.0),(4.0,2.0,6.0),(4.0,6.0,2.0),(6.0,4.0,2.0),(6.0,2.0,4.0),
        (2.0,4.0,7.0),(2.0,7.0,4.0),(4.0,2.0,7.0),(4.0,7.0,2.0),(7.0,4.0,2.0),(7.0,2.0,4.0),(2.0,4.0,9.0),(2.0,9.0,4.0),(4.0,2.0,9.0),(4.0,9.0,2.0),(9.0,4.0,2.0),(9.0,2.0,4.0),
        (5.0,6.0,7.0),(5.0,7.0,6.0),(6.0,5.0,7.0),(6.0,7.0,5.0),(7.0,5.0,6.0),(7.0,6.0,5.0),(5.0,6.0,8.0),(5.0,8.0,6.0),(6.0,5.0,8.0),(6.0,8.0,5.0),(8.0,5.0,6.0),(8.0,6.0,5.0),
        (5.0,6.0,9.0),(5.0,9.0,6.0),(6.0,5.0,9.0),(6.0,9.0,5.0),(9.0,5.0,6.0),(9.0,6.0,5.0),(6.0,7.0,8.0),(6.0,8.0,7.0),(7.0,6.0,8.0),(7.0,8.0,6.0),(8.0,6.0,7.0),(8.0,7.0,6.0),
        (6.0,7.0,9.0),(6.0,9.0,7.0),(7.0,6.0,9.0),(7.0,9.0,6.0),(9.0,6.0,7.0),(9.0,7.0,6.0),(7.0,8.0,9.0),(7.0,9.0,8.0),(8.0,7.0,9.0),(8.0,9.0,7.0),(9.0,8.0,7.0),(9.0,7.0,8.0),
        (5.0,5.0,3.0),(5.0,3.0,5.0),(3.0,5.0,5.0),(5.0,5.0,4.0),(5.0,4.0,5.0),(4.0,5.0,5.0),(5.0,5.0,7.0),(5.0,7.0,5.0),(7.0,5.0,5.0),(5.0,5.0,8.0),(5.0,8.0,5.0),(8.0,5.0,5.0),
        (5.0,5.0,9.0),(5.0,9.0,5.0),(9.0,5.0,5.0),(6.0,6.0,3.0),(6.0,3.0,6.0),(3.0,6.0,6.0),(6.0,6.0,4.0),(6.0,4.0,6.0),(4.0,6.0,6.0),(6.0,6.0,7.0),(6.0,7.0,6.0),(7.0,6.0,6.0),
        (6.0,6.0,8.0),(6.0,8.0,6.0),(8.0,6.0,6.0),(6.0,6.0,9.0),(6.0,9.0,6.0),(9.0,6.0,6.0),(2.0,2.0,5.0),(2.0,5.0,2.0),(5.0,2.0,2.0)]
        list11=[]
        list21=[]
        list31=[]
        list41=[]
        list51=[]
        a1=[]
        b1=[]
        d1=[]
        c1=0
        final_frame_list1=[]
        curr_frame_no1=1




        while(curr_frame_no!=max_frame_no):
            #select all the rows of a particular frame no
            frame=dataframe.loc[(dataframe['A_Frame_no']==curr_frame_no1)]
            count_row = frame.shape[0]
            if(count_row>1):
                #creating the combination of the objects detected
                list11 = list(itertools.combinations(frame['Classes'], 2))
                list21 = np.unique(list11, axis=0)
                for i in list21:
                    list31.append(tuple(i))
                list4=(main_list1+list31)
                a1=len(list41)
                list51=np.unique(list41,axis=0)
                b1=len(list51)
                if(a!=b):
                    c1=frame['A_Frame_no'].unique()      
                    d1=c1[0].astype(np.int64)
                    final_frame_list1.append(d1)
                    curr_frame_no1=curr_frame_no1+1
                    del list11[:]
                    del list31[:]
                    del list41[:]
                else:
                    curr_frame_no1=curr_frame_no1+1
            else:
                curr_frame_no1=curr_frame_no1+1

        resultList=sorted(np.unique(final_frame_list+final_frame_list1))
        resultList_len=[]
        resultList_len=len(resultList)
        ba=((resultList_len*100)/max_frame_no)
        aa=(100-ba)
        values = [max_frame_no,resultList_len]
        size=[aa,ba]
        colors = ['red', 'grey']
        labels = ['total_frames','cross_traffic_frames']
        explode = (0.2, 0)
        plt.pie(size, colors=colors,autopct='%1.1f%%', labels= values,explode=explode,counterclock=False, shadow=True)
        plt.title('frames analysis')
        plt.legend(labels,loc="upper left")
        plt.axis('equal')
        plt.show()


    def annotate(self):
        abc1 = self.gettext()
        CWD_PATH = os.getcwd()
        MODEL_NAME = 'inference_graph'
        VIDEO_NAME = (str(abc1)+'.avi')
        targetdir= 'out'

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
            df = pd.DataFrame(data={"A_Frame_no": curr_frame_no, "Score": i ,"Classes": i1 , "No_of_objects" : no1 , "Boxes" :composite_list })
            #appending df to list
            a.append(df)
            #concating the list
            masterDF = pd.concat(a, ignore_index=True)
            masterDF.to_csv("abc2.csv", sep=',',index=False)
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




    def generate_video(self):
        main_list=[(2.0,5.0), (2.0, 6.0), (2.0, 7.0),(2.0,8.0), (2.0, 9.0), (5.0, 2.0),(6.0,2.0), (7.0, 2.0), (8.0, 2.0),
        (9.0,2.0),(3.0,5.0) ,(3.0, 6.0), (3.0, 8.0),(5.0,3.0), (6.0, 3.0), (8.0, 3.0),(4.0,5.0), (4.0, 6.0), (4.0, 7.0),(4.0,9.0),
        (5.0,4.0), (6.0, 4.0), (7.0, 4.0),(9.0,4.0), (5.0, 6.0), (5.0, 7.0),(5.0,8.0), (5.0, 9.0), (6.0, 5.0),(7.0, 5.0),
        (8.0,5.0), (9.0, 5.0), (6.0, 7.0),(6.0,8.0), (6.0, 9.0), (7.0, 6.0),(8.0,6.0), (9.0, 6.0), (7.0, 8.0),(8.0,7.0),
        (8.0, 9.0),(9.0,8.0)]
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        list5=[]
        a=[]
        b=[]
        d=[]
        c=0
        resultList=[]
        max_frame_no=[]
        curr_frame_no=1
        final_frame_list=[]
        #reading a csv
        data=pd.read_csv("abc.csv")
        A=80
        #selecting frames with detected object accuracy >70
        score_70=data.loc[(data['Score'] >= A)]
        score_70_len=len(score_70)
        #selecting the frames with more than one object detected
        dataframe= score_70.loc[score_70.No_of_objects!= 1]
        #max no of the frame
        max_frame_no=dataframe['A_Frame_no'].max()

        while(curr_frame_no!=max_frame_no):
            #select all the rows of a particular frame no
            frame=dataframe.loc[(dataframe['A_Frame_no']==curr_frame_no)]
            count_row = frame.shape[0]
            if(count_row>1):
                #creating the combination of the objects detected
                list1 = list(itertools.combinations(frame['Classes'], 2))
                list2 = np.unique(list1, axis=0)
                for i in list2:
                    list3.append(tuple(i))
                list4=(main_list+list3)
                a=len(list4)
                list5=np.unique(list4,axis=0)
                b=len(list5)
                if(a!=b):
                    c=frame['A_Frame_no'].unique()      
                    d=c[0].astype(np.int64)
                    final_frame_list.append(d)
                    curr_frame_no=curr_frame_no+1
                    del list1[:]
                    del list3[:]
                    del list4[:]
                else:
                    curr_frame_no=curr_frame_no+1
            else:
                curr_frame_no=curr_frame_no+1



        #3's combination for cross traffic
        main_list1=[(2.0,3.0,4.0),(2.0,4.0,3.0),(3.0,2.0,4.0),(3.0,4.0,2.0),(4.0,3.0,2.0),(4.0,2.0,3.0),(2.0,3.0,5.0),(2.0,5.0,3.0),(3.0,2.0,5.0),(3.0,5.0,2.0),(5.0,3.0,2.0),(5.0,2.0,3.0),
        (2.0,3.0,6.0),(2.0,6.0,3.0),(3.0,2.0,6.0),(3.0,6.0,2.0),(6.0,3.0,2.0),(6.0,2.0,3.0),(2.0,3.0,8.0),(2.0,8.0,3.0),(3.0,2.0,8.0),(3.0,8.0,2.0),(8.0,3.0,2.0),(8.0,2.0,3.0),
        (2.0,4.0,5.0),(2.0,5.0,4.0),(4.0,2.0,5.0),(4.0,5.0,2.0),(5.0,4.0,2.0),(5.0,2.0,4.0),(2.0,4.0,6.0),(2.0,6.0,4.0),(4.0,2.0,6.0),(4.0,6.0,2.0),(6.0,4.0,2.0),(6.0,2.0,4.0),
        (2.0,4.0,7.0),(2.0,7.0,4.0),(4.0,2.0,7.0),(4.0,7.0,2.0),(7.0,4.0,2.0),(7.0,2.0,4.0),(2.0,4.0,9.0),(2.0,9.0,4.0),(4.0,2.0,9.0),(4.0,9.0,2.0),(9.0,4.0,2.0),(9.0,2.0,4.0),
        (5.0,6.0,7.0),(5.0,7.0,6.0),(6.0,5.0,7.0),(6.0,7.0,5.0),(7.0,5.0,6.0),(7.0,6.0,5.0),(5.0,6.0,8.0),(5.0,8.0,6.0),(6.0,5.0,8.0),(6.0,8.0,5.0),(8.0,5.0,6.0),(8.0,6.0,5.0),
        (5.0,6.0,9.0),(5.0,9.0,6.0),(6.0,5.0,9.0),(6.0,9.0,5.0),(9.0,5.0,6.0),(9.0,6.0,5.0),(6.0,7.0,8.0),(6.0,8.0,7.0),(7.0,6.0,8.0),(7.0,8.0,6.0),(8.0,6.0,7.0),(8.0,7.0,6.0),
        (6.0,7.0,9.0),(6.0,9.0,7.0),(7.0,6.0,9.0),(7.0,9.0,6.0),(9.0,6.0,7.0),(9.0,7.0,6.0),(7.0,8.0,9.0),(7.0,9.0,8.0),(8.0,7.0,9.0),(8.0,9.0,7.0),(9.0,8.0,7.0),(9.0,7.0,8.0),
        (5.0,5.0,3.0),(5.0,3.0,5.0),(3.0,5.0,5.0),(5.0,5.0,4.0),(5.0,4.0,5.0),(4.0,5.0,5.0),(5.0,5.0,7.0),(5.0,7.0,5.0),(7.0,5.0,5.0),(5.0,5.0,8.0),(5.0,8.0,5.0),(8.0,5.0,5.0),
        (5.0,5.0,9.0),(5.0,9.0,5.0),(9.0,5.0,5.0),(6.0,6.0,3.0),(6.0,3.0,6.0),(3.0,6.0,6.0),(6.0,6.0,4.0),(6.0,4.0,6.0),(4.0,6.0,6.0),(6.0,6.0,7.0),(6.0,7.0,6.0),(7.0,6.0,6.0),
        (6.0,6.0,8.0),(6.0,8.0,6.0),(8.0,6.0,6.0),(6.0,6.0,9.0),(6.0,9.0,6.0),(9.0,6.0,6.0),(2.0,2.0,5.0),(2.0,5.0,2.0),(5.0,2.0,2.0)]
        list11=[]
        list21=[]
        list31=[]
        list41=[]
        list51=[]
        a1=[]
        b1=[]
        d1=[]
        c1=0
        final_frame_list1=[]
        curr_frame_no1=1




        while(curr_frame_no!=max_frame_no):
            #select all the rows of a particular frame no
            frame=dataframe.loc[(dataframe['A_Frame_no']==curr_frame_no1)]
            count_row = frame.shape[0]
            if(count_row>1):
                #creating the combination of the objects detected
                list11 = list(itertools.combinations(frame['Classes'], 2))
                list21 = np.unique(list11, axis=0)
                for i in list21:
                    list31.append(tuple(i))
                list4=(main_list1+list31)
                a1=len(list41)
                list51=np.unique(list41,axis=0)
                b1=len(list51)
                if(a!=b):
                    c1=frame['A_Frame_no'].unique()      
                    d1=c1[0].astype(np.int64)
                    final_frame_list1.append(d1)
                    curr_frame_no1=curr_frame_no1+1
                    del list11[:]
                    del list31[:]
                    del list41[:]
                else:
                    curr_frame_no1=curr_frame_no1+1
            else:
                curr_frame_no1=curr_frame_no1+1

        resultList=sorted(np.unique(final_frame_list+final_frame_list1))
        resultList_len=[]
        resultList_len=len(resultList)


        frame_array = []
        i=0
        az=resultList[0]
        fps=15
        imageDir='/home/athishm/Documents/college project/annotate1/crosstraffic/research/object_detection/out'
        image_path_list=[]
        valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
        valid_image_extensions = [item.lower() for item in valid_image_extensions]

        for file in os.listdir(imageDir):
            extension = os.path.splitext(file)[1]
            if extension.lower() not in valid_image_extensions:
                continue
            image_path_list.append(os.path.join(imageDir, file))

        for imagePath in image_path_list:
            if i < len(resultList):
                az=resultList[i]
                img1=cv2.imread('/home/athishm/Documents/college project/annotate1/crosstraffic/research/object_detection/out/'+str(az)+'.jpg')
                #img1=cv2.imread(str(az)+'.jpg')
                frame_array.append(img1)
                height , width , layers =  img1.shape
                size=(width,height)
                i += 1
                print(i)
                video = cv2.VideoWriter('video11.avi', cv2.VideoWriter_fourcc(*'DIVX'),fps,size)
                for i in range(len(frame_array)):
                    video.write(frame_array[i])

        cv2.destroyAllWindows()
        video.release()




    

    def close_application(self):
        choice = QtGui.QMessageBox.question(self, 'QUIT',
                                            "Are u sure?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            sys.exit()
        else:
            pass
        
        

    
def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())


run()