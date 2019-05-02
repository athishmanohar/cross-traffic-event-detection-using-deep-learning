import pandas as pd
import numpy as np
import itertools
import cv2
import os
import cv2
import matplotlib.pyplot as plt
#2's combination of cross traffic
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




while(curr_frame_no1!=max_frame_no):
  frame=dataframe.loc[(dataframe['A_Frame_no']==curr_frame_no1)]
  count_row = frame.shape[0]
  if(count_row>2):
    list11 = list(itertools.combinations(frame['Classes'], 3))
    list21 = np.unique(list11, axis=0)
    for i in list21:
      list31.append(tuple(i))
    list41=(main_list1+list31)
    a1=len(list41)
    list51=np.unique(list41,axis=0)
    b1=len(list51)
    if(a1!=b1):
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

 #create a video from all the cross traffic frames
frame_array = []
i=0
az=resultList[0]
fps=10
img2=cv2.imread(str(a)+'.jpg')
height , width , layers =  img2.shape
print(height,width,layers)
while i < len(resultList):
    az=resultList[i]
    img1=cv2.imread(str(az)+'.jpg')
    frame_array.append(img1)
    height , width , layers =  img1.shape
    size=(width,height)
    i += 1
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'),fps,size)
    for i in range(len(frame_array)):
        video.write(frame_array[i])


cv2.destroyAllWindows()
video.release()

"""#pie chart for no of objects with accuracy morethan 70
cz=len(data.index)
dz=(score_70_len*100)/cz
ez=(100-dz)
values = [cz,score_70_len]
size=[ez,dz]
colors = ['red', 'grey']
labels = ['total_no_of_objects_detected','objects_with_predection>70']
explode = (0.1, 0)
plt.pie(size, colors=colors,autopct='%1.1f%%', labels= values,explode=explode,counterclock=False, shadow=True)
plt.title('objects detection accuracy analysis')
plt.legend(labels,loc="upper left")
plt.axis('equal')
plt.show()


#detcted object analysis
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



#dected object analysis of the cross traffic frames
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


#cross traffic frame analysis
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
"""

