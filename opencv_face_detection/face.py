import numpy as np
import sys
sys.path.append("..")
import cv2
import re
from evaluation import evaluate
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
file_path = '../eval_file.txt'
image_folder = '../WIDER_train/images'
image_path = []
GT = []

with open(file_path) as f:
    count = 0
    for line in f:
        if (re.search('_',line)) :
            # erase the '\n' in line
            image_path.append(os.path.join(image_folder,line[:-1]))
            count = 0
        elif len(line.split(' ')) < 2 :
            num = int(line)
            boxes = np.ones((num,4))
        else :
            tmp = line.split(' ')
            boxes[count,0] = int(tmp[0])
            boxes[count,1] = int(tmp[1])
            boxes[count,2] = int(tmp[0]) + int(tmp[2])
            boxes[count,3] = int(tmp[1]) + int(tmp[3])
            count += 1
            if count == num :
                GT.append(boxes)

sum_pre = 0
sum_gt = 0
sum_precision = 0
sum_recall = 0
threshold = 0.1
for i,j in enumerate(image_path):
    img = cv2.imread(str(image_path[i]))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pre_ = np.array(face_cascade.detectMultiScale(gray, 1.3, 5))
    if pre_.shape[0] > 0:
        pre = pre_[:]
        pre[:,2] = pre_[:,0] + pre_[:,2]
        pre[:,3] = pre_[:,1] + pre_[:,3]
    gt = np.array(GT[i])
    count_precision, count_recall = evaluate(pre,gt,threshold = threshold)
    sum_precision += count_precision
    sum_recall += count_recall
    sum_gt += gt.shape[0]
    sum_pre += pre.shape[0]

print ('the precision is :', float(sum_precision) / sum_pre)
print ('the recall is :', float(sum_recall)/ sum_gt)


raise

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('test_3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print (faces)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
