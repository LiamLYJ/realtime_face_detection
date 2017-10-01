import numpy as np

def IOU(Re,GT):
    x1 = Re[0];
    y1 = Re[1];
    width1 = Re[2]-Re[0];
    height1 = Re[3]-Re[1];

    x2 = GT[0];
    y2 = GT[1];
    width2 = GT[2]-GT[0];
    height2 = GT[3]-GT[1];

    endx = max(x1+width1,x2+width2);
    startx = min(x1,x2);
    width = width1+width2-(endx-startx);

    endy = max(y1+height1,y2+height2);
    starty = min(y1,y2);
    height = height1+height2-(endy-starty);

    if width <=0 or height <= 0:
        ratio = 0
    else:
        Area = width*height;
        Area1 = width1*height1;
        Area2 = width2*height2;
        ratio = Area*1./(Area1+Area2-Area);
    return ratio

# Re = [1,1,4,4]
# GT = [3,2,5,5]
# ratio = IOU(Re,GT)
# print (ratio)


def evaluate(pre,gt,threshold):
    count_precision = 0
    for i in range(pre.shape[0]):
        for j in range(gt.shape[0]):
            if IOU(pre[i],gt[j]) > threshold:
                count_precision += 1
                break
    count_recall = 0
    for i in range(gt.shape[0]):
        for j in range(pre.shape[0]):
            if IOU(gt[i],pre[j]) > threshold:
                count_recall += 1
                break
    return count_precision, count_recall

# pre = np.arange(40).reshape((10,4))
# gt = np.arange(40).reshape((10,4))
# precision,recall = evaluate(pre,gt,0.999)
# print ('precision:', precision)
# print ('recall:', recall)
