# -*- coding:utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

margin = 3 # 罫線分のピクセル数


for i in range(1):
    img = cv2.imread('./samples/test5-{0:05d}.jpg'.format(i), 0)
    img = img[10:115, 865:915]
    # cv2.imwrite("hoge.png", img)
    # Now we split the image to 3 cells, each 20x20 size
    cells = [cv2.resize(v[margin:-margin,:], (20, 20)) for v in np.vsplit(img,3)]
    # print len(cells)

    # cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
    ### ガウスブラー
    # cells = map(lambda x:cv2.GaussianBlur(x,(5,5),0), cells)
    ### 大津の2値化
    cells = map(lambda x:cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1], cells)
    ### 白黒反転
    cells = map(lambda x:cv2.bitwise_not(x), cells)
    cv2.imwrite('hoge0.png', cells[0])
    cv2.imwrite('hoge1.png', cells[1])
    cv2.imwrite('hoge2.png', cells[2])
quit()

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare test_data.
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# Create labels for train and test data
k = np.arange(10)
test_labels = np.repeat(k,250)[:,np.newaxis]

# Now load the data
with np.load('knn_data.npz') as data:
    print data.files
    train = data['train']
    train_labels = data['train_labels']


# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)


# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy
