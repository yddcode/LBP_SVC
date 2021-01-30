from skimage import feature
from PIL import Image
import numpy as np
import time
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, precision_score
from PIL import Image

def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data

images = []
for fn in os.listdir('D:/data'):
    if fn.endswith('.jpg'):
        fd = os.path.join('D:/data',fn)
        images.append(read_image(fd))
print('load data success!')

X = np.array(images)
print (X.shape)
y = np.loadtxt('label.txt')
print (y.shape)
# the data, split between train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state= 3)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

radius = 2
n_point = radius * 8
def lbp_texture(train_data, test_data):
    train_hist = np.zeros((len(X_train), 256))
    test_hist = np.zeros(((len(X_train), 256))
    for i in np.arange((len(X_train)):
        # 使用skimage LBP方法提取图像的纹理特征
        lbp = feature.local_binary_pattern(train_data[i],n_point,radius,'default')
        # 统计图像直方图256维
        max_bins = int(lbp.max() + 1)
        # hist size:256
        train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))

    for i in np.arange((len(X_train)):
        lbp = feature.local_binary_pattern(test_data[i],n_point,radius,'default')
        max_bins = int(lbp.max() + 1)
        test_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return train_hist, test_hist

from sklearn import svm
X_train, X_test = lbp_texture(X_train, X_test)

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(X_train, y_train)
p_test = clf.predict(X_test)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
print(clf.score(X_train, y_train))  # 训练精度
print(clf.score(X_test, y_test))  # 测试精度
print('decision_function:\n', clf.decision_function(x_train))
print(precision_score(y_test, p_test, average='macro'))
print(recall_score(y_test, p_test, average='macro'))
print(f1_score(y_test, p_test, average='macro'))
print(accuracy_score(y_test, p_test))

