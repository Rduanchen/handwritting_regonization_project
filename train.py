# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 17:23:52 2022

@author: cheny
"""

import numpy as np
import matplotlib.pyplot as plt
import glob,cv2

from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

####################顯示資料結果#################################
def show_images_labels_predictions(images,labels,predictions,start_id,num=20):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, i+1)
        ax.imshow(images[start_id], cmap='binary')  #顯示黑白圖片
        if( len(predictions) > 0 ) :  #有傳入預測資料
            title = 'ai = ' + str(predictions[start_id])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)') 
            title += '\nlabel = ' + str(labels[start_id])
        else :  #沒有傳入預測資料
            title = 'label = ' + str(labels[start_id])
        ax.set_title(title,fontsize=12)  #X,Y軸不顯示刻度
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()
    
    
#####################將測試片轉成minist 格式############################
#-----讀取圖片，修正格式
testfiles = glob.glob("Handwritting_for_test\*.jpg") 
 #建立測試資料(you can change it to your fold name but please left'\*.jpg')
test_feature=[]
test_label=[]
for file in testfiles:
    img=cv2.imread(file)
    img = cv2.resize(img, (28, 28))
    cv2.imwrite(file, img)
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰階    
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) #轉為反相黑白 
    test_feature.append(img) #將心資料傳入陣列
    label=file[22:23]  # 取出照片解答
    test_label.append(label)
    

#-----進行資料格式轉換
test_feature=np.array(test_feature) # 串列轉為矩陣 
test_label=np.array(test_label)     # 串列轉為矩陣
test_feature_vector = test_feature.reshape(len(test_feature), 784).astype('float32')
 #將資料陣列攤平成總資料量*784的二維陣列
test_feature_normalize = test_feature_vector/255 #資料歸一化

###############將訓練資料轉為mnist格式####################
#-----讀取圖片，修正格式
files = glob.glob("Handwritting_for_train\*.jpg")  #建立訓練資料
train_feature=[]
train_label=[]
for file in files:
    img=cv2.imread(file)
    img = cv2.resize(img, (28, 28))
    cv2.imwrite(file, img)
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰階    
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) #轉為反相黑白 
    train_feature.append(img)
    label=file[23:24]  #"imagedata\1.jpg"第10個字元1為label
    train_label.append(label)
#-----進行資料格式轉換
#print(train_label)
train_feature=np.array(train_feature) # 串列轉為矩陣 
train_label=np.array(train_label)     # 串列轉為矩陣
train_feature_vector = train_feature.reshape(len(train_feature), 784).astype('float32')
 #將資料陣列攤平成總資料量*784的二維陣列
train_feature_normalize = train_feature_vector/255 #資料歸一化


#######################開始訓練##################################
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)
print(train_label_onehot[350:400])
print(test_label_onehot)

model = Sequential()  #建立模型
model.add(Dense(units=256,  #輸入層：784, 隱藏層：256
                input_dim=784, 
                kernel_initializer='normal', 
                activation='relu')) 
model.add(Dense(units=3,  #輸出層：10
                kernel_initializer='normal', 
                activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
model.fit(x=train_feature_normalize, y=train_label_onehot,
          validation_split=0.2, epochs=20, batch_size=20,verbose=2)
scores = model.evaluate(test_feature_normalize, test_label_onehot)  #評估準確率
print('\n準確率=',scores[1])
model.save('Moduel_for_image.h5')
print("Moduel_for_image.h5 模型儲存完畢!")

