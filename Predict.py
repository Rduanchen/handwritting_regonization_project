# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 23:13:49 2022

@author: cheny
"""
#------函式庫引用區----------
import numpy as np  #儲存圖片的陣列格式
import matplotlib.pyplot as plt #繪製辨識結果的工具
from keras.models import load_model #送資料進辨識模型用的工具
import glob,cv2  #處理圖片用的函示庫

#---------------自定義顯示結果的函數-----------------#
def show_images_labels_predictions(images,labels,predictions,start_id,num=20): 
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, i+1) #設定plot 視窗的大小
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
    plt.show()  #將結果顯示於電腦上

#-------------建立圖片資料---------------

test_img_path=str(input("please enter the folder name\n"))+'/*.jpg'
file_len=int(len(test_img_path))
file_len-=6
moduel_name=str(input("please enter the moduel name\n"))


files = glob.glob(test_img_path)  #取出測試資料()
test_feature=[]  #建立儲存圖片的陣列(list)
test_label=[]    #建立儲存圖片正確答案的陣列(list)

for file in files: #讀取每一個在資料夾中的檔案
    img=cv2.imread(file)  
    img = cv2.resize(img, (28, 28))  #將圖片變更成28*28的圖片
    cv2.imwrite(file, img)   
    img=cv2.imread(file) 
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #將圖片轉為灰階圖片
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) #轉為反相黑白 
    test_feature.append(img)  #將圖片陣列儲存至test_feature
    label=file[file_len+1:file_len+2]  #"imagedata\1.jpg"第10個字元1為label
    test_label.append(int(label)) #將圖片答案加入陣列test_label中

#-------圖片資料愈處理------------
test_feature=np.array(test_feature) # 串列轉為numpy矩陣 
test_label=np.array(test_label)     # 串列轉為numpy矩陣
test_feature_vector = test_feature.reshape(len( test_feature), 784).astype('float32')# 將圖片設為矩陣
test_feature_normalize = test_feature_vector/255 #將圖片資料歸一化
model = load_model(moduel_name) #取出以訓練好的模型

#--------- 將圖片資料送進模型比對----------------
predictions=model.predict(test_feature_normalize) #將資料送進去模型中做比對，並將結果以陣列的形式回傳
predictions=np.argmax(predictions,axis=1) #從回傳的陣列中取出最的大的值的索引值(最有可能的數字)
show_images_labels_predictions(test_feature,test_label,predictions,0) #將結果傳入結果顯示函數