# Machine-Learning-NTUT-2018-The-Simpsons-Characters-Recognition-Challenge
第二次作業

# coding: utf-8

# In[13]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import h5py
import glob
import time
from random import shuffle
from collections import Counter

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam


# In[14]:


# 卡通角色的Label-encoding
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', 11: 'lisa_simpson', 
        12: 'marge_simpson', 13:  'mayor_quimby', 14: 'milhouse_van_houten', 15: 'moe_szyslak', 
        16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}

img_width = 42 
img_height = 42


num_classes = len(map_characters) # 要辨識的角色種類

pictures_per_class = 1000 # 每個角色會有接近1000張訓練圖像
test_size = 0.15

imgsPath = "C:/Users/Shirley/Desktop/data/train"


# In[15]:


# 將訓練資料圖像從檔案系統中取出並進行
def load_pictures():
    pics = []
    labels = []
    
    for k, v in map_characters.items(): # k: 數字編碼 v: 角色label
        # 把某一個角色在檔案夾裡的所有圖像檔的路徑捉出來
        pictures = [k for k in glob.glob(imgsPath + "/" + v + "/*")]        
        print(v + " : " + str(len(pictures))) # 看一下每個角色有多少訓練圖像

        for i, pic in enumerate(pictures):
            tmp_img = cv2.imread(pic)
            
            # 由於OpenCv讀圖像時是以BGR (Blue-Green-Red), 我們把它轉置成RGB (Red-Green-Blue)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
            tmp_img = cv2.resize(tmp_img, (img_height, img_width)) # 進行大小歸一位            
            pics.append(tmp_img)
            labels.append(k)
    return np.array(pics), np.array(labels)

# 取得訓練資料集與驗證資料集
def get_dataset(save=False, load=False):
    if load: 
        # 從檔案系統中載入之前處理保存的訓練資料集與驗證資料集
        h5f = h5py.File('dataset.h5','r')
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        h5f.close()
        
        # 從檔案系統中載入之前處理保存的訓練資料標籤與驗證資料集籤
        h5f = h5py.File('labels.h5', 'r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
    else:
        # 從最原始的圖像檔案開始處理
        X, y = load_pictures()
        y = keras.utils.to_categorical(y, num_classes) # 目標的類別種類數
        
        # 將資料切分為訓練資料集與驗證資料集 (85% vs. 15%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) 
        if save: # 保存尚未進行歸一化的圖像數據
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()
            
            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()
    
    # 進行圖像每個像素值的型別轉換與歸一化處理
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test  


# In[16]:


# 取得訓練資料集與驗證資料集  
X_train, X_test, y_train, y_test = get_dataset(save=True, load=False)


# In[17]:


def create_model_six_conv(input_shape):
    model = Sequential()
    model.add(Conv2D(filters =128, kernel_size = 3, padding='same', activation='relu', input_shape=input_shape))
    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu'))
    model.add(Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu'))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu'))
    model.add(Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu'))
    model.add(Conv2D(filters = 256, kernel_size = 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer = 'normal'))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Activation('softmax'))
    
    #model.add(Dense(num_classes, activation='softmax'))

    return model;

#圖像的shape是 (42,42,3)
model = create_model_six_conv((img_height, img_width, 3)) # 初始化一個模型
model.summary() # 秀出模型架構


# In[18]:


# 讓我們先配置一個常用的組合來作為後續優化的基準點
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])


# In[20]:


def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))

batch_size = 200
epochs = 100

history = model.fit(X_train, y_train,
         batch_size=batch_size,
         epochs=epochs,
         validation_data=(X_test, y_test),
         shuffle=True,
         callbacks=[LearningRateScheduler(lr_schedule),
             ModelCheckpoint('model.h5', save_best_only=True)
         ])


# In[21]:


# 透過趨勢圖來觀察訓練與驗證的走向 (特別去觀察是否有"過擬合(overfitting)"的現象)
import matplotlib.pyplot as plt

def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics),'-o')
    plt.plot(history.history.get(val_metrics),'-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])
    
    
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plot_train_history(history, 'loss','val_loss')

plt.subplot(1,2,2)
plot_train_history(history, 'acc','val_acc')

plt.show()


# In[22]:


import os
from pathlib import PurePath # 處理不同作業系統file path的解析問題 (*nix vs windows)

# 載入要驗證模型的數據
def load_test_set(path):
    pics = []
    for i in range(990):
        temp = cv2.imread(path +'/' + str(i+1) + '.jpg')
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = cv2.resize(temp, (img_height,img_width)).astype('float32') / 255.
        pics.append(temp)
    
    X_test = np.array(pics)
    print("Test set", X_test.shape)
    return X_test

imgsPath = "C:/Users/Shirley/Desktop/data/test"

#載入數據
X_valtest = load_test_set(imgsPath)


# In[23]:


# 預測與比對
from keras.models import load_model

# 把訓練時val_loss最小的模型載入
model = load_model('model.h5')

# 預測與比對
y_pred = model.predict_classes(X_valtest)

for y in y_pred:
    y = map_characters[y]
    print(y)
    
print(y_pred)
np.savetxt('C:/Users/Shirley/Desktop/data/testY.csv', y_pred, delimiter = ',')




