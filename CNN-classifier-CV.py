
# 導入基本庫
import urllib.request
import os
import zipfile
# ImageDataGenerator 爲圖像前處理庫
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
# InceptionV3 包含事先訓練好的 CNN 模型
from tensorflow.keras.applications.inception_v3 import InceptionV3
# RMSprop = 均方根傳播
from tensorflow.keras.optimizers import RMSprop


weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
# 取得模型文件
urllib.request.urlretrieve(weights_url, weights_file)
# 實例化
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
# 模型文件載入內存
pre_trained_model.load_weights(weights_file)
# 使模型內每層卷積爲不可訓練
for layer in pre_trained_model.layers:
    layer.trainable = False
# 列出摘要
pre_trained_model.summary()

# 裁切卷積網絡到 mixed7 這一層
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# 平坦化裁切的 CNN
x = layers.Flatten()(last_output)
# CNN 後面增加一層隱藏的 DNN，帶有1024個神經元，激活函數爲校正線性單元（RELU） 
x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.2)(x)

# 最後的輸出層只有一個神經元
x = layers.Dense(1, activation='sigmoid')(x)
# CNN 與 DNN 連接起來
model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

# 下載、解壓訓練與驗證數據集到指定目錄
training_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
training_file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(training_url, training_file_name)
zip_ref = zipfile.ZipFile(training_file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(validation_url, validation_file_name)

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()


# 通過 ImageDataGenerator 執行訓練數據集的前處理：圖像的歸一化和增強
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# 歸一化驗證數據集
test_datagen = ImageDataGenerator(rescale=1.0/255.)

# 按批次流入訓練圖片，每批 20 張
train_generator = train_datagen.flow_from_directory(training_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode='binary',
                                                         target_size=(150, 150))
# 前面 2 個生成器分別作爲模型的訓練和驗證數據集
history = model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=20,
            verbose=1)

# 啓動模型前導入依賴庫
import numpy as np
from google.colab import files
from keras.preprocessing import image

# 此調用阻塞到用戶上傳測試圖片爲止
uploaded = files.upload()

for fn in uploaded.keys():
 
  # 將圖像轉換爲我們的 CNN+DNN 能夠處理的格式
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  # 增加一個維度
  x = np.expand_dims(x, axis=0)
  # 垂直堆疊
  image_tensor = np.vstack([x])
  
  # 預測未知圖片的分類
  classes = model.predict(image_tensor)
  print(classes)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
    
   
