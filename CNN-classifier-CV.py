
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

