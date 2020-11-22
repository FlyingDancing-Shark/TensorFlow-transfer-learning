
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

