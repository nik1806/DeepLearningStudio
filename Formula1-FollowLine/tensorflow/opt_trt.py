# Re-run after Kernel restart
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

if __name__ == '__main__':

    # model = ResNet50()
    # model.save('resnet50_saved_model') 

    # # model_path = 'trained_models/pilotnet.h5'
    model_path = 'resnet50_saved_model'

    print('Converting to TF-TRT FP32...')
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                                max_workspace_size_bytes=8000000000)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir= model_path,
                                        conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir='pilotnet_TFTRT_FP32')
    print('Done Converting to TF-TRT FP32')
    