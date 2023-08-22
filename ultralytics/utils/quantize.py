import tensorflow as tf
import os 
import cv2
import numpy as np 
import sys

'''
python3 quantize.py <path/to/saved_model> <path/to/dataset>
'''

def representative_dataset_gen():
    folder_path = sys.argv[-1]
    image_files = os.listdir(folder_path)
    for image_file in image_files:
        image = cv2.imread(folder_path +"/"+image_file)
        resized_image = cv2.resize(image, (image.shape[0], image.shape[1]))
        resized_image = resized_image.astype(np.float32) / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)
        image_tensor = tf.convert_to_tensor(resized_image)
        yield [image_tensor]

converter = tf.lite.TFLiteConverter.from_saved_model(sys.argv[-2],)

# Full Integer Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_disable_per_channel = True
converter._experimental_disable_batchmatmul_unfold = not False
converter.representative_dataset = representative_dataset_gen

inf_type = tf.uint8
converter.inference_input_type = inf_type
converter.inference_output_type = inf_type
tflite_model = converter.convert()

with open(sys.argv[-2], 'wb') as w:
    w.write(tflite_model)