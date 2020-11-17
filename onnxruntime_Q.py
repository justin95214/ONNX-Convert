import onnx
import tracemalloc
from onnxruntime.quantization import quantize, QuantizationMode
import onnxruntime
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import time


np.random.seed = 100
# dataset
dataset = cifar10
dataset_name = "cifar10"
(x_train, y_train), (x_test, y_test) = dataset.load_data()
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
batch_size = 512

if len(x_train.shape) == 4:
    img_channels = x_train.shape[3]
else:
    img_channels = 1

input_shape = (img_rows, img_cols, img_channels)
num_classes = len(np.unique(y_train))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255.

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


model = onnx.load("C:/Users/82109/PycharmProjects/Quantization/original_model.onnx")

sess0 = onnxruntime.InferenceSession("C:/Users/82109/PycharmProjects/Quantization/original_model.onnx")

input_name = sess0.get_inputs()[0].name
n = 1000
start = time.time()
pred0_onnx = sess0.run(None,{input_name: x_test[:n].astype(np.float32)})
print("ori_pred : ", (time.time()-start)/n)


snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')


quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
onnx.save(quantized_model, "C:/Users/82109/PycharmProjects/Quantization/orginal_model_test.onnx")


Q_model = onnx.load("C:/Users/82109/PycharmProjects/Quantization/orginal_model_test.onnx")
sess = onnxruntime.InferenceSession("C:/Users/82109/PycharmProjects/Quantization/orginal_model_test.onnx")

input_name = sess.get_inputs()[0].name

start = time.time()
pred_onnx = sess.run(None,{input_name: x_test[:n].astype(np.float32)})
print("Q_pred : ",( time.time()-start)/n)
