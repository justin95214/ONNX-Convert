from tensorflow.python.keras import backend as K
#from tensorflow.keras.models import load_model
import os
os.environ['TF_KERAS'] = '1'

import onnx
import keras2onnx
import numpy as np
import onnxruntime
import sys
from tensorflow.keras.models import load_model
import onnxmltools
import onnx2keras
from onnx2keras import onnx_to_keras
onnx_model_name = 'C:/Users/82109/PycharmProjects/Quantization/original_model.onnx'
#img_path = 'D:/test/test9.h5'
img_path = 'C:/Users/82109/PycharmProjects/Quantization/test_keras.h5'
model = load_model(img_path)

onnx_model = keras2onnx.convert_keras(model, model.name,debug_mode=1)
onnx.save_model(onnx_model, onnx_model_name)

x = np.load('D:/test/X5.npy')
print(x)
print(x.shape)

onnx_model = onnxmltools.utils.load_model('C:/Users/82109/PycharmProjects/untitled/original_model.onnx')

endpoint_names = []
"""
for i in range(len(onnx_model.graph.node)):
    for j in range(len(onnx_model.graph.node[i].input)):
        if onnx_model.graph.node[i].input[j] not in endpoint_names:
            # print(onnx_model.graph.node[i].name)
            # print(onnx_model.graph.node[i].input)
            # print(onnx_model.graph.node[i].output)

            splitted = onnx_model.graph.node[i].input[j].split(':')
            if len(splitted) > 1:
                onnx_model.graph.node[i].input[j] = splitted[0] + "_" + splitted[1]
            else:
                onnx_model.graph.node[i].input[j] = splitted[0]

            # onnx_model.graph.node[i].input[j] = "input" + str(inputcount)
            # onnx_model.graph.node[i].name = "name" + str(namecount)

    for j in range(len(onnx_model.graph.node[i].output)):
        if onnx_model.graph.node[i].output[j] not in endpoint_names:
            # print(onnx_model.graph.node[i].name)
            # print(onnx_model.graph.node[i].input)
            # print(onnx_model.graph.node[i].output)

            splitted = onnx_model.graph.node[i].output[j].split(':')
            if len(splitted) > 1:
                onnx_model.graph.node[i].output[j] = splitted[0] + "_" + splitted[1]
            else:
                onnx_model.graph.node[i].output[j] = splitted[0]

            # onnx_model.graph.node[i].output[j] = "output" + str(outputcount)
            # onnx_model.graph.node[i].name = "name" + str(namecount)
            # namecount += 1

for i in range(len(onnx_model.graph.input)):
    if onnx_model.graph.input[i].name not in endpoint_names:
        # print(onnx_model.graph.input[i])

        splitted = onnx_model.graph.input[i].name.split(':')
        if len(splitted) > 1:
            onnx_model.graph.input[i].name = splitted[0] + "_" #+ splitted[1]
        else:
            onnx_model.graph.input[i].name = splitted[0]

        # onnx_model.graph.input[i].name = "name" + str(namecount)
        # namecount += 1

for i in range(len(onnx_model.graph.output)):
    if onnx_model.graph.output[i].name not in endpoint_names:
        # print(onnx_model.graph.output[i])

        splitted = onnx_model.graph.output[i].name.split(':')
        if len(splitted) > 1:
            onnx_model.graph.output[i].name = splitted[0] + "_" #+ splitted[1]
        else:
            onnx_model.graph.output[i].name = splitted[0]
        # onnx_model.graph.output[i].name = "name" + str(namecount)
        # namecount += 1

k_model = onnx2keras.onnx_to_keras(onnx_model,["input"])
"""
"""
# runtime prediction
#content = onnx_net.SerializeToString()
#sess = onnxruntime.InferenceSession(content)
sess = onnxruntime.InferenceSession(onnx_model)
x = x if isinstance(x, list) else [x]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)[0]
pred = np.squeeze(pred_onnx)
top_inds = pred.argsort()[::-1][:5]
print(img_path)
for i in top_inds:
    print('    {:.3f}'.format(pred[i]))
"""