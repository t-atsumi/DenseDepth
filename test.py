import os
import glob
import argparse
import matplotlib
import zmq
import cv2
import struct
import numpy as np
from PIL import Image

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/default/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--input_depth', default='', type=str, help='Input Deoth filename or folder.')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images、globモジュールのglob関数は引数に受け取った条件に合致するファイルのパスを返す
# Connection String
conn_str      = "tcp://*:5555"

# Open ZMQ Connection
ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind(conn_str)

# Receve Data from C++ Program
byte_rows, byte_cols, byte_mat_type, data=  sock.recv_multipart()

# Convert byte to integer
rows = struct.unpack('i', byte_rows)
cols = struct.unpack('i', byte_cols)
mat_type = struct.unpack('i', byte_mat_type)

if mat_type[0] == 0:
    # Gray Scale
    image = np.frombuffer(data, dtype=np.uint8).reshape((rows[0],cols[0]));
else:
    # BGR Color
    image = np.frombuffer(data, dtype=np.uint8).reshape((rows[0],cols[0],3));
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

loaded_images = []
x = np.clip(rgb_image / 255, 0, 1)#画像を正規化してnumpy配列に置き換える
loaded_images.append(x)
inputs = np.stack(loaded_images, axis=0)

print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs, minDepth=args.mindepth, maxDepth=args.maxdepth, batch_size=args.bs)
#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
if args.input_depth=='examples/eye/*.png':
    inputs_depth = load_images( glob.glob(args.input_depth))    
    viz = display_images(outputs.copy(), inputs.copy(), inputs_depth.copy())
else:
    viz = display_images(outputs.copy(), inputs.copy())

plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
