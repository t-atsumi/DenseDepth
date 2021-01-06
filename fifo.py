import os
import glob
import argparse
import matplotlib
import cv2
import pipemethod
from PIL import Image
import time
import numpy as np
import struct
import zmq

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='models/laparo_trained_model/model.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--mindepth', type=float, default=5.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=40.0, help='Maximum of input depths')
args = parser.parse_args()

fifow = pipemethod.openFIFO("/tmp/fifo_test/points", "w")

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

while True:
    try:
        # Connection String
        conn_str = "tcp://*:5555"
        print("zmq connection opening")

        # Open ZMQ Connection
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(conn_str)

        print("data waiting")
        # Recieve Data from C++ Program、送る側が起動してないときはここで待機する
        byte_rows, byte_cols, byte_mat_type, data = sock.recv_multipart()
        print("data recieved")

        # Convert byte to integer
        rows = struct.unpack('i', byte_rows)
        cols = struct.unpack('i', byte_cols)
        mat_type = struct.unpack('i', byte_mat_type)

        # BGR Color
        image = np.frombuffer(data, dtype=np.uint8).reshape((rows[0],cols[0],3))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        loaded_images = []
        x = np.clip(rgb_image / 255, 0, 1)#画像を正規化してnumpy配列に置き換える
        loaded_images.append(x)
        inputs = np.stack(loaded_images, axis=0)

        print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
        print("predicting")

        # Compute results
        outputs = predict(model, inputs, minDepth=args.mindepth, maxDepth=args.maxdepth, batch_size=args.bs)
        print("pedicted")
        outputs2d = np.reshape(outputs, (128, 240))
        print("二次元配列に変換")
        outputimg = Image.fromarray((outputs2d * 255).astype(np.uint8))
        print("0から255に変換")
        outputresize = outputimg.resize((480, 270))
        print("480*270に変換")

        # print(data.getpixel((yoko,tate)))
        i = 0
        data = []
        while i < 480:
            j = 0
            while j < 270:
                data.append(outputresize.getpixel((i,j)))
                j += 10
            i += 10

        pipemethod.pack_and_write(data, fifow)
        
        time.sleep(5)

    except KeyboardInterrupt:
            print ('Finish')
            os.close(fifow)
            break