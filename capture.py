import os
import glob
import argparse
import matplotlib
import cv2
import pipemethod
from PIL import Image
import time
import numpy as np

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

cap = cv2.VideoCapture(0)  ###カメラのオープン

fifo = pipemethod.openFIFO("/tmp/fifo_test/points", "w")

while True:
    try:
        ret, frame = cap.read()###1フレームの読み込み

        #resizeはwidth,heightの順に指定する
        frame=cv2.resize(frame,frame.shape[1],frame.shape[0]*256/270)###shape[1]:width,shape[0]:height

        # Custom object needed for inference and training
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

        print('Loading model...')

        # Load model into GPU / CPU
        model = load_model(args.model, custom_objects=custom_objects, compile=False)

        print('\nModel loaded ({0}).'.format(args.model))

        # Compute results
        outputs = predict(model, frame, minDepth=args.mindepth, maxDepth=args.maxdepth, batch_size=args.bs)
        outputs2d = np.reshape(outputs, (128, 240))
        outputimg = Image.fromarray((outputs2d*255).astype(np.uint8))
        outputresize = outputimg.resize((480, 270))

        # print(data.getpixel((yoko,tate)))
        i = 0
        data = []
        while i < 480:
            j = 0
            while j < 270:
                data.append(outputresize.getpixel((i,j)))
                j += 10
            i += 10

        pack_and_write(data, fifo)
        
        time.sleep(5)

    except KeyboardInterrupt:
            print ('Finish')
            os.close(fifo)
            cap.release()
            break