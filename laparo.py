import os
import glob
import argparse
import matplotlib
import cv2

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)###カメラのオープン

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')###変えろカス
parser.add_argument('--input', default='examples/default/*.png', type=str, help='Input filename or folder.')###変えろカス
parser.add_argument('--input_depth', default='', type=str, help='Input Deoth filename or folder.')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--mindepth', type=float, default=5.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=40.0, help='Maximum of input depths')
args = parser.parse_args()

ret, frame = cap.read()###1フレームの読み込み

frame=cv2.resize(frame,frame.shape[1],frame.shape[0])###いい感じにリサイズ

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Compute results
outputs = predict(model, frame, minDepth=args.mindepth, maxDepth=args.maxdepth, batch_size=args.bs)

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