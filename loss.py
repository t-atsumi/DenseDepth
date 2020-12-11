import keras.backend as K
import tensorflow as tf
import numpy as np

def depth_loss_function(y_true, y_pred, theta3=0.1, theta4=0.0, maxDepthVal=25.0):
    
    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # 画像一枚内での深度の平滑性
    l_smooth = (K.max(y_pred) - K.min(y_pred))
    # kernelX = np.array([[[[0, 0, 0], 
    #                     [0, -1, 1],
    #                     [0, 0, 0]]]]).reshape(3,3,1,1)
    # kernelX = K.constant(kernelX)
    # output = K.conv2d(x=y_pred, kernel=kernelX, strides=(1, 1), padding='valid')
    # output = K.square(output)
    # l_smooth = K.sum(output)

    # Weights
    w1 = 1.0
    w3 = theta3
    w4 = theta4

    return (w1 * l_ssim) + (w3 * K.mean(l_depth)) + (w4 * l_smooth)