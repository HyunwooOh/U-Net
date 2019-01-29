import tensorflow as tf
import numpy as np
import PIL.Image as pilimg
from os import listdir
import os
import math
import random

def read_images(path):
    file_list = [f for f in listdir(path)]
    file_list.sort()
    images_np = np.zeros(shape=[1, 512, 512])
    for file in file_list:
        image_PIL = pilimg.open(path+"/"+file) # [512, 512, 1]
        images_np = np.append(images_np, np.array(image_PIL).reshape([1, 512, 512]), axis=0)
    return images_np[1:], images_np[1:].shape[0]

def _largest_rotated_rect(w, h, angle):
    """
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = h if (w < h) else w
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (bb_w - 2 * x, bb_h - 2 * y)

def random_aug(name, new_input_data_path, new_label_data_path, origin_images):
    inputs = tf.placeholder(tf.float32, [512, 512, 2])
    rotation_degree = random.randint(0, 360)
    rotation_radians = math.radians(rotation_degree)
    image = tf.contrib.image.rotate(inputs, rotation_radians, interpolation='BILINEAR')
    lrr_width, lrr_height = _largest_rotated_rect(512, 512, rotation_radians)
    resized_image = tf.image.central_crop(image, float(lrr_height) / 512)
    resized_image = tf.image.resize_images(resized_image, [int(lrr_height), int(lrr_width)], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    cropping = tf.random_crop(resized_image, [256, 256, 2])
    sess = tf.Session()
    cropped_images = sess.run(cropping, feed_dict={inputs:origin_images})
    cropped_input_images = np.array(cropped_images[:,:,0])
    cropped_label_images = np.array(cropped_images[:,:,1])
    #save
    cropped_input_images = pilimg.fromarray(cropped_input_images.astype(np.uint8))
    cropped_input_images.save(new_input_data_path+"/X_"+str(name)+".png")
    cropped_label_images = pilimg.fromarray(cropped_label_images.astype(np.uint8))
    cropped_label_images.save(new_label_data_path+"/Y_"+str(name)+".png")

def aug(args):
    origin_input_data_path = "./original_data/X"
    origin_label_data_path = "./original_data/Y"
    new_input_data_path = "./new_data/X"
    new_label_data_path = "./new_data/Y"
    if not os.path.exists(new_input_data_path): os.makedirs(new_input_data_path)
    if not os.path.exists(new_label_data_path): os.makedirs(new_label_data_path)
    origin_input_images_np, num_images = read_images(origin_input_data_path) # inputs_np shape: [30, 512, 512]
    origin_label_images_np, _ = read_images(origin_label_data_path) # inputs_np shape: [30, 512, 512]
    origin_images_np = np.concatenate([np.reshape(origin_input_images_np, [num_images, 512, 512, 1]), np.reshape(origin_label_images_np, [num_images, 512, 512, 1])], axis=-1) # [30, 512, 512, 2]
    name = 0
    for j in range(args.aug_size):
        for i in range(num_images):
            name+=1
            random_aug(name, new_input_data_path, new_label_data_path, origin_images_np[i])
