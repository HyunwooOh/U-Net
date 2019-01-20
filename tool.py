import PIL.Image as pilimg
import numpy as np
from os import listdir
import tensorflow as tf

def read_images(path, shape):
    file_list = [f for f in listdir(path)]
    file_list.sort()
    images_np = np.zeros(shape=[1]+shape)
    for file in file_list:
        image_PIL = pilimg.open(path+"/"+file)
        resized_image_np = np.array(image_PIL.resize((shape[:-1])))
        images_np = np.append(images_np, resized_image_np.reshape([1]+shape), axis=0)
    return images_np[1:], images_np.shape[0]

def next_batch(inputs_np, labels_np, b, batch_size):
    batch_xs = inputs_np[b:b+batch_size, :, :, :]
    batch_ys = labels_np[b:b+batch_size, :, :, :]
    return np.array(batch_xs), np.array(batch_ys)

def total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
      #  print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
      #  print(variable_parameters)
        total_parameters += variable_parameters
    print("number of trainable parameters: %d"%(total_parameters))