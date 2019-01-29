import PIL.Image as pilimg
import numpy as np
from os import listdir
import tensorflow as tf
import os

def read_images_(path, shape):
    file_list = [f for f in listdir(path)]
    file_list.sort()
    images_np = np.zeros(shape=[1]+shape)
    for file in file_list:
        image_PIL = pilimg.open(path+"/"+file)
        resized_image_np = np.array(image_PIL.resize((shape[:-1])))
        images_np = np.append(images_np, resized_image_np.reshape([1]+shape), axis=0)
    return images_np[1:], images_np.shape[0]-1

def read_images(path, shape, data_from, data_to):
    file_list = [f for f in listdir(path)]
    file_list.sort()
    images_np = np.zeros(shape=[1]+shape)
    a = file_list[data_from:data_to]
    a.sort()
    for file in file_list[data_from:data_to]:
        image_PIL = pilimg.open(path+"/"+file)
        resized_image_np = np.array(image_PIL.resize((shape[:-1])))
        images_np = np.append(images_np, resized_image_np.reshape([1]+shape), axis=0)
    return images_np[1:]

def next_batch(inputs_np, labels_np, b, batch_size):
    batch_xs = inputs_np[b:b+batch_size, :, :, :]
    batch_ys = labels_np[b:b+batch_size, :, :, :]
    return np.array(batch_xs), np.array(batch_ys)

def total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("number of trainable parameters: %d"%(total_parameters))

def setup_summary(list):
    variables = []
    for i in range(len(list)):
        variables.append(tf.Variable(0.))
        tf.summary.scalar(list[i], variables[i])
    summary_vars = [x for x in variables]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op

def make_path():
    save_path = "./save_model/"
    if not os.path.exists(save_path): os.makedirs(save_path)
    summary_path = "./summary/"
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    report_path = "./report/"
    if not os.path.exists(report_path): os.makedirs(report_path)
    input_data_path = "./new_data/train/X"
    if not os.path.exists(input_data_path): os.makedirs(input_data_path)
    label_data_path = "./new_data/train/Y"
    if not os.path.exists(label_data_path): os.makedirs(label_data_path)
    val_input_data_path = "./new_data/valid/X"
    if not os.path.exists(val_input_data_path): os.makedirs(val_input_data_path)
    val_label_data_path = "./new_data/valid/Y"
    if not os.path.exists(val_label_data_path): os.makedirs(val_label_data_path)
