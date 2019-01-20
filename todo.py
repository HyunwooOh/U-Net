from network import UNetwork
from tool import read_images, next_batch, total_parameters
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def train(args):
    save_path = "./save_model/"
    input_data_path = "./new_data/train/X"
    label_data_path = "./new_data/train/Y"
    inputs_np, num_images = read_images(input_data_path, args.input_shape) # inputs_np shape: [60, 256, 256, 1]
    labels_np, _ = read_images(label_data_path, args.label_shape) # labels_np shape: [60, 256, 256, 1]
    model = UNetwork(args.input_shape, args.label_shape)
    total_parameters()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    batch_size = args.batch_size
    total_steps = int(num_images/batch_size)+1
    for epoch in range(args.epoch):
        loss_sum = 0
        b = 0
        for i in range(total_steps):
            batch_inputs = inputs_np[b:b+batch_size, :, :, :]
            batch_labels = labels_np[b:b+batch_size, :, :, :]
            _, loss_val = sess.run([model.training, model.loss], feed_dict = {model.inputs:np.float32(batch_inputs/255.), model.labels:np.float32(batch_labels/255.)})
            loss_sum += loss_val
            b+=batch_size
        print("Epoch: %4d | Loss: %.8f"%(epoch, loss_sum/total_steps))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if (epoch+1) % args.save_model_rate == 0:
            print("Saving model...")
            saver.save(sess, save_path + "model_"+str(epoch+1)+".cptk")
            save_image_path = "./assets/"
            if not os.path.exists(save_image_path): os.makedirs(save_image_path)
            test_input_data_path = "./new_data/test/X"
            test_label_data_path = "./new_data/test/Y"
            test_inputs_np, num_images = read_images(test_input_data_path, args.input_shape)  # inputs_np shape: [60, 256, 256, 1]
            test_labels_np, _ = read_images(test_label_data_path, args.label_shape)  # labels_np shape: [60, 256, 256, 1]
            image_index = random.randint(1, num_images)
            test_input_image = test_inputs_np[image_index]
            test_sample = sess.run(model.logits, feed_dict={model.inputs:np.float32(np.reshape(test_input_image, [1]+args.input_shape)/255.)})
            reshaped_input = np.reshape(test_input_image, args.input_shape[:-1])
            reshaped_label = np.reshape(test_labels_np[image_index], args.label_shape[:-1])
            reshaped_sample = np.reshape(test_sample, args.label_shape[:-1])
            _, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].set_axis_off()
            ax[0].imshow(reshaped_input, cmap="gray")
            ax[0].set_title("Raw image")
            ax[1].set_axis_off()
            ax[1].imshow(reshaped_label, cmap="gray")
            ax[1].set_title("Label image")
            ax[2].set_axis_off()
            ax[2].imshow(reshaped_sample, cmap="gray")
            ax[2].set_title("Created image")
            plt.show()
            plt.savefig(save_image_path+"result_"+str(epoch+1)+".png")

def test(args):
    save_image_path = "./assets/"
    if not os.path.exists(save_image_path): os.makedirs(save_image_path)
    save_path = "./save_model/"
    input_data_path = "./new_data/test/X"
    label_data_path = "./new_data/test/Y"
    inputs_np, num_images = read_images(input_data_path, args.input_shape) # inputs_np shape: [60, 256, 256, 1]
    labels_np, _ = read_images(label_data_path, args.label_shape) # labels_np shape: [60, 256, 256, 1]

    model = UNetwork(args.input_shape, args.label_shape)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, save_path+"model_"+args.model_num+".cptk")
    image_index = random.randint(1, num_images)
    input_image = inputs_np[image_index]
    sample = sess.run(model.logits, feed_dict={model.inputs:np.float32(np.reshape(input_image, [1]+args.input_shape)/255.)})

    reshaped_input = np.reshape(input_image, args.input_shape[:-1])
    reshaped_label = np.reshape(labels_np[image_index], args.label_shape[:-1])
    reshaped_sample = np.reshape(sample, args.label_shape[:-1])
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].set_axis_off()
    ax[0].imshow(reshaped_input, cmap="gray")
    ax[0].set_title("Raw image")
    ax[1].set_axis_off()
    ax[1].imshow(reshaped_label, cmap="gray")
    ax[1].set_title("Label image")
    ax[2].set_axis_off()
    ax[2].imshow(reshaped_sample, cmap="gray")
    ax[2].set_title("Created image")
    plt.show()
