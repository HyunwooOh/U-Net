from network import UNetwork
from tool import read_images, next_batch, total_parameters, setup_summary, make_path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from os import listdir

def train(args):
    make_path()
    save_path = "./save_model/"
    input_data_path = "./new_data/train/X"
    label_data_path = "./new_data/train/Y"
    val_input_data_path = "./new_data/valid/X"
    val_label_data_path = "./new_data/valid/Y"
    file_list = [f for f in listdir(input_data_path)]
    num_images = len(file_list)
    val_file_list = [f for f in listdir(val_input_data_path)]
    val_num_images = len(val_file_list)
    val_inputs_np = read_images(val_input_data_path, args.input_shape, 0, val_num_images)  # inputs_np shape: [30, 256, 256, 1]
    val_labels_np = read_images(val_label_data_path, args.label_shape, 0, val_num_images)  # labels_np shape: [30, 256, 256, 1]
    model = UNetwork(args.input_shape, args.label_shape)
    print("model initialized")
    total_parameters()
    sess = tf.Session()
    summary_placeholders, update_ops, summary_op = setup_summary(["Training Accuracy", "Validation Accuracy"])
    summary_writer = tf.summary.FileWriter('summary/', sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    batch_size = args.batch_size
    total_steps = int(num_images/batch_size)+1
    for epoch in range(args.epoch):
        if args.drop_out == "True": model.drop_out = True
        loss_sum = 0
        acc_sum = 0
        b = 0
        for i in range(total_steps):
            batch_inputs = read_images(input_data_path, args.input_shape, b, b+batch_size)
            batch_labels = read_images(label_data_path, args.input_shape, b, b+batch_size)
            train_sample, _, loss_val = sess.run([model.logits, model.training, model.loss], feed_dict = {model.inputs:np.float32(batch_inputs/255.), model.labels:np.float32(batch_labels/255.)})
            loss_sum += loss_val
            b+=batch_size
            acc = float((1 - np.sum(abs(np.float32(batch_labels/255.) - train_sample)) / (256 * 256 * int(batch_inputs.shape[0]))) * 100)
            acc_sum += acc
        model.drop_out = False
        image_index = random.randint(0, val_num_images-1)
        val_sample, val_loss = sess.run([model.logits, model.loss], feed_dict={model.inputs:np.float32(val_inputs_np/255.), model.labels:np.float32(val_labels_np/255.)})
        reshaped_input = np.reshape(np.float32(val_inputs_np[image_index]/255.), args.input_shape[:-1])
        reshaped_label = np.reshape(np.float32(val_labels_np[image_index]/255.), args.label_shape[:-1])
        reshaped_sample = np.reshape(val_sample[image_index], args.label_shape[:-1])
        print("Epoch: %4d |-------| TEST Loss: %.8f | Accuracy: %.2f%% |-------| VAL Loss = %.8f | Accuracy = %.2f%%"%(epoch, loss_sum/total_steps, acc_sum/total_steps, val_loss, float((1 - np.sum(abs(np.float32(val_labels_np/255.) - val_sample)) / (256 * 256 * int(val_inputs_np.shape[0]))) * 100)))
        f = open("./report/training.txt", 'a')
        f.write("%4d\t%.8f\t%.2f\t%.8f\t%.2f\n" % (epoch, loss_sum/total_steps, acc_sum/total_steps, val_loss, float((1 - np.sum(abs(np.float32(val_labels_np/255.) - val_sample)) / (256 * 256 * int(val_inputs_np.shape[0]))) * 100)))
        f.close()
        summary_stats = [acc_sum/total_steps, float((1 - np.sum(abs(np.float32(val_labels_np/255.) - val_sample)) / (256 * 256 * int(val_inputs_np.shape[0]))) * 100)]#, step]
        for i in range(len(summary_stats)):
            sess.run(update_ops[i], feed_dict={summary_placeholders[i]: float(summary_stats[i])})
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, epoch + 1)
        if (epoch+1) % args.save_model_rate == 0:
            print("Saving model...")
            saver.save(sess, save_path + "model_"+str(epoch+1)+".cptk")
            save_image_path = "./assets/"
            if not os.path.exists(save_image_path): os.makedirs(save_image_path)
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
    input_data_path = "./new_data/train/X"
    label_data_path = "./new_data/train/Y"
    file_list = [f for f in listdir(input_data_path)]
    num_images = len(file_list)
    inputs_np = read_images(input_data_path, args.input_shape, 0, num_images) # inputs_np shape: [30, 256, 256, 1]
    labels_np = read_images(label_data_path, args.label_shape, 0, num_images) # labels_np shape: [30, 256, 256, 1]

    model = UNetwork(args.input_shape, args.label_shape, drop_out=False)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, save_path+"model_"+args.model_num+".cptk")
    image_index = random.randint(0, num_images-1)
    input_image = inputs_np[image_index]
    input_label = labels_np[image_index]
    sample, loss = sess.run([model.logits, model.loss], feed_dict={model.inputs:np.float32(np.reshape(input_image, [1]+args.input_shape)/255.), model.labels: np.float32(np.reshape(input_label, [1]+args.input_shape)/255.)})
    reshaped_input = np.reshape(input_image, args.input_shape[:-1])/255.
    reshaped_label = np.reshape(labels_np[image_index], args.label_shape[:-1])/255.
    reshaped_sample = np.reshape(sample, args.label_shape[:-1])
    print("Accuracy = %.2f%%"%float((1-np.sum(abs(reshaped_label-reshaped_sample))/(256*256))*100))

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