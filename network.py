import tensorflow as tf
class UNetwork():
    def __init__(self, input_shape, output_shape, drop_out=False):
        if input_shape != output_shape: self.padding = "valid"
        else: self.padding = "same"
        self.drop_out = drop_out
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.logits = self.model(input_shape)
        self.labels = tf.placeholder(tf.float32, [None]+output_shape)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.logits))
        self.training = self.optimizer.minimize(self.loss)

    def model(self, input_shape):
        self.inputs = tf.placeholder(tf.float32, [None]+input_shape)
        # Contracting Path
        c1_1 = self.conv(self.inputs, 64)
        c1_2 = self.conv(c1_1, 64)
        c2_1 = self.max_pool(c1_2)
        c2_2 = self.conv(c2_1, 128)
        c2_3 = self.conv(c2_2, 128)
        c3_1 = self.max_pool(c2_3)
        c3_2 = self.conv(c3_1, 256)
        c3_3 = self.conv(c3_2, 256)
        c4_1 = self.max_pool(c3_3)
        c4_2 = self.conv(c4_1, 512)
        c4_3 = self.conv(c4_2, 512)
        ################
        c5_1 = self.max_pool(c4_3)
        c5_2 = self.conv(c5_1, 1024)
        c5_3 = self.conv(c5_2, 1024)
        # Expansive Path
        c_4_1= self.up_conv(c5_3, 512)
        c_4_1= self.copy_and_crop(c4_3, c_4_1)
        c_4_2= self.conv(c_4_1, 512)
        c_4_3= self.conv(c_4_2, 512)
        c_3_1= self.up_conv(c_4_3, 256)
        c_3_1= self.copy_and_crop(c3_3, c_3_1)
        c_3_2= self.conv(c_3_1, 256)
        c_3_3= self.conv(c_3_2, 256)
        c_2_1= self.up_conv(c_3_3, 128)
        c_2_1= self.copy_and_crop(c2_3, c_2_1)
        c_2_2= self.conv(c_2_1, 128)
        c_2_3= self.conv(c_2_2, 128)
        c_1_1= self.up_conv(c_2_3, 64)
        c_1_1= self.copy_and_crop(c1_2, c_1_1)
        c_1_2= self.conv(c_1_1, 64)
        c_1_3= self.conv(c_1_2, 64)
        c_1_4 = self.conv(c_1_3, 1, kernel_size=1, activation=tf.keras.activations.sigmoid)
        return c_1_4

    def conv(self, input, num_outputs, kernel_size=3, stride=1, activation=tf.keras.activations.relu):
        tmp = tf.keras.layers.Conv2D(filters = num_outputs, kernel_size = kernel_size, strides = [stride, stride], padding = self.padding, activation = None)(input)
        tmp = tf.keras.layers.BatchNormalization()(tmp)
        tmp = activation(tmp)
        if self.drop_out == True: return tf.keras.layers.Dropout(rate=0.5)(tmp)
        else: return tmp

    def max_pool(self, input, pool_size=2, stride=2):
        return tf.keras.layers.MaxPool2D(pool_size=(pool_size, pool_size), strides=stride)(input)

    def up_conv(self, input, num_outputs, kernel_size=2, stride=2, activation=tf.keras.activations.relu):
        tmp = tf.keras.layers.Conv2DTranspose(filters = num_outputs, kernel_size = kernel_size, strides = (stride, stride), padding = self.padding, activation = None)(input)
        tmp = tf.keras.layers.BatchNormalization()(tmp)
        tmp = activation(tmp)
        return tmp

    def copy_and_crop(self, source, target):
        source_h = int(source.get_shape().as_list()[1])
        source_w = int(source.get_shape().as_list()[2])
        target_h = int(target.get_shape().as_list()[1])
        target_w = int(target.get_shape().as_list()[2])
        offset_h = int((source_h - target_h)/2)
        offset_w = int((source_w - target_w)/2)
        crop = tf.image.crop_to_bounding_box(source, offset_h, offset_w, target_h, target_w)
        copy = tf.concat([crop, target], -1)
        return copy