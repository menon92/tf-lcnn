import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class Bottleneck2D(tf.keras.Model):
    expansion = 2
    
    def __init__(self, filters, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = layers.BatchNormalization(axis=-1)
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv2D(filters, kernel_size=1, padding='SAME', activation='relu')
        
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu2 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(
                filters, kernel_size=3, padding='SAME', strides=1, activation='relu')
        
        self.bn3 = layers.BatchNormalization(axis=-1)
        self.relu3 = layers.Activation('relu')
        self.conv3 = layers.Conv2D(filters*2, kernel_size=1, padding='SAME', activation='relu')
        
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, x):
        residual = x # conv->bn->relu

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out = self.add([out, residual])

        return out

    def model(self):
        '''Use only for print model summary'''
        x = layers.Input(shape=(16, 16, 8*self.expansion))
        return Model(inputs=[x], outputs=self.call(x))
