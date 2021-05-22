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

        print("out", out)
        print('residual', residual)
        out = self.add([out, residual])

        return out

    def model(self):
        '''Use only for print model summary'''
        x = layers.Input(shape=(16, 16, 8*self.expansion))
        return Model(inputs=[x], outputs=self.call(x))


def bottleneck2d(x, filters, downsample=None):
    residual = x

    if downsample:
        residual = downsample(x)

    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=1, padding='SAME', activation='relu')(x)
    
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size=3, padding='SAME', activation='relu')(x)
    
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters*2, kernel_size=1, padding='SAME', activation='relu')(x)
    
    x = layers.Add()([x, residual])

    return x


class HourglassNet(tf.keras.Model):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, block: Bottleneck2D):
        super(HourglassNet, self).__init__()
        
        self.conv1 = layers.Conv2D(64, kernel_size=7, padding='SAME', strides=2)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')

        # layer1 block
        self.downsample1 = layers.Conv2D(128, kernel_size=1, strides=1)
        self.layer1 = block(filters=64, downsample=self.downsample1)

        self.max_pool = layers.MaxPooling2D(pool_size=2, strides=2)

        # layer2 block
        self.downsample2 = layers.Conv2D(256, kernel_size=1, strides=1)
        self.layer2 = block(filters=128, downsample=self.downsample2)

        # layer3 block, no downsample
        self.layer3 = block(filters=128)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.max_pool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def model(self):
        x = layers.Input(shape=(128, 128, 3))
        return Model(inputs=[x], outputs=self.call(x))


def hourglass_net():
    input_ = layers.Input(shape=(128, 128, 3))

    x = layers.Conv2D(64, kernel_size=7, padding='SAME', strides=2)(input_)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # block 1
    x = bottleneck2d(
        x,
        filters=64,
        downsample=layers.Conv2D(128, kernel_size=1, strides=1, name='downsample_1')
    )

    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # block 2
    x = bottleneck2d(
        x,
        filters=128,
        downsample=layers.Conv2D(256, kernel_size=1, strides=1, name='downsample_2')
    )

    # block 3
    x = bottleneck2d(x, filters=128)

    return  Model(inputs=input_, outputs=x)