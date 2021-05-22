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


def hourglass(x, depth=4):
    '''Define hourglass accordin to depth depth
    Args:
        x (Tensor): input of hourgalss
        depth (int): depth of hourglass
    
    Returns:
        Tensor output
    '''
    # depth 0
    d0_up1 = bottleneck2d(x, filters=128)
    d0_low1 = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    d0_low1 = bottleneck2d(x=d0_low1, filters=128)

    # depth 1
    d1_up1 = bottleneck2d(x=d0_low1, filters=128)
    d1_low1 = layers.MaxPooling2D(pool_size=2, strides=2)(d0_low1)
    d1_low1 = bottleneck2d(x=d1_low1, filters=128)

    # depth 2
    d2_up1 = bottleneck2d(x=d1_low2, filters=128)
    d2_low1 = layers.MaxPooling2D(pool_size=2, strides=2)(d1_low1)
    d2_low1 = bottleneck2d(x=d2_low1, filters=128)

    # depth 3
    d3_up1 = bottleneck2d(x=d2_low1, filters=128)
    d3_low1 = layers.MaxPooling2D(pool_size=2, strides=2)(d2_low1)
    d3_low1 = bottleneck2d(x=d3_low1, filters=128)

    # calculate d3 low2, low3, up2 and output
    d3_low2 = bottleneck2d(x=d3_low1, filters=128)
    d3_low3 = bottleneck2d(x=d3_low2, filters=128)
    d3_up2 = layers.UpSampling2D(size=2)(d3_low3)
    d3_out = layers.Add()([d3_up1 + d3_up2])

    # calculate d2 low2, low3, up2 and output
    d2_low2 = d3_out
    d2_low3 = bottleneck2d(x=d2_low2, filters=128)
    d2_up2 = layers.UpSampling2D(size=2)(d2_low3)
    d2_out = layers.Add()([d2_up1, d2_up2])

    # calculate d1 low2, low3, up2, output
    d1_low2 = d2_out
    d1_low3 = bottleneck2d(x=d1_low2, filters=128)
    d1_up2 = layers.UpSampling2D(size=2)(d1_low3)
    d1_out = layers.Add()([d1_up1, d1_up2])

    # calculate d0 low2, low3, up2, output
    d0_low2 = d1_out
    d0_low3 = bottleneck2d(x=d0_low2, filters=128)
    d0_up2 = layers.UpSampling2D(size=2)(d0_low3)
    d0_out = layers.Add()([d0_up1, d0_up2])

    return d0_out


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

    def make_fc(x_in):
        x_out = layers.Conv2D(256, kernel_size=1)(x_in)
        x_out = layers.BatchNormalization()(x_out)
        x_out = layers.Activation('relu')(x_out)

        return x_out


    # hourglass net 0
    hg0 = hourglass(x)
    res0 = bottleneck2d(x=hg0, filters=128)
    fc0 = make_fc(res0)
    score0 = ''
    
    # hourglass net 1
    hg1 = hourglass(hg0)
    res1 = bottleneck2d(x=hg1, filters=128)
    fc1 = make_fc(res1)
    score1 = ''

    return  Model(inputs=input_, outputs=x)