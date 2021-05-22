import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from lcnn.models.hourglass_pose import Bottleneck2D
from lcnn.models.hourglass_pose import HourglassNet
from lcnn.models.hourglass_pose import hourglass_net


# filters = 8
# bottleneck2d = Bottleneck2D(filters)
# bottleneck2d = bottleneck2d.model()
# bottleneck2d.summary()

# plot_model(bottleneck2d, 'figs/Bottleneck2D.png')

# print('-' * 50)
# hourglass_net = HourglassNet(Bottleneck2D)
# # hourglass_net = hourglass_net.model()
# x = tf.random.uniform(shape=(1, 128, 128, 3))
# hourglass_net(x)
# hourglass_net.summary()

# plot_model(hourglass_net, 'figs/HourglassNet.png')

model = hourglass_net()
model.summary()
plot_model(model, 'figs/HourglassNet-Static-Graph.png')