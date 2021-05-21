from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from lcnn.models.hourglass_pose import Bottleneck2D


filters = 8
bottleneck2d = Bottleneck2D(filters)
bottleneck2d = bottleneck2d.model()
bottleneck2d.summary()

plot_model(bottleneck2d, 'figs/Bottleneck2D.png')