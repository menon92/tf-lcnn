import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def multitask_head(x, num_class):
	'''Multitaks head implimentaiton
	Args:
		x (Tensor): input to multitaks head
		num_class (int): number of class or task
	'''
	head_size = [2, 1, 2]
	print(f"multitask_head input shape: {tf.shape(x)}")
	m_out = int(tf.shape(x)[-1]) // 4
	print(f'm_out {m_out}')

	# head size 2
	x2 = layers.Conv2D(m_out, kernel_size=3, padding='SAME')(x)
	x2 = layers.Activation('relu')(x2)
	x2 = layers.Conv2D(head_size[0], kernel_size=1, padding='SAME')(x2)

	# head size 1
	x1 = layers.Conv2D(m_out, kernel_size=3, padding='SAME')(x)
	x1 = layers.Activation('relu')(x1)
	x1 = layers.Conv2D(head_size[1], kernel_size=1, padding='SAME')(x1)
 
	# head size 2
	x3 = layers.Conv2D(m_out, kernel_size=3, padding='SAME')(x)
	x3 = layers.Activation('relu')(x3)
	x3 = layers.Conv2D(head_size[2], kernel_size=1, padding='SAME')(x3)

	# combine three head
	out = layers.Concatenate(axis=-1)([x2, x1, x3])

	return out

