from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation

model = Sequential([
    Conv2D(2, kernel_size=(3,3), input_shape=(20,20,3)),
    Activation('relu'),
    Conv2D(3, kernel_size=(3,3), activation='relu'),
    Flatten(),
    Dense(2, activation='softmax')
])
print(model.trainable_variables)
model.summary()