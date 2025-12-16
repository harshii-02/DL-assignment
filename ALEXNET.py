from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.initializers import HeNormal
import numpy as np


class OptimizedAlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

       
        self.add(Conv2D(
            filters=64,
            kernel_size=(11, 11),
            strides=4,
            padding='same',
            kernel_initializer=HeNormal(),
            input_shape=input_shape
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # -------- Block 2 --------
        self.add(Conv2D(
            filters=192,
            kernel_size=(5, 5),
            padding='same',
            kernel_initializer=HeNormal()
        ))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        # -------- Block 3 --------
        self.add(Conv2D(
            filters=384,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=HeNormal()
        ))
        self.add(BatchNormalization())

        self.add(Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=HeNormal()
        ))
        self.add(BatchNormalization())

        self.add(Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=HeNormal()
        ))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2))

       
        self.add(GlobalAveragePooling2D())
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(num_classes, activation='softmax'))



model = OptimizedAlexNet(
    input_shape=(224, 224, 3),
    num_classes=10
)


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()


dummy_input = np.random.rand(1, 224, 224, 3)
prediction = model.predict(dummy_input)

print("\nPrediction shape:", prediction.shape)
print("Sum of probabilities:", prediction.sum())
