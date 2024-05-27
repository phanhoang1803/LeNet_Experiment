import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D

class LeNet(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        # Define LeNet layers
        self.conv1 = Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='tanh')
        self.avgpool1 = AveragePooling2D(pool_size=(2,2), strides=(2, 2))

        self.conv2 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='tanh')
        self.avgpool2 = AveragePooling2D(pool_size=(2,2), strides=(2, 2))
        
        self.flatten = Flatten()
        
        self.dense1 = Dense(units=120, activation='tanh')
        self.dense2 = Dense(units=84, activation='tanh')
        self.dense3 = Dense(units=self.num_classes, activation='softmax')
    
    # Override call function of keras.Model
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
    
    # Override get_config
    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super(LeNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))