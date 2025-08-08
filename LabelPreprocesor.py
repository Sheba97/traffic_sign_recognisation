from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

class Augmenter:
    def __init__(self, X_train, X_val, X_test, y_train, image_dim, batch_size=32):
        self.generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.batch_size = batch_size
        self.image_dim = image_dim

    def reshape(self):
        """Reshape the images to the format expected by the generator."""
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.X_val = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[1], self.X_val.shape[2], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)
        self.fit()
        return self.X_train, self.X_val, self.X_test
    
    def fit(self):
        """Fit the augmenter to the training data."""
        self.generator.fit(self.X_train)

    def flow(self):
        batches =  self.generator.flow(self.X_train, self.y_train, batch_size=self.batch_size)
        return next(batches)

    def plot_batch(self):
        """Plot a batch of augmented images."""
        self.X_batch, self.y_batch = self.flow()
        fig,axs = plt.subplots(1, self.batch_size, figsize=(20, 5))  
        fig.tight_layout()
        for i in range(15):
            axs[i].imshow(self.X_batch[i].reshape(self.image_dim[0],self.image_dim[1]))
            axs[i].axis('off')
        plt.show()

class LabelPreprocessor:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def encode(self, y):
        return to_categorical(y, num_classes=self.num_classes)
    
    def process_all(self, y_train, y_val, y_test):
        return self.encode(y_train), self.encode(y_val), self.encode(y_test)