from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class TSRModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = 60  # Example number of filters
        self.kernel_size_1 = (5, 5)
        self.kernel_size_2 = (3, 3)
        self.pool_size = (2, 2)
        self.num_nodes = 500  # Example number of nodes in the dense layer
        self.model = self._build_model() 

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(self.num_filters, self.kernel_size_1, activation='relu', input_shape=(self.input_shape[0], self.input_shape[1], 1)))
        model.add(Conv2D(self.num_filters, self.kernel_size_1, activation='relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))

        model.add(Conv2D(self.num_filters//2, self.kernel_size_2, activation='relu'))
        model.add(Conv2D(self.num_filters//2, self.kernel_size_2, activation='relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(self.num_nodes, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model
    

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, generator, steps_per_epoch_val, batch_size=32, epochs=10):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.generator = generator
        self.steps_per_epoch_val = steps_per_epoch_val
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        history=self.model.fit_generator(self.generator.flow(self.X_train,self.y_train,batch_size=self.batch_size),steps_per_epoch=self.steps_per_epoch_val,
                                         epochs=self.epochs,validation_data=(self.X_val,self.y_val),shuffle=1)
        return history
    
    def plot_history(self, history):
        plt.figure(1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend(['training', 'validation'])
        plt.title('Loss over epochs')
        plt.xlabel('epoch')

        plt.figure(2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(['training', 'validation'])
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.show()
