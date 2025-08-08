from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam

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