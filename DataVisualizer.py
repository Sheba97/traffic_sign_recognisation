import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

class DataVisualizer:
    def __init__(self, X_train, y_train, lable_data, num_classes):
        self.X_train = X_train
        self.y_train = y_train
        self.lable_data = lable_data
        self.num_classes = num_classes
        self.num_samples = []

    def show_sample_images(self):
        data = pd.read_csv(self.lable_data)
        cols = 5
        fig, axs = plt.subplots(nrows=self.num_classes, ncols=cols, figsize=(5, 300))
        fig.tight_layout()
        for i in range(cols):
            for j,row in data.iterrows():
                x_selected = self.X_train[self.y_train == j]
                axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)- 1), :, :], cmap=plt.get_cmap("gray"))
                axs[j][i].axis("off")
                if i == 2:
                    axs[j][i].set_title(str(j)+ "-"+row["Name"])
                    self.num_samples.append(len(x_selected))
        '''fig.tight_layout(pad=3.0)
        for i in range(self.num_classes):
            img = self.X_train[self.y_train == i][0]
            axs[i // cols, i % cols].imshow(img)
            axs[i // cols, i % cols].set_title(str(i))
            axs[i // cols, i % cols].axis('off')
        plt.tight_layout()
        plt.show()'''


    def plot_distribution(self):
        #samples_per_class = [np.sum(self.y_train == i) for i in range(self.num_classes)]
        #plt.bar(range(self.num_classes), samples_per_class)
        plt.figure(figsize=(12, 4))
        plt.bar(range(0, self.num_classes), self.num_samples)
        plt.title("Distribution of the training dataset")
        plt.xlabel("Class number")
        plt.ylabel("Number of images")
        plt.show()