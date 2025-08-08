import os
import cv2
import numpy as np
import sys
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, label_file, image_dim, test_ratio, val_ratio, data_path):
        self.data_path = data_path
        self.label_file = label_file
        self.image_dim = image_dim
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.images = []
        self.class_labels = []
        self.classNo = 0

    def load_data(self):
        class_folders = os.listdir(self.data_path)
        self.classNo = len(class_folders)

        for folder in class_folders:
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            class_num = int(folder)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (self.image_dim[0], self.image_dim[1]))
                    self.images.append(img)
                    self.class_labels.append(class_num)
        
        return np.array(self.images), np.array(self.class_labels)

    def split_data(self, images, class_labels):
        X_train, X_test, y_train, y_test = train_test_split(images, class_labels, test_size=self.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_ratio)
        return X_train, X_val, X_test, y_train, y_val, y_test