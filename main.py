from DataLoader import DataLoader
from DataVisualizer import *
from ImagePreprocessor import *
from LabelPreprocesor import *
from TSRModel import *              # Importing the CNN TSRModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")


def main():
    data_path = 'D:/Anhalt/Machine Learning/TSR/env1/TSR/image_data/traffic_data_32'
    label_file = 'D:/Anhalt/Machine Learning/TSR/env1/TSR/image_data/labels/labels.csv'
    image_dim = (32, 32, 3)
    test_ratio = 0.2
    val_ratio = 0.2
    batch_size_val = 32
    epochs_val = 20
    steps_per_epoch_val = 100

    data_loader = DataLoader(label_file, image_dim, test_ratio, val_ratio, data_path)
    images, class_labels = data_loader.load_data()
    num_classes = data_loader.classNo
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(images, class_labels)
    print(X_train.shape[0], X_val.shape[0], X_test.shape[0])
    assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
    assert(X_val.shape[0]==y_val.shape[0]), "The number of images in not equal to the number of lables in validation set"
    assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
    assert(X_train.shape[1:]==(image_dim))," The dimesions of the Training images are wrong "
    assert(X_val.shape[1:]==(image_dim))," The dimesionas of the Validation images are wrong "
    assert(X_test.shape[1:]==(image_dim))," The dimesionas of the Test images are wrong"

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of classes: {num_classes}")

    data_visualizer = DataVisualizer(X_train, y_train, label_file, num_classes)
    data_visualizer.show_sample_images()
    data_visualizer.plot_distribution()

    # Preprocess images
    img_preprocessor = ImagePreprocessor()
    X_train=np.array(list(map(img_preprocessor.preprocess,X_train))) 
    X_validation=np.array(list(map(img_preprocessor.preprocess,X_val)))
    X_test=np.array(list(map(img_preprocessor.preprocess,X_test)))
    cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Augment images
    augmenter = Augmenter(X_train, X_validation, X_test, y_train, image_dim, 20)
    X_train, X_val, X_test = augmenter.reshape()
    augmenter.plot_batch()

    # Process labels
    label_preprocessor = LabelPreprocessor(num_classes)
    y_train_encoded, y_val_encoded, y_test_encoded = label_preprocessor.process_all(y_train, y_val, y_test)
    
    keras_model(image_dim, num_classes, X_train, X_test , y_train_encoded, y_test_encoded, X_validation, y_val_encoded, augmenter.generator,steps_per_epoch_val, batch_size_val, epochs_val)
 
    #train_and_eval_pytorch(X_train, y_train_encoded, X_validation, y_val_encoded, X_test, y_test_encoded, num_classes, epochs=10, batch_size=32)

def keras_model(image_dim, num_classes, X_train, X_test, y_train_encoded, y_test_encoded, X_validation, y_val_encoded, generator, steps_per_epoch_val, batch_size_val, epochs_val):
    # Build and compile the model
    model = TSRModel(image_dim, num_classes)
    tsr_model = model.model

    if tsr_model is not None and hasattr(tsr_model, "summary"):
        print(tsr_model.summary())
    else:
        print("Error: tsr_model is None or does not have a summary() method.")

    # Train the model
    trainer = Trainer(tsr_model, X_train, y_train_encoded, X_validation, y_val_encoded, generator, steps_per_epoch_val, batch_size_val, epochs_val)
    history = trainer.train()
    trainer.plot_history(history)

    # Evaluate the model
    test_loss, test_accuracy = tsr_model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Save the model
    tsr_model.save("D:/Anhalt/Machine Learning/TSR/env1/TSR/model/traffic_sign_cnn_32.h5")
    print("Model saved as traffic_sign_cnn_32.h5")

if __name__ == "__main__":
    main()   