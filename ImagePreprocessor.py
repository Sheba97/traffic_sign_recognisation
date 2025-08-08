import cv2

class ImagePreprocessor:
    @staticmethod
    def grayscale(image):
        """Convert an image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def equalize_histogram(image):
        """Apply histogram equalization to an image."""
        return cv2.equalizeHist(image)
    
    @staticmethod
    def normalize(image):
        """Normalize an image to the range [0, 1]."""
        return image / 255.0
    
    @staticmethod
    def preprocess(image):
        """Preprocess the image by converting to grayscale, equalizing histogram, and normalizing."""
        image = ImagePreprocessor.grayscale(image)
        image = ImagePreprocessor.equalize_histogram(image)
        image = ImagePreprocessor.normalize(image)
        return image