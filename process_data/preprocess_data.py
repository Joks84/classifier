import os
import numpy as np
from PIL import Image


# pixel size for resizing the images
pixel_size = (100, 100)


def addNoise(image: Image) -> Image:
    """
    Adds a noise to the image and returns it.
    Params:
        image(Image): The image to add the noise to.

    Return:
        image(Image): The image with the noise.
    """

    # create an image with noise.
    noise_imgage = Image.effect_noise(pixel_size, 50)
    # overlap noise image with image parameter.
    overlap_image = Image.blend(noise_imgage.convert("RGBA"), image.convert("RGBA"), alpha = 0.5)
    return overlap_image


def imageToArray(image: Image, isChestnut: int) -> np.array:
    """
    Converts image to numpy array and returns it.
    Params:
        image(Image): The image to convert.
        isChestnut(int): The label for the image.

    Return:
        array(np.array): The image converted to array.
    """

    matrix = np.asarray(image).astype("uint8")
    flat = matrix.flatten()
    labeled_array = np.append(flat, [isChestnut])

    return labeled_array

def preprocessTrainImages(path: str, degrees_increase: int or float) -> np.array:
    """
    Prepares the images for model training.
    Params:
        path(str): The path to training images.
        degrees_increase(int or float): The number of degrees to rotate the images.

    Return:
        images_matrix(np.array): The array of image pixels.
    """

    # matrix to store the images pixel
    images_matrix = []
    degrees = 0

    for file in os.listdir(path):
         # get the values for is chestnut or not
        if "chestnut" not in file: isChestnut = 0
        else: isChestnut = 1

        while degrees <= 360:
            # get the image.
            image = Image.open(f"{path}/{file}").resize(pixel_size)
            # rotate the image
            rotated_image = image.rotate(degrees)
            rotated_image_noise = addNoise(rotated_image)
            images_matrix.append(imageToArray(rotated_image.convert('RGB'), isChestnut))
            images_matrix.append(imageToArray(rotated_image_noise.convert('RGB'), isChestnut))
            degrees += degrees_increase
        degrees = 0
    
    images_array = np.asarray(images_matrix)
    return images_array


def createMulticlassData(path: str, degrees_increase: int or float) -> np.array:
    """
    Preprocess multiclass data by assigning the classes and adding noise to the images.

        Params:
            path (str): Path to the photo's directory.
            degrees_increase (float or int): The number of degrees with which the image is rotated.
    """
    # matrix to store the images pixel
    images_matrix = []
    degrees = 0

    for file in os.listdir(path):
         # get the values for is chestnut or not
        if "chestnut" in file: target = 1
        if "dog" in file: target = 2
        if "cat" in file: target = 3
        if "bird" in file: target = 4

        while degrees <= 360:
            # get the image.
            image = Image.open(f"{path}/{file}").resize(pixel_size)
            # rotate the image
            rotated_image = image.rotate(degrees)
            rotated_image_noise = addNoise(rotated_image)
            images_matrix.append(imageToArray(rotated_image.convert('RGB'), target))
            images_matrix.append(imageToArray(rotated_image_noise.convert('RGB'), target))
            degrees += degrees_increase
        degrees = 0
    
    images_array = np.asarray(images_matrix)
    return images_array


def createTrainTestDataLinearRegression(images_array: np.array) -> set:
    """
    Creates and returns data for train and test.
    Params:
        images_array(np.array): The array of data from which train and test data is created.

    Returns:
        train_test_data(set): The set of training and test data.
    """
    x_train = [item[:1001] for item in images_array]
    y_train = [item[-1] for item in images_array]
    return x_train, y_train


def createTrainDataLogisticRegression(images_array: np.array) -> set:
    """
    Creates and returns data for training from list of image data.
    Params:
        images_array(np.array): The array of image data for testing.
    """
    x_train = [item[:-1] for item in images_array]
    y_train = [item[-1] for item in images_array]
    return x_train, y_train


