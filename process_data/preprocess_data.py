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