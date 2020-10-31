# Get the images

import pathlib
import os

images_root = pathlib.Path("Images")

images = {directory:[filename for filename in os.listdir(os.path.join(images_root, directory))] for directory in os.listdir(images_root) if pathlib.Path(os.path.join(images_root, directory)).is_dir()}

print("Extracted {} test images from {} directories".format(sum([len(images[directory]) for directory in images]), len(images)))


# Process the images

# The fixed preset area of the ring width in case the first version of the get_ring_area function is used instead of the dynamic contour-based approach
RING_AREA_X = 0  # Dummy testing values
RING_AREA_Y = 0

RING_AREA_WIDTH = 100
RING_AREA_HEIGHT = 100

# The lower and upper bounds for the HSV colorspace values where only the rings will be visible
HSV_RING_COLOR_BOUND_LOW = (0, 0, 0)  # More dummy values for testing
HSV_RING_COLOR_BOUND_HIGH = (10, 10, 10)

RING_PIXEL_THRESHOLD_LOW = 0  # Even more dummy values for... you guessed it: testing
RING_PIXEL_THRESHOLD_HIGH = 500

import cv2

def threshold_ring_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # We probably want the HSV colorspace for color thresholding for reasons described here: https://en.wikipedia.org/wiki/HSL_and_HSV
    return cv2.inRange(hsv_image, HSV_RING_COLOR_BOUND_LOW, HSV_RING_COLOR_BOUND_HIGH)  # Threshold the image for the constant HSV colors representing the approximate color range of the rings

def get_ring_area(thresholded):
    """Returns the cropped portion of the previously thresholded image inside the constant bounderies"""
    
    return thresholded[RING_AREA_Y:RING_AREA_Y+RING_AREA_HEIGHT, RING_AREA_X:RING_AREA_X+RING_AREA_WIDTH]  # Use NumPy slicing to crop the image

def get_ring_area_version_2(image, thresholded):
    """Takes the image and a thresholded version of the image and returns the first image cropped to the computed bounding box around the largest contour matching the constant HSV color range for the rings"""
    
    _, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour = sorted(contours, key=cv2.contourArea)[0]  # This assumes a contiguous region for the thresholded ring colors, which should be reasonable
    box = cv2.boundingRect(contour)
    
    return image[box[0]:box[0]+box[2], box[1]:box[1]+box[3]]  # Same cropping method as get_ring_area()

def count_pixels(thresholded):  # Note that the pixel count could also be a ratio of ring pixels to other pixels in the selected area. This may be more or less accurate
    """Takes a thresholded version of the image cropped to just the rings and counts the number of white pixels (those which matched the inRange threshold of the previous function)"""
    
    #thresholded_bgr = cv2.cvtColor(thresholded, cv2.COLOR_HSV2BGR)  # Probably deprecated. Don't uncomment

    return thresholded[:, 2].tolist().count(255)  # Any white pixel in the mask is a pixel which matched the inRange threshold of the first function

def count_rings(pixel_count):
    """Takes a count of the number of pixels the rings occupy in the image (calculated previously) and returns an estimated number of rings based off that number"""

    if pixel_count <= RING_PIXEL_THRESHOLD_LOW:  # Almost no pixels = probably no rings
        return 0
    elif pixel_count > RING_PIXEL_THRESHOLD_LOW and pixel_count < RING_PIXEL_THRESHOLD_HIGH:  # Some pixels but not many = probably 1 ring
        return 1
    else:  # Must be above RING_PIXEL_THRESHOLD_HIGH; many pixels = probably 3 rings
        return 3

def count_rings_from_image(image, ring_area_version=1):
    """Takes an image and uses all functions defined above to return the estimated number of rings in the image"""

    assert ring_area_version in [1, 2]

    return count_rings(count_pixels(get_ring_area(threshold_ring_colors(image))))

ring_counts = []

for directory in images:
    for image in images[directory]:
        print("Counting rings from image '{}' (directory '{}')".format(image, directory))
        cv_image = cv2.imread(os.path.join(images_root, directory, image))
        ring_counts.append(count_rings_from_image(cv_image))
        print("    Found {} rings".format(ring_counts[-1]))
        cv2.imshow("Image '{}' (found {} rings)".format(image, ring_counts[-1]), cv_image)

cv2.waitKey()
