# Get the images

import pathlib
import os

images_root = pathlib.Path("Images")

images = {directory:[filename for filename in os.listdir(os.path.join(images_root, directory))] for directory in os.listdir(images_root) if pathlib.Path(os.path.join(images_root, directory)).is_dir()}

print("Extracted {} test images from {} directories".format(sum([len(images[directory]) for directory in images]), len(images)))


# Process the images

AVERAGE_WALL_IMAGE_AREA = 100  # The average area, in pixels, an image on the field wall will take up. Currently set to a dummy value
WALL_IMAGE_AREA_THRESHOLD = 10  # The amount of area, in pixels, a wall image may differentiate from the average wall image area while still being considered as a wall image. Also set to a dummy value for now

RECTANGLEISH_TIE_BREAK = 0  # The ID for the tie break which considers how parallel the sides of the rectangle are
AREA_TIE_BREAK = 1  # The ID for the tie break which considers which rectangle candidate best matched the expected area defined above

tie_break_priority = [RECTANGLISH_TIE_BREAK,  # In what order ties should be broken, sorted by pritority (if all ties are broken by the first criteria, the second is ignored)
                      AREA_TIE_BREAK]

OPPOSITE_RECTANGLE_SIDE_DIFFERENCE_THRESHOLD = 5  # The maximum amount (supposedly in pixels) two slopes can be different from each other while still being considered parallel (and thus opposite sides of the rectangle)


import cv2
import warnings

def threshold_wall_photo_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # We probably want the HSV colorspace for color thresholding for reasons described here: https://en.wikipedia.org/wiki/HSL_and_HSV
    return cv2.inRange(hsv_image, HSV_WALL_PHOTO_COLOR_BOUND_LOW, HSV_WALL_PHOTO_COLOR_BOUND_HIGH)  # Threshold the image for the constant HSV colors representing the approximate color range of the WALL_PHOTOs

def get_orientiation_from_wall_photo(image):
    """Takes an image of a wall photo from the robot camera and returns either the estimated orientation of the robot given the difference in slopes between the sides of the contour enclosing the wall photo, or None if no such positive match was made"""

    edges = cv2.Canny(cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17), 30, 200)  # FIXME: Maybe add all these integer values as arguments?
    
    _, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea)[:10]  # FIXME: This value too

    candidates = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.015*perimeter, True)  # FIXME: Also this 0.015

        if len(approximation) == 4:  # Our wall photo will have 4 points, as it is a rectangle
            candidates.append(contour)  # Add it to the list of contours which might be our wall image

    prewarning = False

    if len(candidates) == 1:
        prewarning = True
    elif len(candidates) == 0:
        warnings.warn("No 4-point contours were detected. Returning None")
        return

    candidates_2 = []

    for contour in candidates:
        difference = abs(cv2.contourArea(contour)-AVERAGE_WALL_IMAGE_AREA) > WALL_IMAGE_AREA_THRESHOLD  # Our rectangle should also encompass a certain specified range of areas
        
        if difference:
            candidates_2.append((contour, difference))  # Append the difference along with the contour so the list can later be sorted based on this criteria
        elif prewarning:
            warnings.warn("The only detected 4-point contour failed the area threshold test")

    # TODO (?): More criteria could be added here, such as a position test to make sure the contour isn't in a place it couldn't be under normal circumstances (such as on the top of the image)

    # FIXME: The tie break priority represents more of the singular method used to determine which contour is chosen rather than a priority since there's no way to tell whether the tie remained after the new sorting method was applied
    if tie_break_priority[0] == RECTANGLISH_TIE_BREAK:  # Break ties based on parallel sides
        candidates_2.sort(key=lambda x: abs(cv2.contourArea(x[0])-cv2.contourArea(cv2.minAreaRect(x[0]).points())))  # If a contour is more similar to a rectangle than another, the rotated rectangle with the minimum area which encloses it will have an less of a difference in area from the original contour than the other contour does
        contour = candidates_2[0]  # Choose the more rectangle-ey contour
    elif tie_break_priority[0] == AREA_TIE_BREAK:  # Break ties based on the previously calculated area
        candidates_2.sort(key=lambda x: x[1], reverse=True)
        contour = candidates_2[0]  # Choose the contour most conforming to the expected area
    else:
        raise NotImplementedError("Unknown tie breaking method: {}".format(tie_break_priority[0]))

    slopes_and_distances = [((y2-y1)/(x2-x1), sqrt((x2-x1)**2 + (y2-y1)**2)) if (x2-x1) != 0 else float("inf") for (x2, y2) in contour for (x1, y1) in contour]  # Extract the slopes of the lines, grouped with their distances
    slopes = [slope for slope, distance in slopes_and_distances if distance not in [distance_ for slope_, distance_ in slopes_and_distance][1::-1]]  # Remove the slopes which were extracted from the points which formed the line segment with the longest length, which were most likely the diagonal segments

    # FIXME: Perhaps slopes should be compared by division rather than subtraction
    sides = [(slope_1, slope_2) for slope_1 in slopes for slope_2 in slopes if abs(slope_1-slope_2) < OPPOSITE_RECTANGLE_SIDE_DIFFERENCE_THRESHOLD]  # Group them by opposite sides, which are likely those with slopes within a few pixels of each other (otherwise our rectangle is very rhombus-like and was probably a false-positive)

    # TODO: Somehow take the difference or quotient of the opposite sides' slopes to derive a rotation, and maybe combine this with the position of the contour relative to the image center to determine a relatively exact location

for directory in images:
    for image in images[directory]:
        print("Finding wall photo in image '{}' (directory '{}')".format(image, directory))
        cv_image = cv2.imread(os.path.join(images_root, directory, image))
        cv2.imshow("Image '{}' (found {} wall photos)".format(image, wall_photo_counts[-1]), cv_image)

cv2.waitKey()
