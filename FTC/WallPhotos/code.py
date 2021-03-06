# Get the images

import pathlib
import os
import sys

sys.setrecursionlimit(100000000)

images_root = pathlib.Path("Images")

images = {directory:[filename for filename in os.listdir(os.path.join(images_root, directory))] for directory in os.listdir(images_root) if pathlib.Path(os.path.join(images_root, directory)).is_dir()}

print("Extracted {} test images from {} directories".format(sum([len(images[directory]) for directory in images]), len(images)))


# Process the images

AVERAGE_WALL_IMAGE_AREA = 100  # The average area, in pixels, an image on the field wall will take up. Currently set to a dummy value
WALL_IMAGE_AREA_THRESHOLD = 10  # The amount of area, in pixels, a wall image may differentiate from the average wall image area while still being considered as a wall image. Also set to a dummy value for now

RECTANGLEISH_TIE_BREAK = 0  # The ID for the tie break which considers how parallel the sides of the rectangle are
AREA_TIE_BREAK = 1  # The ID for the tie break which considers which rectangle candidate best matched the expected area defined above

tie_break_priority = [RECTANGLEISH_TIE_BREAK,  # In what order ties should be broken, sorted by pritority (if all ties are broken by the first criteria, the second is ignored)
                      AREA_TIE_BREAK]

DEFAULT_EPSILON_FACTOR = 0.015  # The default value for the factor multiplied by the perimeter of a rectangle candidate contour to find a good epsilon value
NO_CANDIDATES_EPSILON_FACTOR_INCREMENT = 0.01  # The value added repeatedly to the default epsilon factor until at least one candidate is found with the correct number of approximated points
NO_CANDIDATES_EPSILON_FACTOR_MAXIMUM = 0.025  # The maximum epsilon factor tested in an attempt to detect the correct number of points before giving up (should be 1, is 0 for debug)

OPPOSITE_RECTANGLE_SIDE_DIFFERENCE_THRESHOLD = 5  # The maximum amount (supposedly in pixels) two slopes can be different from each other while still being considered parallel (and thus opposite sides of the rectangle)


import cv2
import warnings
import imutils
import math
import numpy as np

def segment_distance(segment_1, segment_2):
    """Convenience function to measure the distance between the medians of two segments_slopes_lengths triplets"""

    #medians = (((segment_1[0][0][0]+segment_1[0][1][0])/2, (segment_1[0][0][1]+segment_1[0][1][1])/2),  # The median is the average of the values of each axis
    #            ((segment_2[0][0][0]+segment_1[0][1][0])/2, (segment_1[0][0][1]+segment_1[0][1][1])/2))

    #return math.sqrt((medians[0][0]-medians[0][1])**2+(medians[1][0]-medians[1][1])**2)

    lines = [segment_1[0], segment_2[0]]

    #distances = [math.sqrt((point_1[0]-point_2[0])**2, (point_1[1]-point_2[1])**2) for point_1, point_2 in line for line in lines]
    distances = []

    for line in lines:
        point_1, point_2 = line
        distances.append(math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2))

    return max(distances)  # Return the greatest distance between the segments

def slope_distance(segment_1, segment_2):
    """Convenience function to return the difference between the slopes of two segments_slopes_lengths triplets"""

    return abs(segment_1[1]-segment_2[1])

def renest_points(points):
    """Convenience funtion to nest all points within the given array in arrays only containing that point. OpenCV needs this for a UMat argument when a rectangle returned from boxPoints is used as a contour"""

    return np.array([[point] for point in points]).astype(np.float32)

def unnest_points(points):
    """Convenience function to undo the double nesting the previous function applies for manually doing math with points in contours"""

    return [tuple(point) for point in np.array([point[0] if type(point) != bool else point for point in points]).tolist()]

def threshold_wall_photo_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # We probably want the HSV colorspace for color thresholding for reasons described here: https://en.wikipedia.org/wiki/HSL_and_HSV
    return cv2.inRange(hsv_image, HSV_WALL_PHOTO_COLOR_BOUND_LOW, HSV_WALL_PHOTO_COLOR_BOUND_HIGH)  # Threshold the image for the constant HSV colors representing the approximate color range of the WALL_PHOTOs

def get_orientation_from_wall_photo(image, image_name, epsilon_factor=DEFAULT_EPSILON_FACTOR):
    """Takes an image of a wall photo from the robot camera and returns either the estimated orientation of the robot given the difference in slopes between the sides of the contour enclosing the wall photo, or None if no such positive match was made"""

    #image = imutils.resize(image, height=50)  # Rescale to help accuracy when dealing with round-looking corners
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    thresholded = cv2.inRange(hsv, (0, 0, 100), (180, 60, 255), cv2.THRESH_BINARY)
    thresholded = cv2.dilate(thresholded, None, iterations=1)

    #edges = cv2.Canny(cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17), 30, 200)  # FIXME: Maybe add all these integer values as arguments?
    edges = cv2.Canny(cv2.bilateralFilter(thresholded, 11, 17, 17), 30, 200)  # FIXME: Maybe add all these integer values as arguments?

    #edges = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)

    #edges = thresholded
    
    cv2.imwrite("Output/Edges ({}).jpg".format(image_name), edges)
    
    contours = imutils.grab_contours(cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # FIXME: This value too
    contours = [cv2.convexHull(contour) for contour in contours]  # Grab the convex hulls of the contours
    contours = [contour for contour in contours if cv2.contourArea(contour) > 9000]  # Filter out very small contours

    candidates = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, perimeter*epsilon_factor, True)  # FIXME: Also this 0.015

        if len(approximation) == 4:  # Our wall photo will have 4 points, as it is a rectangle
            candidates.append(approximation)  # Add it to the list of contours which might be our wall image

    cv2.drawContours(image, contours, -1, (0, 0, 0), 3)

    prewarning = False

    if len(candidates) == 1:
        prewarning = True
    elif len(candidates) == 0:
        if epsilon_factor + NO_CANDIDATES_EPSILON_FACTOR_INCREMENT > NO_CANDIDATES_EPSILON_FACTOR_MAXIMUM:  # This epsilon exceeds our maximum; give up
            warnings.warn("No 4-point contours were detected after retrying with at least one higher epsilon value, exceeding the maximum; giving up and returning None")
            return None
        
        epsilon_factor += NO_CANDIDATES_EPSILON_FACTOR_INCREMENT
        
        warnings.warn("No 4-point contours were detected. Retrying with a higher epsilon factor: {}".format(epsilon_factor))
        return get_orientation_from_wall_photo(image, image_name, epsilon_factor=epsilon_factor)

    candidates_2 = []

    for contour in candidates:
        difference = abs(cv2.contourArea(contour)-AVERAGE_WALL_IMAGE_AREA) > WALL_IMAGE_AREA_THRESHOLD  # Our rectangle should also encompass a certain specified range of areas
        
        if difference:
            candidates_2.append((contour, difference))  # Append the difference along with the contour so the list can later be sorted based on this criteria
        elif prewarning:
            warnings.warn("The only detected 4-point contour failed the area threshold test")

    # TODO (?): More criteria could be added here, such as a position test to make sure the contour isn't in a place it couldn't be under normal circumstances (such as on the top of the image)

    # FIXME: The tie break priority represents more of the singular method used to determine which contour is chosen rather than a priority since there's currently no way to tell whether the tie remained after the new sorting method was applied
    if tie_break_priority[0] == RECTANGLEISH_TIE_BREAK:  # Break ties based on parallel sides
        #candidates_2.sort(key=lambda x: abs(cv2.contourArea(x[0])-cv2.contourArea(cv2.minAreaRect(x[0]).points())))  # If a contour is more similar to a rectangle than another, the rotated rectangle with the minimum area which encloses it will have an less of a difference in area from the original contour than the other contour does
        candidates_2.sort(key=lambda x: abs(cv2.contourArea(x[0]) / cv2.contourArea(renest_points(cv2.boxPoints(cv2.minAreaRect(x[0]))))), reverse=True)

        for x in candidates_2:
            points = cv2.boxPoints(cv2.minAreaRect(x[0])).astype(np.int0)
            cv2.rectangle(image, tuple(points[0]), tuple(points[2]), (0, 255, 0))  # Debug
        contour = candidates_2[0]  # Choose the more rectangle-ey contour
    elif tie_break_priority[0] == AREA_TIE_BREAK:  # Break ties based on the previously calculated area difference
        candidates_2.sort(key=lambda x: x[1], reverse=True)
        contour = candidates_2[0]  # Choose the contour most conforming to the expected area
    else:
        raise NotImplementedError("Unknown tie breaking method: {}".format(tie_break_priority[0]))

    cv2.drawContours(image, np.array(candidates)[:, 0], -1, (255, 0, 0), 3)
    cv2.drawContours(image, np.array(candidates_2)[:, 0], -1, (0, 0, 255), 3)
    cv2.drawContours(image, [contour[0]], -1, (0, 255, 0), 3)

    print("Area of {}: {}".format(image_name, cv2.contourArea(contour[0])))
    #cv2.imwrite("Output/Contours ({})".format(image_name), image)  # This line moved to later for debugging

    unnested_contour = unnest_points(contour[0])

    segments = [tuple(sorted((point_1, point_2))) for point_1 in unnested_contour \
                for point_2 in unnested_contour if point_1 != point_2]  # Collect groupings of points from the contour as segments
    segments = list(set(segments))  # Remove duplicates

    #print(segments)
    
    segments_slopes_lengths = [(tuple(sorted(((x1, y1), (x2, y2)))), (y2-y1)/(x2-x1), math.sqrt((x2-x1)**2 + (y2-y1)**2)) \
                               if x2-x1 != 0 else (tuple(sorted(((x1, y1), (x2, y2)))), float("inf"), math.sqrt((x2-x1)**2 + (y2-y1)**2)) \
                               for ((x1, y1), (x2, y2)) in segments if (x1, y1) != (x2, y2)]  # Group each segment with its corresponding slope and length
    
    segments_slopes_lengths = list(set(segments_slopes_lengths))  # Remove duplicates
    
    segments_slopes_lengths.sort(key=lambda s: s[2])  # Sort by largest length

    segments_slopes_lengths.pop(); segments_slopes_lengths.pop()  # Remove the longest 2 segments, which are likely the diagonals

    #print(len(segments), len(segments_slopes_lengths))
    #print(len(segments_slopes_lengths))
    
    #sides = [tuple(sorted([(segment_1, slope_1, length_1), (segment_2, slope_2, length_2)])) for (segment_1, slope_1, length_1) in segments_slopes_lengths \
    #         for (segment_2, slope_2, length_2) in segments_slopes_lengths if abs(slope_1-slope_2) < OPPOSITE_RECTANGLE_SIDE_DIFFERENCE_THRESHOLD \
    #         and segment_1 != segment_2]  # Group segments by opposite side. Segments opposite each other will likely appear almost or exactly parallel

##    for (segment_1, slope_1, length_1) in segments_slopes_lengths:  # TODO: Redo this by a tie-breaking method so that more parallel lines are grouped and no doubles are accidnetally formed
##        for (segment_2, slope_2, length_2) in segments_slopes_lengths:
##            if segment_1 == segment_2 or segment_1[0] == segment_2[0] or segment_1[1] == segment_2[1]:  # Ensure points are unique within a pairing
##                continue
##
##            continuing = False
##            
##            for pair in sides: # Ensure segments are unique within the resulting list
##                if segment_1 in pair or segment_2 in pair:
##                    continuing = True
##
##            if continuing:
##                continue
##
##            s = set([(segment_1, slope_1, length_1), (segment_2, slope_2, length_2)])
##            
##            if s not in sides:
##                sides.append(s)

    # Side groupings should be formed from the furthest segments; sides will be arbitrarily grouped, then resorted if necessary

    assert len(segments_slopes_lengths) == 4  # Requried criteria

    sides = [(segments_slopes_lengths[0], segments_slopes_lengths[2]),  # One of the two possible combinations of segment groupings
             (segments_slopes_lengths[1], segments_slopes_lengths[3])]

    #  Ensure that grouped sides are those furthest from each other
    #if segment_distance(sides[0][0], sides[0][1]) < segment_distance(sides[1][0], sides[1][1]):
    #    sides = [(segments_slopes_lengths[0], segments_slopes_lengths[2]),  # Use the other possible grouping
    #             (segments_slopes_lengths[1], segments_slopes_lengths[3])]

    #  Ensure that grouped sides are those with the least similar slope (so the difference in slopes of the sides we consider to be opposite is greater than that of the other grouping)
    

    for segment_1, segment_2 in sides:
        line_1, line_2 = segment_1[0], segment_2[0]

        cv2.line(image, line_1[0], line_1[1], (0, 0, 255), 5)
        cv2.line(image, line_2[0], line_2[1], (255, 0, 0), 5)

    cv2.imwrite("Output/Contours ({}).jpg".format(image_name), image)

    print(len(sides))
    
##    slopes_and_lengths = [((y2-y1)/(x2-x1), math.sqrt((x2-x1)**2 + (y2-y1)**2)) if (x2-x1) != 0 else (float("inf"), math.sqrt((x2-x1)**2 + (y2-y1)**2)) for (x2, y2) in unnested_contour for (x1, y1) in unnested_contour]  # Extract the lengths of the segments, grouped with their slopes. This should generate 6 lines (including 2 diagonals of the rectangle) after duplicates are removed
##    slopes_and_lengths = list(set(slopes_and_lengths))  # Remove duplicates
##
##    if len(slopes_and_lengths) != 6:
##        warnings.warn("Got unexpected number of segments in rectangle: {} (expected 6)".format(len(slopes_and_lengths)))
##    
##    #print(slopes_and_lengths)
##    slopes_and_lengths = [(slope, distance) for slope, distance in slopes_and_lengths if distance not in [distance_ for slope_, distance_ in slopes_and_lengths][1::-1]]  # Remove the 2 segments which were extracted from the points which formed the line segment with the longest length, which were most likely the diagonal segments
##
##    #print(slopes_and_lengths)
##
##    sides = [(slope_and_length_1[1], slope_and_length_2[1]) for slope_and_length_1 in slopes_and_lengths for slope_and_length_2 in slopes_and_lengths if abs(slope_and_length_1[0]-slope_and_length_2[0]) < OPPOSITE_RECTANGLE_SIDE_DIFFERENCE_THRESHOLD and slope_and_length_1 != slope_and_length_2]  # Group the distances by opposite sides, which are likely those with slopes within a few pixels of each other (otherwise our rectangle is very rhombus-like and was probably a false-positive)
##
##    if len(sides) != 4:
##        warnings.warn("Got unexpected number of sides in rectangle: {} (expected 4)".format(len(sides)))
##
##    print("Found {} points, {} sides, {} slopes/lengths".format(len(unnested_contour), len(sides), len(slopes_and_lengths)))

    #print(len(sides))

    for (segment_slope_length_1, segment_slope_length_2) in sides:
        length_ratio = segment_slope_length_1[2]/segment_slope_length_2[2]
        #print(length_ratio)

    # TODO: Somehow take the difference or quotient of the opposite sides' slopes to derive a rotation, and maybe combine this with the position of the contour relative to the image center to determine a relatively exact location

for directory in images:
    for image in images[directory]:
        #print("Getting orientation from wall photo in image '{}' (directory '{}')".format(image, directory))
        cv_image = cv2.imread(os.path.join(images_root, directory, image))
        orientation = get_orientation_from_wall_photo(cv_image, image)

        #print("    Got orientation {}".format(orientation))
        
        #cv2.imshow("Image '{}' (computed orientation: {})".format(image, orientation), cv_image)

cv2.waitKey()
