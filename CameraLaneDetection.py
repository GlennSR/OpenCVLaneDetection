import matplotlib.pyplot as plt
import cv2
import numpy as np
from Commande import SetSpeed

def region_of_interest(image, vertices):
    """Code that defines the zone of interest of the image
    and set the rest to black"""
    # Defines a blank mask to start
    mask = np.zeros_like(image)
    # channel_count = image.shape[2]
    match_mask_color = 225

    # Fills the zone inside the polygon defined by the vertices
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returns the image only where the mask pixels are non zero
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def draw_lines(image, lines):
    image = np.copy(image)
    blank_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    left_x1 = []
    left_x2 = []
    right_x1 = []
    right_x2 = []
    y_min = image.shape[0]
    y_max = int(image.shape[0] * 0.611)

    if lines is not None:
        for line in lines:
            '''This part devides the left and right lines of the lanes by the
            angular inclination of it's equation'''
            for x1, y1, x2, y2 in line:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                a = mc[0]
                b = mc[1]
                if a < 0:
                    # p1.append([line[0][0], line[0][1]])
                    left_x1.append(np.int(np.float((y_min - b)) / np.float(a)))
                    left_x2.append(np.int(np.float((y_max - b)) / np.float(a)))
                elif a > 0:
                    # p2.append([line[0][2], line[0][3]])
                    right_x1.append(np.int(np.float((y_min - b)) / np.float(a)))
                    right_x2.append(np.int(np.float((y_max - b)) / np.float(a)))

        '''Verifies if the vectors are not empty'''
        if len(left_x1) and len(left_x2) and len(right_x1) and len(right_x2):
            l_avg_x1 = np.int(np.nanmean(left_x1))
            l_avg_x2 = np.int(np.nanmean(left_x2))
            r_avg_x1 = np.int(np.nanmean(right_x1))
            r_avg_x2 = np.int(np.nanmean(right_x2))

            # Draw the lines
            cv2.line(image, (l_avg_x1, y_min), (l_avg_x2, y_max), (255, 0, 0), thickness=6)
            cv2.line(image, (r_avg_x1, y_min), (r_avg_x2, y_max), (255, 0, 0), thickness=6)
            # cv2.line(blank_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=6)

            '''This part of the code calculates the mean value of the center
            of the lanes, using the distance calculated between the equivalent left
            and right points, after that, it adds the half of this distance 
            with one of the left points to have the coordinate of the center'''
            dist1 = (r_avg_x1 - l_avg_x1) / 2
            dist2 = (r_avg_x2 - l_avg_x2) / 2
            half = ((l_avg_x1 + dist1) + (l_avg_x2 + dist2)) / 2

            # Calcul of the descentralization vector of the robot.
            # Uses the coordinate of the lane's central (half) to centralizate the robot
            if half is not None:
                Command = SetSpeed(half, width)

            cv2.line(blank_img, (int(half), height - 46), (int(width / 2), height - 46), (255, 255, 0), thickness=3)

            # Calculated central of the lanes plot
            cv2.line(blank_img, (int(half), height - 51), (int(half), height - 41), (255, 0, 0), thickness=6)

    # Central position of the camera (Direction that the robot is going)
    cv2.line(blank_img, (int(width / 2), height - 51), (int(width / 2), height - 41), (255, 255, 0), thickness=6)

    # This line joins the drawn lines with the original image
    image = cv2.addWeighted(image, 0.8, blank_img, 1, 0.0)
    return image


def canny(img):

    region_of_interest_vertices = [(0, height), (width*0.4, height/2), (width*0.6, height/2), (width, height)]
    region_of_interest_vertices_2 = [(0, 0), (width / 2, height / 2), (width, 0)]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 100, 200)

    '''Calling the function to define the zone of interest'''
    cropped_img = region_of_interest(canny_img, np.array([region_of_interest_vertices], np.int32))
    cropped_img_2 = region_of_interest(canny_img, np.array([region_of_interest_vertices_2], np.int32))
    final_cropped_img = cv2.addWeighted(cropped_img, 1, cropped_img_2, 1, 0.0)

    return final_cropped_img


def process(img):
    """Main function that calls all of the processes to do the detection
    :param img: source image to be treated"""
    # general vertices of a triangle that converge into the center of the image
    region_of_interest_vertices = [(0, height), (width*0.4, height/2), (width*0.6, height/2), (width, height)]

    '''The function cvtColor turns the image (img) into gray scale (COLOR_RGB2GRAY)
    while the function Canny applies the Canny algorithm to find the borders on the image
    (gray_img) with the two thresholds defined'''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 100, 200)

    '''Calling the function to define the zone of interest'''
    cropped_img = region_of_interest(canny_img, np.array([region_of_interest_vertices], np.int32))

    '''Hough Transform application and all the thresholds that we can change
    where rho and theta are the parameters of the line equation'''
    lines = cv2.HoughLinesP(cropped_img,
                            rho=1,
                            theta=np.pi/180,
                            threshold=32,
                            lines=np.array([]),
                            minLineLength=50,
                            maxLineGap=100)

    lined_img = draw_lines(img, lines)
    return lined_img


cap = cv2.VideoCapture('test.mp4') # If you want to connect your webcam just enter 0 in the function

while cap.isOpened():
    '''Main Programm'''
    ret, frame = cap.read()
    global height, width
    height, width = frame.shape[:2]

    edgeDetection = process(frame)
    cannyEdges = canny(frame)
    cv2.imshow('Detected Edges', edgeDetection)
    #cv2.imshow('Canny image', cannyEdges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
