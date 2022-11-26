import cv2
import numpy as np

def draw_the_lines(image, lines):
    #  create a distinct image for lines
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # there are (x,y) for the starting adn end points of the lines
    for line in lines:
        for x1, y1, x2, y2 in line: 
            cv2.line(lines_image, (x1,y1), (x2,y2), (255,0,0), thickness=3) # BGR

    # finally we have to merge the image with lines
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)

    return image_with_lines



def region_of_interest(image, region_points):

    # we are going to replace pixels with 0(black) - the regions we are not interested
    mask = np.zeros_like(image)
    # the region we are interested in is the lower traingle - 255 white pixels
    cv2.fillPoly(mask, region_points, 255)

    # we have to use the mask: we want to keep the regions of the original image where the mask has white colored pixels
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])
    # print(width)
    # we have to turn the image into grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection kernel (Canny's algorithm)
    canny_image = cv2.Canny(gray_image, 100, 120)    # 100 - lower threshold  120 -higher threshold

    # we are interested in the lower region of images (there are the driving lanes)
    region_of_interest_vertices = [
        (0, height), # bottom left corner of the image
        (width/2, height*0.65), # center of the image, i.e. height little below thaN center
        (width, height)    #  bottom right corner of the image
    ]

    # we can rid of the unrelavent part of the image, we just keep  the lower image
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # use the line detection algorithm (radian instead of degrees 1 degree = pi/180)
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=150)

    # draw the lines on the image
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


# video = several frames (images shown after each other)
video = cv2.VideoCapture('lane_detection_video.mp4')

while video.isOpened():

    is_grabbed, frame = video.read()   # is_grabbed = boolean value

    # because the end of the video  i.e.  at the end is_grabbed is false
    if not is_grabbed:
        break

    frame = get_detected_lanes(frame)

    cv2.imshow('Lane Detection Video', frame)
    cv2.waitKey(50)

video.release()  # closes the opening video file
cv2.destroyAllWindows()