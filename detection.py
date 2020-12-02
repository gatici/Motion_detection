import cv2
import numpy as np
import parameters as var


#f_preparation function  resize  image, convert to  grayscale  and  applies to  GaussianBlur filter  to  remove  the  noise.
def f_preparation(old_img):
    # Resize the image
    new_img = cv2.resize(old_img, var.img_size)

    # Convert frames to grayscale
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    #  Smooth the  frames
    res_img = cv2.GaussianBlur(gray_img, var.blur_kernel, 0)

    return res_img


#f_difference  function first  create  a new array with the same shape and type as current frame.
#cv2.absdiff take  diffence  between  previous  and  current  frame
#threshold  is  applied to convert the  grayscale  image to  binary  image, white  area means  there is  motion.
#morhology (opening) is  applied  to get  more  smooth shapes
def f_difference(cur_frame, prev_frame):

    # Create empty frame as original frame then get difference
    difference = np.empty_like(cur_frame)

    cv2.absdiff(cur_frame, prev_frame, difference)

    # Get binary image from a gray image with binary threshold
    _, diff_tresh = cv2.threshold(difference, var.trash_val, var.max_val, cv2.THRESH_BINARY)


    # Makes the object in white bigger with dilate operation
    diff_dilation = cv2.dilate(diff_tresh, None, iterations=3)

    return diff_dilation

#f_motion  function takes the  diffrence area  and  find  and draw the  contours around  it.
#for  every  contour, contour  area is  calculated and  min_area  threshold  is  applied.
#If the  contour area  is  bigger than  min_area, a  bounding  box  is  drawn  around this area which  indicates the  motion.
#The frame  with the  selected  contour as  motion detected  are  saved  as images.
def f_motion(diff_img, src, name):
    # Find contours
    # RETR_TREE for retrieve all of the contours
    # CHAIN_APPROX_NONE for store absolutely all the contour points
    image, contours, hierarchy = cv2.findContours(diff_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draws contours outlines
    result_img = cv2.resize(src, var.img_size)


    for contour in contours:

        if cv2.contourArea(contour) > var.min_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), var.rect_color, var.rect_line)
            cv2.imwrite('output_%s/%s.jpg' %(str(name), str(contour[1])+ str(x) + '-' + str(y)), result_img)

    return result_img





