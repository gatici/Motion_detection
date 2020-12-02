import sys
from detection import *

if __name__ == '__main__':


    video = str('dolu2.mp4')

    # Capture the  video
    cap = cv2.VideoCapture(video)

    # Get first frame as a previous
    _, prev_frame = cap.read()

    while True:

        # Get current frame
        hasNext, cur_frame = cap.read()

        #  If the last frame break the  operation
        if not hasNext:
            break

        # Prepare the  previous and  current  frames  before  getting difference
        res_cur_frame = f_preparation(cur_frame)
        res_prev_frame = f_preparation(prev_frame)

        # Getting difference  of  previous  and  current  frames
        difference = f_difference(res_cur_frame, res_prev_frame)

        # Show the  motion  between  consecutive  frames
        result = f_motion(difference, cur_frame, video)

        # This current frame is previous frame now
        prev_frame = cur_frame

        # Show the diffrence and indicate the  motion with  bounding box
        cv2.imshow("Difference", difference)


        cv2.imshow("Motion", result)


        cv2.waitKey(1)

