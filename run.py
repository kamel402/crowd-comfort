import cv2
import numpy as np
from camera import VideoCam
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="test-comfort.mp4", type=str,
	help="path to the (optional) video file")
ap.add_argument("-s", "--skip", default=1, type=int,
	help="number of frame to be skipped")
args = vars(ap.parse_args())

 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
THICKNESS = 1
 
# Colors.
BLUE   = (255,178,50)
YELLOW = (0,255,255)

# Blue rectangle coordinates
START_POINT = (1066, 188)
END_POINT = (1996, 1360)

# Crowd states
COMFORT_STATES = ['Comfortable', 'Normal', 'Crowded']

def find_masked_img(img, range):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Range for lower red
    # green [38, 163, 0, 121, 255, 255], [36, 164, 0, 108, 255, 255]
    # yellow [19, 177, 187, 53, 255, 255]
    lower_red = np.array(range[0])
    upper_red = np.array(range[1])

    # lower_white = np.array([0,70,0])
    # upper_white = np.array([30,255,255])

    mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170,70,0])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv_img,lower_red,upper_red)

    # mask for lower and upper red
    mask = mask1#+mask2

    # Get image in red pixel only
    result = cv2.bitwise_and(img.copy(), img.copy(), mask=mask)
    return result

def find_thresh(masked_img):
    red = masked_img.copy()
    # set blue and green channels to 0
    red[:, :, 0] = 0
    red[:, :, 1] = 0

    # otsu threshhold for red image
    gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh = cv2.threshold(blured,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def find_closing(thresh, kernel):
    kernel = np.ones((9,9),np.uint8)
    big = thresh.copy()
    closing = cv2.morphologyEx(big, cv2.MORPH_CLOSE, kernel)
    return closing

def find_negative(closing):
    negative = cv2.bitwise_not(closing) # negative image
    contours, hierarchy = cv2.findContours(negative, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    hierarchy = hierarchy[0]
    max_area = cv2.contourArea(contours[0])
    total = 0 # total contour size
    for con in contours:
        area = cv2.contourArea(con) # get contour size
        total += area
        if area > max_area:
            max_area = area

    diff = 0.1 # smallest contour have to bigger than (diff * max_area)
    max_area = int(max_area * diff) # smallest contour have to bigger than max_area

    average = int(total / (len(contours))) # average size for contour

    average = int(average * diff) # average area size
    return negative, contours, hierarchy, average

def find_mask(negative, contours, hierarchy, average):
    mask = np.zeros(negative.shape[:2],dtype=np.uint8)
    #  For each contour, find the bounding rectangle and draw it
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        area = cv2.contourArea(currentContour)
 
        if currentHierarchy[3] < 0: # if currentHirerarcy[3] = -1, it means the countour DON'T have parent, shouldn't remove it
            if area > average:
                #  If contour don't have parent AND contour size > average size, draw it into mask
                cv2.drawContours(mask, [currentContour], 0, (255), -1)  
    return mask

def find_sure_fg(mask, kernel):
    sure_bg = cv2.erode(mask, kernel)
    dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0) # 0.2 is important, the bigger it is, the object is smaller (to the object center)
    sure_fg = np.uint8(sure_fg)
    return sure_fg

def draw(sure_fg, img, count):
    #Find contour for sure figure
    contours, hierarchy = cv2.findContours(sure_fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    contours_poly = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    totalRadius = 0
    for i in range(len(contours)):
        totalRadius += radius[i]
        averageRadius = totalRadius / len(contours)
        diff_average_radius = 0.3
    
    people_inside = 0

    for i in range(len(contours)):
        if radius[i] > averageRadius * diff_average_radius:
            if is_inside([START_POINT[0], START_POINT[1], END_POINT[0], END_POINT[1]], [int(centers[i][0]), int(centers[i][1])]):
                people_inside += 1
                count += 1
                cv2.circle(result, (int(centers[i][0]), int(centers[i][1])), 15, color=(0, 255, 0), thickness=-1)
                #cv2.circle(result, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2) # Draw circle
                cv2.putText(result, str(count), (int(centers[i][0]), int(centers[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # Put text

    return result, people_inside

def processing(img, range, count=0):
    kernel = np.ones((5,5),np.uint8)
    masked_img = find_masked_img(img, range)
    thresh = find_thresh(masked_img)
    closing = find_closing(thresh, kernel)
    negative, contours, hierarchy, average = find_negative(closing)
    mask = find_mask(negative, contours, hierarchy, average)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sure_fg = find_sure_fg(mask, kernel)
    frame, count = draw(sure_fg, img, count)

    return frame, count

def draw_rec(frame):
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 3 px
    thickness = 3
    frame = cv2.rectangle(frame, START_POINT, END_POINT, color, thickness)
    return frame

def is_inside(obj1, obj2):
    a, b, c, d = obj1[0], obj1[1], obj1[2], obj1[3]

    x, y = obj2[0], obj2[1]

    if a < x and x < a+c and b < y and y < b + d:
        return True
    else:
        return False

if __name__ == '__main__':
    
    # Get the video
    source = args["video"]
    

    # Initiate people counter
    people_counter = 0
    SKIPFRAME = args['skip']

    # Infernce 
    cap = VideoCam(source)
    cap.check_camera(cap.cap)
    ct = 0
    while True:
        ct += 1
        try:
            ret = cap.cap.grab()
            if ct % SKIPFRAME == 0:  # skip some frames
                ret, frame = cap.get_frame()
                if not ret:
                    cap.restart_capture(cap.cap)
                    cap.check_camera(cap.cap)
                    continue

                # Process frame.

                # Draw the blue rectangle
                frame = draw_rec(frame)

                # Color range of a yellow helmet
                y_range = [[0, 237, 0], [255,255,255]]
                try:
                    # Process people with yellow helmet
                    y_frame, y_people_counter = processing(frame, y_range)
                except:
                    y_frame, y_people_counter = frame, 0

                # Color range of a green helmet
                g_range = [[38, 163,0], [121,255,255]]
                try:
                    # Process people with green helmet
                    g_frame, g_people_counter = processing(frame, g_range, y_people_counter)
                except:
                    g_frame, g_people_counter = frame, 0

                
                # Add number of people with blue and yellow helmets
                people_counter = y_people_counter + g_people_counter
                
                # Merge two processed frame
                frame = cv2.addWeighted(y_frame,0.5, g_frame,0.5, 0)

                # People count label
                label = 'People Count: '+str(people_counter)
                cv2.putText(frame, label, (165, 98), FONT_FACE, 1,  (0, 0, 255), 2, cv2.LINE_AA)

                # Calculate crowd comfort
                if people_counter < 30:
                    state = 'State: '+COMFORT_STATES[0]
                    state_color = (0, 255, 0)
                elif people_counter >= 30 and  people_counter < 40:
                    state = 'State: '+COMFORT_STATES[1]
                    state_color = (255,165,0)
                else:
                    state = 'State: '+COMFORT_STATES[2]
                    state_color = (0,0,255)

                # Crowd state label
                cv2.putText(frame, state, (165, 150), FONT_FACE, 1,  state_color, 2, cv2.LINE_AA)

                # Show the result
                cap.show_frame(frame, 'frame')
                key = cv2.waitKey(1) & 0xFF
                # if the 'q' key is pressed, stop the loop
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            cap.close_cam()
            exit(0)

