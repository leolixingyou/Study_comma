import os
import re
import cv2
import glob
import argparse
import numpy as np
from pathlib import Path
 
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def canny_edge_detection(img, low_threshold, high_threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_detected_lanes(img, left_coeffs, right_coeffs):

    # Generate y values
    y = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    # Generate x values for left and right lanes
    left_x = left_coeffs[0]*y**2 + left_coeffs[1]*y + left_coeffs[2]
    right_x = right_coeffs[0]*y**2 + right_coeffs[1]*y + right_coeffs[2]
    
    # Create an image to draw the lanes
    lane_img = np.zeros_like(img)
    
    # Draw the left and right lanes
    for idx in range(1, len(y)):
        cv2.line(lane_img, (int(left_x[idx-1]), int(y[idx-1])), (int(left_x[idx]), int(y[idx])), (0, 255, 0), 2)
        cv2.line(lane_img, (int(right_x[idx-1]), int(y[idx-1])), (int(right_x[idx]), int(y[idx])), (0, 0, 255), 2)
    
    # Combine the lane image with the original image
    result = cv2.addWeighted(img, 1, lane_img, 0.8, 0)
    return result


def calculate_offset(left_coeffs, right_coeffs, h, w):
    lane_center = (left_coeffs[0]*h**2 + left_coeffs[1]*h + left_coeffs[2] +
                   right_coeffs[0]*h**2 + right_coeffs[1]*h + right_coeffs[2]) / 2
    img_center = w / 2
    offset = img_center - lane_center
    xm_per_pix = 3.7/700  # Assuming lane width is 3.7 meters and image width is 700 pixels
    offset_meters = offset * xm_per_pix
    return offset_meters

def lane_departure_warning(offset_meters):
    print(f'Mid Bias is :{offset_meters}!')
    if abs(offset_meters) > 0.5:
        print('Warning: Lane Departure Detected!')
    
def lane_keeping_assist(img, left_coeffs, right_coeffs):
    h, w = img.shape[:2]
    y_eval = h - 1  # Evaluate at the bottom of the image
    lane_center = (left_coeffs[0]*y_eval**2 + left_coeffs[1]*y_eval + left_coeffs[2] +
                   right_coeffs[0]*y_eval**2 + right_coeffs[1]*y_eval + right_coeffs[2]) / 2
    img_center = w / 2
    steering_angle = np.arctan((lane_center - img_center) / h)  # Simplified steering angle calculation
    print(f'Steering Angle: {np.degrees(steering_angle)} degrees')
    return steering_angle
    
def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic

    dirs = glob.glob(f"{path}{sep}*")  # similar paths
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]  # indices
    n = max(i) + 1 if i else 0  # increment number
    return f"{path}{sep}{n}"  # update path

def arguments_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', nargs='+', type=str, default='./temp.jpg', help='input image')
    parser.add_argument('--raw_img', nargs='+', type=str, default='./image.png', help='input image')
    parser.add_argument('--project', default='runs/results', help='save results to project/name')
    parser.add_argument('--subject', default='line lr_line curve full_draw bev', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    opt = parser.parse_args()
    return opt

def get_list(path, file_type):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in file_type:
                image_names.append(apath)
    return image_names

def get_Ms(src, dst):
    M_per = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M_per, M_inv

def warp_image(img, M, size):
    # Compute the perspective transform, M
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, size)
    return warped

def threshold_image(warped_img):
    # Convert to grayscale
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return binary


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    for idx in range(binary_warped.shape[0]):
        cv2.circle(out_img, (int(left_fitx[idx]), int(ploty[idx])), 3, (0,255,0), -1)
        cv2.circle(out_img, (int(right_fitx[idx]), int(ploty[idx])), 3, (0,255,0), -1)

    return out_img, left_fit, right_fit

def calculate_curvature_and_offset(binary_warped, left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.5/(1047-235) # meters per pixel in x dimension

    # Maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    center_fit = [(left_fit[i] + right_fit[i]) / 2 for i in range(3)]
    # 计算中心线的曲率半径
    curvature = ((1 + (2*center_fit[0]*y_eval + center_fit[1])**2)**1.5) / np.absolute(2*center_fit[0])


    # Calculation for the offset
    car_position = binary_warped.shape[1]/2
    left_lane_bottom_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_lane_bottom_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_lane_bottom_x + right_lane_bottom_x) / 2
    offset = (car_position - lane_center) * xm_per_pix
    center_bottom_point = [lane_center,y_eval] 

    ### wheelbase
    steering_angle_rad = np.arctan(3 / curvature)
    return steering_angle_rad, offset, center_bottom_point, center_fit

def draw_lane_lines(original_image, left_fit, right_fit, center_fit, M_per, M_inv):
    try:
        original_image = cv2.cvtColor(original_image,cv2.COLOR_GRAY2BGR)
    except:
        pass
    ploty = np.linspace(0, original_image.shape[0]-1, original_image.shape[0])

    # 计算左右车道线x坐标
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    center_fitx = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]

    # 绘制车道线
    for i in range(ploty.shape[0]):
        cv2.circle(original_image, (int(left_fitx[i]), int(ploty[i])), 3, (255,0,0), -1)
        cv2.circle(original_image, (int(right_fitx[i]), int(ploty[i])), 3, (0,0,255), -1)
        cv2.circle(original_image, (int(center_fitx[i]), int(ploty[i])), 3, (0,255,0), -1)
    return original_image





