import os
import cv2
import sys
import time
import scipy
import numpy as np
import matplotlib.pyplot as plt

import utils.orientation as orient
import utils.coordinates as coord

from tools.lib.framereader import FrameReader
from utils.camera import img_from_device, denormalize, view_frame_from_device_frame

def plot_ANGLE(example_segment):
    # we can plot the orientation of the camera in 
    # euler angles respective to the local ground plane,
    # i.e. the North East Down reference frame. This is more
    # intuitive than the quaternion.
    frame_times = np.load(example_segment + 'global_pose/frame_times')
    frame_positions = np.load(example_segment + 'global_pose/frame_positions')
    frame_orientations = np.load(example_segment + 'global_pose/frame_orientations')
    euler_angles_ned_deg = (180/np.pi)*orient.ned_euler_from_ecef(frame_positions[0], orient.euler_from_quat(frame_orientations))


    plt.plot(frame_times, euler_angles_ned_deg[:,0], label='roll', linewidth=3)
    plt.plot(frame_times, euler_angles_ned_deg[:,1], label='pitch', linewidth=3)
    plt.plot(frame_times, euler_angles_ned_deg[:,2], label='yaw', linewidth=3)
    plt.title('Orientation in local frame (NED)', fontsize=25)
    plt.legend(fontsize=25)
    plt.xlabel('boot time (s)', fontsize=18)
    plt.ylabel('Euler angle (deg)', fontsize=18)
    plt.show()

def draw_frame(example_segment, img):
    frame_orientations = np.load(example_segment + 'global_pose/frame_orientations')
    frame_positions = np.load(example_segment + 'global_pose/frame_positions')
    ecef_from_local = orient.rot_from_quat(frame_orientations[0])
    local_from_ecef = ecef_from_local.T
    frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions - frame_positions[0])

    def draw_path(device_path, img, width=1, height=1.2, fill_color=(128,0,255), line_color=(0,255,0)):
        device_path_l = device_path + np.array([0, 0, height])                                                                    
        device_path_r = device_path + np.array([0, 0, height])                                                                    
        device_path_l[:,1] -= width                                                                                               
        device_path_r[:,1] += width

        img_points_norm_l = img_from_device(device_path_l)
        img_points_norm_r = img_from_device(device_path_r)
        img_pts_l = denormalize(img_points_norm_l)
        img_pts_r = denormalize(img_points_norm_r)

        # filter out things rejected along the way
        valid = np.logical_and(np.isfinite(img_pts_l).all(axis=1), np.isfinite(img_pts_r).all(axis=1))
        img_pts_l = img_pts_l[valid].astype(int)
        img_pts_r = img_pts_r[valid].astype(int)

        for i in range(1, len(img_pts_l)):
            u1,v1,u2,v2 = np.append(img_pts_l[i-1], img_pts_r[i-1])
            u3,v3,u4,v4 = np.append(img_pts_l[i], img_pts_r[i])
            pts = np.array([[u1,v1],[u2,v2],[u4,v4],[u3,v3]], np.int32).reshape((-1,1,2))
            cv2.fillPoly(img,[pts],fill_color)
            cv2.polylines(img,[pts],True,line_color)


    # img = plt.imread(example_segment + 'preview.png')
    draw_path(frame_positions_local[11:250], img)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_video(example_segment):

    frame_index = 600
    save_dir = './results/'
    mkdir(save_dir)

    fr = FrameReader(example_segment + 'video.hevc')
    for frame_index in range(0, 600):

        img = fr.get(frame_index, count=1, pix_fmt='rgb24')[0]
        draw_frame(example_segment, img)
        cv2.imwrite(f'{save_dir}video_{frame_index}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ =='__main__':
    example_segment = '../data/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/4/'
    # example_segment = '../Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/'
    read_video(example_segment)
    # plot_ANGLE(example_segment)