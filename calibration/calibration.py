# Generating the visualization figures of the mask prediction
# and calculate the midline estimation

# -*- coding: utf-8 -*-
from calibration_config import (
    CSV_PATH,out_path,summary_path, pixels, x_size, y_size, r_size
)

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def get_homography():
    pt_src = np.array(pixels, dtype=np.float32)
    [x,y] = pixels[0]
    pt_tgt = np.array([[x,y],[x+x_size*10,y],[x,y+y_size*10],[x+x_size*10,y+y_size*10]], dtype=np.float32)
    H, status = cv2.findHomography(pt_src, pt_tgt)
    return H

def load_csv():
    data = pd.read_csv(CSV_PATH)
    return data

def pixel_to_real_world_points(midline_point, H):
    midline_point = np.array(midline_point) # n by 2
    n = midline_point.shape[0]
    z_coordinate = np.ones((n,1))
    
    midline_points = np.hstack((midline_point, z_coordinate))
    converted_midline_points =H.astype(np.float16).dot(midline_points.astype(np.float16).T).T
    for i in range(converted_midline_points.shape[0]):
        
        converted_midline_points[i,:] = converted_midline_points[i,:]/converted_midline_points[i,2]

    return converted_midline_points

def calculate_length(midline_points):
    dist = 0
    for i in range(midline_points.shape[0]-1):
        dist+=np.linalg.norm(midline_points[i,:]-midline_points[i+1,:])
    return dist*5/10

def convert_coordinates(H):
    converted_length = []
    pixel_length_list = []
    for i in range(data.shape[0]):
        midline_points = data['Midline Points'].iloc[i]
        pixel_length = data['Pixel Length'].iloc[i]
        img_name = data['Frame Name'].iloc[i]

        midline2 = []
        midline_points = midline_points.replace('(',"").replace(')',"").split(',')
        vidname = img_name[:-8]
        framenum = img_name[-7:]
        for j in range(0,len(midline_points),2):
            midline2.append([int(midline_points[j]),int(midline_points[j+1])])

        converted_midline =  pixel_to_real_world_points(midline2,H)
        dist = calculate_length(converted_midline)
        pixel_length_list.append(pixel_length)
        converted_length.append(dist)
    return converted_length
def track_len(data, converted_length):
    real_length = {}
    track_len = []
    track_len_dict = {}
    for i in range(data.shape[0]):
        name = data["Frame Name"].iloc[i]
        split = name.split("_")
        name = name[:-(len(split[-1])+1)]
        if(name not in real_length):
            real_length[name] = []
        real_length[name].append(converted_length[i])
    for vidname in real_length.keys():
        lengths_arr = np.array(real_length[vidname])
        lengths_arr = lengths_arr[lengths_arr<1000]
        smooth = savgol_filter(lengths_arr, 5, 3)
        track_len_dict[vidname] = max(smooth)
    for i in range(data.shape[0]):
        name = data["Frame Name"].iloc[i]
        split = name.split("_")
        name = name[:-(len(split[-1])+1)]
        track_len.append(track_len_dict[name])
    return track_len, track_len_dict

track_len_max = {}
if __name__ == "__main__":
    homography = get_homography()
    data = load_csv()
    converted_length = convert_coordinates(homography)
    track_len, track_len_dict = track_len(data, converted_length)
    
    summary = pd.DataFrame([])
    summary.insert(0,"video_name",track_len_dict.keys(), True)
    summary.insert(1,"track len (cm)",list(track_len_dict.values()), True)
    summary.to_csv(summary_path, index = False)
    
    copy = data.copy()
    copy.insert(5,"converted len (cm)",converted_length,True)
    copy.insert(6,"converted track len (cm)",track_len,True)
    copy.to_csv(out_path, index = False)
