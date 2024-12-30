# Generating the visualization figures of the mask prediction
# and calculate the midline estimation

# -*- coding: utf-8 -*-
from config import (
    MEAN, DIREC, MIDLINE_PTS, KERNEL_SIZE, VIDEO_DIR,
    VISUALIZE_FLAG, VISUALIZE_OUTPUT_PATH, MODEL_PATH,
    CSV_PATH, VIDEO_FLAG, VIDEO_OUTPUT_PATH, JSON_FLAG,
    JSON_OUTPUT_PATH
)

import os
import numpy as np
import cv2
import csv
import json
from math import atan2, cos, sin, sqrt, pi, degrees, radians
from ultralytics import YOLO
import imageio.v2 as imageio
import torch

FIND_KERNEL_SIZE = True
PREVIOUS_ANGLE_OVERALL = None
PREVIOUS_ANGLE_HEAD = None
PREVIOUS_ANGLE_TAIL = None
EIGENVECTORS = None
EIGENVALUES = None
EIGENVECTORS_HEAD = None
EIGENVALUES_HEAD = None
EIGENVECTORS_TAIL = None
EIGENVALUES_TAIL = None

def drawAxis(img, p_, q_, colour, scale):
    """
    Draws an axis with arrow hooks on an image that represents the vector direction
    Parameters:
        img: The image to draw the axis on
        p_: Starting point of the axis (x, y)
        q_: End point of the axis (x, y)
        colour: Color of the axis in BGR format
        scale: Scale factor to lengthen the axis
    """
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) ** 2 + (p[0] - q[0]) ** 2)

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 2, cv2.LINE_AA)

def getLength(pts):
    """
    Calculates the total length of a polyline by summing the Euclidean distances between consecutive points
    Parameters:
        pts (list of tuples): List of (x, y) coordinates representing points of the polyline
    Returns:
        float: The total length of the polyline
    """
    length = 0.0
    for i in range(len(pts) - 1):
        length += ((pts[i+1][0] - pts[i][0])**2 + (pts[i+1][1] - pts[i][1])**2)**0.5
    return length

def get_bounding_box(result):
    """
    Retrieves the bounding box coordinates from YOLO detection results
    Parameters:
        result: Detection result object from YOLO model inference
    Returns:
        x1, y1, x2, y2: Bounding box coordinates (top-left and bottom-right)
    """
    if result and result[0].boxes:
        box = result[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        return x1, y1, x2, y2
    else:
        return None, None, None, None
    
def save_frame_data_to_csv(csv_writer, frame_name, species, midline_pts, bbox, pixel_length):
    """
    Saves the data of each frame to the CSV file
    Parameters:
        csv_writer: CSV writer object
        frame_name: Name of the frame being processed
        species: Name of the detected species
        midline_pts: Pixel coordinates of each midline point
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        pixel_length: Length of the fish in pixels
    """
    midline_pts_str = ', '.join([f"({int(pt[0])}, {int(pt[1])})" for pt in midline_pts])
    bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})" if bbox[0] is not None else "No Bounding Box"
    csv_writer.writerow([frame_name, species, midline_pts_str, bbox_str, pixel_length])

def save_fish_detection_to_json(frame_name, mask, image_width, image_height):
    """
    Creates and saves a JSON file with fish detection data using the largest contour.
    Parameters:
        frame_name: Name of the frame being processed
        mask: Binary mask representing the detected fish
        image_width: Width of the original image
        image_height: Height of the original image
    """

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.squeeze().tolist()

    json_content = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [
            {
                "label": "SleeperShark",
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
        ],
        "imagePath": f"{frame_name}.jpg",
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    os.makedirs(JSON_OUTPUT_PATH, exist_ok=True)
    json_path = os.path.join(JSON_OUTPUT_PATH, f"{frame_name}.json")
    with open(json_path, 'w') as json_file:
        json.dump(json_content, json_file, indent=4)

def split_mass(pts):
    """
    Splits the contour points into two regions (head and tail) using PCA analysis to find the orthogonal axis
    Parameters:
        pts: Points representing the contour
    Returns:
        head_pts: Points belonging to the head region
        tail_pts: Points belonging to the tail region
    """
    sz = len(pts)
    if sz < 2:
        print("Not enough points to perform PCA")
        return None, None

    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    MEAN = np.empty((0))
    MEAN, EIGENVECTORS, EIGENVALUES = cv2.PCACompute2(data_pts, MEAN)
    if EIGENVECTORS.shape[0] < 2:
        print("PCA returned fewer than 2 components")
        return None, None

    cntr = (int(MEAN[0, 0]), int(MEAN[0, 1]))
    orthogonal_vector = EIGENVECTORS[0]
    head_pts = []
    tail_pts = []

    for pt in data_pts:
        projection = (pt[0] - cntr[0]) * orthogonal_vector[0] + (pt[1] - cntr[1]) * orthogonal_vector[1]
        if projection < 0:
            head_pts.append(pt)
        else:
            tail_pts.append(pt)

    head_pts = np.array(head_pts, dtype=np.float64)
    tail_pts = np.array(tail_pts, dtype=np.float64)
    return head_pts, tail_pts

def correct_angle(current_angle, previous_angle):
    """
    Corrects the angle to ensure smooth transitions and avoid large sudden changes in the midline
    Parameters:
        current_angle: The current angle in radians
        previous_angle: The previous angle in degrees for comparison
    Returns:
        corrected_angle: The corrected angle in radians
        angle_degrees: The corrected angle in degrees
    """
    angle_degrees = degrees(current_angle)

    if previous_angle is not None:
        angle_diff = abs(angle_degrees - previous_angle)

        if angle_diff > 180:
            angle_diff = abs(previous_degrees - angle_degrees)
        
        if angle_diff > 90:
            angle_degrees -= 180
            if angle_degrees < -360:
                angle_degrees += 360

        if angle_diff > 30:
            if angle_degrees > previous_angle:
                angle_degrees = previous_angle + 20
            else:
                angle_degrees = previous_angle - 20
        
        if angle_degrees > 180:
            angle_degrees -= 360
        elif angle_degrees < -180:
            angle_degrees += 360

    corrected_angle = radians(angle_degrees)
    return corrected_angle, angle_degrees

def getHeadpoint(pts, cent, eigenvectors, direc):
    """
    Identifies the furthest head point along the principal direction from the center point
    Parameters:
        pts: Array of points
        cent: Center of mass point from PCA
        eigenvectors: Eigenvectors from PCA
        direc: Direction of the principal component
    Returns:
        maxidx: Index of the head point
    """
    maxmag = 0
    maxidx = -1
    for i in range(pts.shape[0]):
        mag = (pts[i, 0] - cent[0]) * eigenvectors[0, 0] * direc + (pts[i][1] - cent[1]) * eigenvectors[0, 1] * direc
        if(mag > maxmag):
            maxmag = mag
            maxidx = i
    return maxidx

def getTailpoint(pts, cent, eigenvectors, direc):
    """
    Identifies the furthest tail point along the principal direction from the center point
    Parameters:
        pts: Array of points
        cent: Center of mass point from PCA
        eigenvectors: Eigenvectors from PCA
        direc: Direction of the principal component
    Returns:
        minidx, secminidx: Indices of the two furthest tail points
    """
    minmag = 1e+5
    minidx = -1
    secminmag = 1e+5
    secminidx = -1
    for i in range(pts.shape[0]):
        mag = (pts[i, 0] - cent[0]) * eigenvectors[0, 0] * direc + (pts[i][1] - cent[1]) * eigenvectors[0, 1] * direc
        if(mag < minmag):
            secminmag = minmag
            secminidx = minidx
            minmag = mag
            minidx = i
    return minidx, secminidx

def getPts(pts, img):
    """
    Computes the PCA for a given set of contour points, splits the points into head and tail regions,
    and applies angle correction based on previous PCA angles
    Parameters:
        pts: List of contour points
        img: The image for visualization
    Returns:
        hpt: The head point after PCA
        tpt: The tail point after PCA
    """
    global PREVIOUS_ANGLE_OVERALL, PREVIOUS_ANGLE_HEAD, PREVIOUS_ANGLE_TAIL, \
        EIGENVECTORS, EIGENVALUES, EIGENVECTORS_HEAD, EIGENVALUES_HEAD, \
        EIGENVECTORS_TAIL, EIGENVALUES_TAIL

    sz = len(pts)
    if sz < 2:
        print("Not enough points to perform PCA")
        return None, None, None

    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    MEAN = np.empty((0))
    if PREVIOUS_ANGLE_OVERALL == None:
        MEAN, EIGENVECTORS, EIGENVALUES = cv2.PCACompute2(data_pts, MEAN)
    else:
        MEAN, _, _ = cv2.PCACompute2(data_pts, MEAN)
    if EIGENVECTORS.shape[0] < 2:
        print("PCA returned fewer than 2 components")
        return None, None, None

    cntr = (int(MEAN[0, 0]), int(MEAN[0, 1]))
    angle = atan2(EIGENVECTORS[0, 1], EIGENVECTORS[0, 0])

    # Apply angle correction for the overall PCA
    corrected_angle_degrees, angle  = correct_angle(angle, PREVIOUS_ANGLE_OVERALL)
    PREVIOUS_ANGLE_OVERALL = corrected_angle_degrees

    orthogonal_vector = EIGENVECTORS[1]
    head_pts = []
    tail_pts = []

    for pt in data_pts:
        projection = (pt[0] - cntr[0]) * orthogonal_vector[0] + (pt[1] - cntr[1]) * orthogonal_vector[1]
        if projection < 0:
            head_pts.append(pt)
        else:
            tail_pts.append(pt)

    head_pts = np.array(head_pts, dtype=np.float64)
    tail_pts = np.array(tail_pts, dtype=np.float64)

    if len(head_pts) == 0:
        print("No head points")
        return None, None, None
    if len(tail_pts) == 0:
        print("No tail points")
        return None, None, None

    mean_head = np.empty((0))
    if PREVIOUS_ANGLE_HEAD == None:
        mean_head, EIGENVECTORS_HEAD, EIGENVALUES_HEAD = cv2.PCACompute2(head_pts, mean_head)
    else:
        mean_head, _, _ = cv2.PCACompute2(head_pts, mean_head)
    if EIGENVECTORS_HEAD.shape[0] < 2:
        print("PCA returned fewer than 2 components")
        return None, None, None

    mean_tail = np.empty((0))
    if PREVIOUS_ANGLE_HEAD == None:
        mean_tail, EIGENVECTORS_TAIL, EIGENVALUES_TAIL = cv2.PCACompute2(tail_pts, mean_tail)
    else:
        mean_tail, _, _ = cv2.PCACompute2(tail_pts, mean_tail)
    if EIGENVECTORS_TAIL.shape[0] < 2:
        print("PCA returned fewer than 2 components")
        return None, None, None

    cntr_head = (int(mean_head[0, 0]), int(mean_head[0, 1]))
    cntr_tail = (int(mean_tail[0, 0]), int(mean_tail[0, 1]))

    cv2.circle(img, cntr_head, 3, (255, 0, 255), 2)
    cv2.circle(img, cntr_tail, 3, (0, 0, 255), 2)
    
    p1_head = (cntr_head[0] + 0.02 * EIGENVECTORS_HEAD[0, 0] * EIGENVALUES_HEAD[0, 0],
               cntr_head[1] + 0.02 * EIGENVECTORS_HEAD[0, 1] * EIGENVALUES_HEAD[0, 0])
    p2_head = (cntr_head[0] - 0.02 * EIGENVECTORS_HEAD[1, 0] * EIGENVALUES_HEAD[1, 0],
               cntr_head[1] - 0.02 * EIGENVECTORS_HEAD[1, 1] * EIGENVALUES_HEAD[1, 0])
    drawAxis(img, cntr_head, p1_head, (0, 255, 0), 1)
    drawAxis(img, cntr_head, p2_head, (255, 255, 0), 5)

    p1_tail = (cntr_tail[0] + 0.02 * EIGENVECTORS_TAIL[0, 0] * EIGENVALUES_TAIL[0, 0],
               cntr_tail[1] + 0.02 * EIGENVECTORS_TAIL[0, 1] * EIGENVALUES_TAIL[0, 0])
    p2_tail = (cntr_tail[0] - 0.02 * EIGENVECTORS_TAIL[1, 0] * EIGENVALUES_TAIL[1, 0],
               cntr_tail[1] - 0.02 * EIGENVECTORS_TAIL[1, 1] * EIGENVALUES_TAIL[1, 0])
    drawAxis(img, cntr_tail, p1_tail, (0, 0, 255), 1)
    drawAxis(img, cntr_tail, p2_tail, (255, 0, 0), 5)

    hpt = getHeadpoint(head_pts, cntr_head, EIGENVECTORS_HEAD, np.dot(EIGENVECTORS_HEAD[0], EIGENVECTORS[0]))
    tpt1, tpt2 = getTailpoint(tail_pts, cntr_tail, EIGENVECTORS_TAIL, np.dot(EIGENVECTORS_TAIL[0], EIGENVECTORS[0]))

    cv2.circle(img, (int(head_pts[hpt, 0]), int(head_pts[hpt, 1])), 3, (255, 0, 255), 2)
    cv2.circle(img, (int(tail_pts[tpt1, 0]), int(tail_pts[tpt1, 1])), 3, (0, 0, 255), 2)
    cv2.circle(img, (int(tail_pts[tpt2, 0]), int(tail_pts[tpt2, 1])), 3, (0, 0, 255), 2)

    return [int(head_pts[hpt, 0]), int(head_pts[hpt, 1])], [int(tail_pts[tpt1, 0]), int(tail_pts[tpt1, 1])], cntr

def getMidline(binary_mask, MIDLINE_PTS):
    """
    Computes the midline of the largest contour in the binary mask by eroding the mask, 
    then splitting it into head and tail regions
    Parameters:
        binary_mask (np.ndarray): The binary image mask of the object
        MIDLINE_PTS (int): Number of points to approximate along the midline
    Returns:
        hpts: List of head midline points
        tpts: List of tail midline points
        head_pts: Points belonging to the head region
        tail_pts: Points belonging to the tail region
    """
    global FIND_KERNEL_SIZE, KERNAL_SIZE, PREVIOUS_ANGLE_OVERALL, PREVIOUS_ANGLE_HEAD, PREVIOUS_ANGLE_TAIL

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    maxidx = 0
    maxarea = 0

    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if(area > maxarea):
            maxarea = area
            maxidx = i
            x, y, w, h = cv2.boundingRect(contour)

    if(maxarea > 0):

        head_pts, tail_pts = split_mass(contours[maxidx])

        if(FIND_KERNEL_SIZE):
            KERNAL_SIZE = max(1, int(min(w, h) / MIDLINE_PTS))
            FIND_KERNEL_SIZE = False

        kernel = np.ones((KERNAL_SIZE, KERNAL_SIZE), np.uint8)
        erosion = cv2.erode(binary_mask, kernel, iterations=1)

        hpt, tpt, cntr = getPts(contours[maxidx], binary_mask)
        if hpt is None or tpt is None:
            print("Invalid head or tail point detected.")
            return None, None, None, None
            
        # Check for U-shaped fish
        if cv2.pointPolygonTest(contours[maxidx], cntr, False) < 0:
            return None, None, None, None

        hpts, tpts, _, _ = getMidline(erosion, MIDLINE_PTS)

        if hpts is None or tpts is None:
            return None, None, None, None

        hpts.insert(0, hpt)
        tpts.append(tpt)
    else:
        hpts = []
        tpts = []
        head_pts = []
        tail_pts = []
        PREVIOUS_ANGLE_OVERALL = None
        PREVIOUS_ANGLE_HEAD = None
        PREVIOUS_ANGLE_TAIL = None
    return hpts, tpts, head_pts, tail_pts

def process_image(model, image_path):
    """
    Processes a single image using a segmentation model
    Parameters:
        model: The segmentation model used to make predictions
        image_path: Path to the image file
    Returns:
        img: The original image read from the path
        masks: The predicted segmentation mask
        result: The prediction results from the model
    """
    img = cv2.imread(image_path)
    result = model.predict(image_path, verbose=False)
    
    # Check if result and masks are valid
    if result and result[0].masks:
        masks = result[0].masks[0].data.numpy().transpose(1, 2, 0)
        height, width, _ = img.shape
        masks = cv2.resize(masks, (width, height), interpolation=cv2.INTER_CUBIC)
        masks = np.uint8(masks) * 255
        kernel = np.ones((5, 5), np.uint8)
        masks = cv2.morphologyEx(masks, cv2.MORPH_CLOSE, kernel)
    else:
        masks = None
    
    return img, masks, result

def visualize_and_save(img, mask, frame_name, result, csv_writer, frame_output_dir):
    """
    Visualizes the segmentation and midline estimation, saves the output image, and logs data to the CSV
    Parameters:
        img: The original image
        mask: The predicted segmentation mask
        frame_name: Name of the frame being processed
        result: Prediction results from the segmentation model
        csv_writer: CSV writer object for logging data
        frame_output_dir: Directory to save the visualized output frames
    """
    global MIDLINE_PTS

    # Initialize overlay as a copy of the original image
    overlay = img.copy()

    if mask is not None:
        hpts, tpts, head_pts, tail_pts = getMidline(mask, MIDLINE_PTS)

        if hpts is not None and tpts is not None:
            # Convert hpts and tpts to numpy arrays
            hpts = np.array(hpts)
            tpts = np.array(tpts)
            all_midline_pts = list(hpts) + list(tpts)

            if JSON_FLAG:
                save_fish_detection_to_json(
                    frame_name,
                    mask,
                    img.shape[1],
                    img.shape[0]
                )

            if VISUALIZE_FLAG:
                # Overlay segmentation mask on image
                colored_mask = np.zeros_like(img)
                colored_mask[mask > 0] = [255, 0, 0]
                overlay = cv2.addWeighted(img, 1, colored_mask, 0.3, 0)

                # Draw midline
                for pt in hpts:
                    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1)
                for i in range(len(hpts) - 1):
                    cv2.line(overlay, (int(hpts[i][0]), int(hpts[i][1])), (int(hpts[i + 1][0]), int(hpts[i + 1][1])), (0, 255, 255), 2)

                for pt in tpts:
                    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
                for i in range(len(tpts) - 1):
                    cv2.line(overlay, (int(tpts[i][0]), int(tpts[i][1])), (int(tpts[i + 1][0]), int(tpts[i + 1][1])), (255, 0, 0), 2)

                # Draw a line connecting the first head point to the first tail point
                if len(hpts) > 0 and len(tpts) > 0:
                    cv2.line(overlay, (int(hpts[-1][0]), int(hpts[-1][1])), (int(tpts[0][0]), int(tpts[0][1])), (255, 255, 255), 2)

            # Get bounding box
            x1, y1, x2, y2 = get_bounding_box(result)
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate length
            length = getLength(all_midline_pts)
            cv2.putText(overlay, f"Length: {length:.2f} px", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Save frame data to CSV
            species_name = result[0].names[0]  # Assuming first detection is relevant species
            save_frame_data_to_csv(csv_writer, frame_name, species_name, all_midline_pts, (x1, y1, x2, y2), length)
    else:
        cv2.putText(overlay, "No mask found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # only save img if object detected
    if mask is not None:
        if VISUALIZE_FLAG:
            visualize_path = os.path.join(frame_output_dir, f"{frame_name}.jpg")
            if not os.path.exists(os.path.dirname(visualize_path)):
                os.makedirs(os.path.dirname(visualize_path))
            cv2.imwrite(visualize_path, overlay)

def create_video_from_frames(frame_dir, output_file):
    """
    Creates a video from a sequence of image frames
    Parameters:
        frame_dir: Directory containing the image frames
        output_file: Path to save the generated video file
    """
    frame_files = [f for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')]
    frame_files.sort()  # Ensure frames are in order

    # Create video writer
    writer = imageio.get_writer(output_file, format='mp4')

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        image = imageio.imread(frame_path)
        writer.append_data(image)

    writer.close()

def process_directory(model, root_directory_path, csv_filepath):
    """
    Processes all images in subfolders (representing video clips) within a root directory
    Parameters:
        model: The segmentation model used for image processing
        root_directory_path: Path to the root directory containing subfolders with images
        csv_filepath: Path to save the CSV file containing processed data
    """
    global VISUALIZE_OUTPUT_PATH, VIDEO_FLAG, VIDEO_OUTPUT_PATH
    csv_dir = os.path.dirname(csv_filepath)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    with open(csv_filepath, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Frame Name', 'Species', 'Midline Points', 'Bounding Box', 'Pixel Length'])
        vidcnt = 0
        total = len(os.listdir(root_directory_path))
        for foldername in os.listdir(root_directory_path):
            vidcnt +=1
            folder_path = os.path.join(root_directory_path, foldername)
            if os.path.isdir(folder_path):
                print(f"Processing folder (video clip): {folder_path}, {vidcnt}/{total}")
                
                # Create a directory for storing visualized frames for the current video
                video_frame_dir = os.path.join(VISUALIZE_OUTPUT_PATH, foldername)
                if VISUALIZE_FLAG:
                    if not os.path.exists(video_frame_dir):
                        os.makedirs(video_frame_dir)

                cnt = 1
                totalcnt = len(os.listdir(folder_path))
                for filename in os.listdir(folder_path):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        image_path = os.path.join(folder_path, filename)
                        frame_number = os.path.splitext(filename)[0]
                        new_frame_name = f"{foldername}_{frame_number}"
                        if(cnt%100 == 0):
                            print(f"Processing %s: %d/%d frames"%(folder_path,cnt, totalcnt))
                        cnt+=1
                        img, mask, result = process_image(model, image_path)
                        visualize_and_save(img, mask, new_frame_name, result, csv_writer, video_frame_dir)
                
                # Create a video for the current folder of frames
                if VIDEO_FLAG:
                    if not os.path.exists(VIDEO_OUTPUT_PATH):
                        os.makedirs(VIDEO_OUTPUT_PATH)
                    output_video_path = os.path.join(VIDEO_OUTPUT_PATH, f"{foldername}.mp4")
                    create_video_from_frames(video_frame_dir, output_video_path)

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    process_directory(model, VIDEO_DIR, CSV_PATH)