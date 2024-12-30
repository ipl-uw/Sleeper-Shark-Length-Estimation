# Fish Mask Prediction and Midline Estimation

This repository contains code for generating visualization figures of mask predictions and calculating midline estimations for fish images. The pipeline uses a YOLO model for segmentation and processes the data to compute midlines, bounding boxes, and fish lengths.

## Requirements

Ensure you have **Python** (version 3.8 or higher) installed on your system. You will also need `venv` or `virtualenv` for creating a virtual environment. All other dependencies will be installed during the setup process.

## Setup

Follow these steps to set up the environment for running the project. 

1. First, clone the repository to your local machine using the following commands:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Next, create a virtual environment using the following commands:
    ```bash
    python -m venv shark
    ```

    If on macOS/linux:
    ```bash
    source shark/bin/activate
    ```

    If on Windows:
    ```bash
    shark\Scripts\activate
    ```

3. With the virtual environment active (you’ll see the environment name, such as shark, displayed at the beginning of your terminal prompt), install the required Python libraries using pip using the following commands:
    ```bash
    pip install numpy opencv-python torch ultralytics imageio
    ```

## Running the Script

Make sure your input data (fish images) is placed in the VIDEO_DIR directory (specified by the config file), organized into subfolders. Each subfolder represents a separate set of images or a video clip. Run the script with the virtual environment activated:
```bash
python main.py
```

## Outputs
The following outputs will be generated (based on what flags you set in the config file):

1. Visualization Images: Segmentation masks and midline estimations will be saved in the VISUALIZE_OUTPUT_PATH (if enabled).
2. Videos: Videos combining processed frames will be saved in the VIDEO_OUTPUT_PATH (if enabled).
3. JSON Files: Metadata for each processed frame will be saved in the JSON_OUTPUT_PATH (if enabled).
4. CSV File: A file containing results with midline points, bounding boxes, and pixel lengths will be saved in the CSV_PATH (a csv file will always be generated).