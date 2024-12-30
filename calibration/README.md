# Midline pixel length to real world length 

## Calibration config
For calibration, we have three sets of configuration parameters.  First are the input and output paths. Next are the parameters for the checkboard pattern used for calibration.  Finally, we have the method used to determine overall track length.
![Alt text](images/calibration_config.jpg "calibration file")

## Input
We use the csv file output from the main program as input 

## Output
We have two outputs.  One is the input csv with two additional columns, the frame real world length and the track real world length.  The second output is a summary csv file with only the video/track name and corresponding track length
![Alt text](images/detailed_csv.jpg "detailed csv file with real world length added")
![Alt text](images/summary_csv.jpg "summary csv file with only the track length for each video")