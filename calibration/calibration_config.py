
#######################################################################
# input CSV directory
CSV_PATH = '../all_data_ak.csv'

# detailed output path
out_path = "output.csv"

# summary output path
summary_path ="track_summary.csv"

######################################################################

# Calibration 
# square pixel locations [[upper left point],[upper right point],[lower left point],[lower right point]]
pixels = [[644,358], [695, 347], [661, 430], [712, 418]]

# checkerboard size
x_size = 4
y_size = 5

# checkerboard single square size in cm
r_size = 5

#####################################################################

# track length method
# 'max' - uses the maximum frame length as the track length
# 'smooth max' - first performs filtering to smooth curve before using maximum smoothed frame length as track length
# 'gaussian fit' - finds the mean and std length of the track and uses a set number of std above the mean as the track length
method = 'max'
