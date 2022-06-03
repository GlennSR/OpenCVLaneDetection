# OpenCVLaneDetection
Code developped for a college project where the mobile robot had to harvest plants in a farm, so it had to move following the plantation line

## Image lane detection
Original code that I modified to create the Camera lane detection, it work with images

## Camera lane detection
Final code of the project that identifies the two lanes (left and right) and plots their center (blue point), then it calculates the distance (cian line) between the the lanes center and the car center (cian point)

### Lane detection demo

<p align="center">
  <img width="600" height="400" src="lane_detection.gif">
</p>

## How to use
At line 131 "cap = cv2.VideoCapture('video_name.sufix')" write the path tp your video file or put 0 if you want to use your camera as input instead.

If you want to play with the thresholds, in the function process() at line 119, change the values and test the output, read about cv2.HoughLinesP() for more details.
