# import the necessary packages
from pyimagesearch.tempimage import TempImage
from picamera.array import PiYUVArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import dropbox
import imutils
import json
import time
import numpy as np
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

# check to see if the Dropbox should be used
if conf["use_dropbox"]:
    # connect to dropbox and start the session authorization process
    client = dropbox.Dropbox(conf["dropbox_access_token"])
    print("[SUCCESS] dropbox account linked")

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiYUVArray(camera, size=tuple(conf["resolution"]))

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.8,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Take first frame and find corners in it
#ret, old_frame = cap.read()
#old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
camera.capture(rawCapture, format="yuv", use_video_port=True)
old_gray = rawCapture.array[:, :, 0]
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)
rawCapture.truncate(0)

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="yuv", use_video_port=True):
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    frame_gray = f.array[:, :, 0]  # accedemos a todas las filas, columnas del canal Y
    frame = frame_gray.copy()

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0xff), 2)
        frame = cv2.circle(frame, (a, b), 5, (0xff), -1)
    img = cv2.add(frame, mask)

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display the security feed
        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

cv.destroyAllWindows()
