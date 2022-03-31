from Camera import getprojMat, readCameraMatrix, readExtrinsics, Camera
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## get the image coordinates with 3d world coordinates
def estimateCamExtrinsics(objPoints, imgPoints, Cam):


_,rvec,tvec = cv2.solvePnP(objPoints, imgPoints, Cam.camMat, Cam.distortion[:Cam.nDistortionParams]