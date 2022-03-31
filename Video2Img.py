import cv2

vidcap = cv2.VideoCapture(r"C:\Users\jinch\OneDrive\Desktop\Camera calibration\Camera calibration\Extrinsics\Vids\Cube_CornerNW.mkv")
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(r"C:\Users\jinch\OneDrive\Desktop\Camera calibration\Camera calibration\Extrinsics\Images\frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1