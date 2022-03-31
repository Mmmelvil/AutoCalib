# pts_l - set of n 2d points in left image. nx2 numpy float array
# pts_r - set of n 2d points in right image. nx2 numpy float array
#
# K_l - Left Camera matrix. 3x3 numpy float array
# K_r - Right Camera matrix. 3x3 numpy float array

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np
import json


from Camera import getprojMat, readCameraMatrix, readExtrinsics


#NW
# pts_l_x =np.array( [1158, 1356, 1371, 1162, 1360, 1374, 1745, 1715, 967, 1089], dtype=np.float64)
# pts_l_y = np.array([300, 307, 308, 414, 434, 434, 484, 682, 790, 830], dtype=np.float64)
# pts_l_x =np.array( [1060, 1197, 1324, 1114, 1261, 1396, 1174, 1337, 1479, 1068, 1203, 1325, 1119, 1263, 1394, 1180, 1341, 1478], dtype=np.float64)
# pts_l_y = np.array([399, 383, 370, 425, 409, 394, 458, 438, 425, 544, 519, 506, 579, 560, 541, 632, 610, 583], dtype=np.float64)
#
#
# pts_r_x = np.array( [1216, 1399, 1604, 1065, 1246, 1476, 876, 1051, 1293, 1230, 1393, 1584, 1092, 1257, 1464, 925, 1088, 1303], dtype=np.float64)
# pts_r_y = np.array( [202, 234, 285, 255, 302, 362, 336, 392, 478, 378, 434, 494, 453, 522, 603, 559, 643, 747], dtype=np.float64)
#
#
# #SW
# pts_l_x =np.array([921, 1259, 1238, 920, 502, 587,  74, 121], dtype=np.float64)
# pts_l_y = np.array([259, 308, 557, 473, 598, 573, 636, 846], dtype=np.float64)
#
#
# #WallS
#
# pts_r_x = np.array( [1294, 1496, 1470, 1304, 828, 960, 469, 514], dtype=np.float64)
# pts_r_y = np.array( [256, 431, 835, 542, 585, 588, 533, 728], dtype=np.float64)
#
#
# pts_l = np.dstack([pts_l_x, pts_l_y]).reshape((8,2))
# pts_r = np.dstack([pts_r_x, pts_r_y]).reshape((8, 2))

#
pts_l =np.array( json.load(open('./Extrinsics/WallS.extrinsics.points.json'))['imgPoints'])
pts_r = np.array(json.load(open('./Extrinsics/CornerSE.extrinsics.points.json'))['imgPoints'])


Intrinsics_l = readCameraMatrix('./Intrinsics/WallS.intrinsics.json')
Intrinsics_r = readCameraMatrix('./Intrinsics/CornerSE.intrinsics.json')


K_l = Intrinsics_l['C_mat']
K_r = Intrinsics_r['C_mat']
dParams_l = Intrinsics_l['d_coeffs']
dParams_r = Intrinsics_r['d_coeffs']

pts_l_norm = cv2.undistortPoints(pts_l, cameraMatrix=K_l, distCoeffs=dParams_l)[:,0,:]
pts_r_norm = cv2.undistortPoints(pts_r, cameraMatrix=K_r, distCoeffs=dParams_r)[:,0,:]


# print(np.vstack((pts_l_norm, np.ones((27,1), dtype=float))))
pts_l_norm_hmg = np.concatenate([pts_l_norm, np.ones((pts_l_norm.shape[0],1), dtype=float)], axis=1).T
pts_r_norm_hmg = np.concatenate([pts_r_norm, np.ones((pts_r_norm.shape[0],1), dtype=float)], axis=1).T


# print(pts_r_norm_hmg.T@E@pts_l_norm_hmg)


pts_l_norm = cv2.undistortPoints(pts_l, cameraMatrix=K_l, distCoeffs=dParams_l)
pts_r_norm = cv2.undistortPoints(pts_r, cameraMatrix=K_r, distCoeffs=dParams_r)

E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999,
                               threshold=1e-4)
R, R2, t = cv2.decomposeEssentialMat(E)


print('Relative rotation matrix calculated:\n', R)



## calculate relative rotation matrix
view1_Rmat = readExtrinsics('./Extrinsics/WallS.extrinsics.position.json')['R_mat']
view2_Rmat = readExtrinsics('./Extrinsics/CornerSE.extrinsics.position.json')['R_mat']

view1_t = readExtrinsics('./Extrinsics/WallS.extrinsics.position.json')['T']
view2_t = readExtrinsics('./Extrinsics/CornerSE.extrinsics.position.json')['T']

print('view1 R mat:', view1_Rmat)
view1_extrMat = np.concatenate([np.concatenate([view1_Rmat, view1_t[:,None]], axis=1), np.array([[0,0,0,1]], dtype=float)], axis=0)
view2_extrMat = np.concatenate([np.concatenate([view2_Rmat, view2_t[:,None]], axis=1), np.array([[0,0,0,1]], dtype=float)], axis=0)


## triangulate points and plot in Euclidean space
projMat_l = getprojMat(K_l)
projMat_r = getprojMat(K_r, R, t)

print('Projection Matrix 1:\n', projMat_l, '\n Projection Matrix 2:\n', projMat_r)


triangulated_points = cv2.triangulatePoints(projMat_l, projMat_r, np.transpose(pts_l), np.transpose(pts_r))

print(triangulated_points.shape)
cube_world_points = np.linalg.inv(view1_extrMat)@triangulated_points
Euclidean_points = cv2.convertPointsFromHomogeneous(np.transpose(cube_world_points))
# Euclidean_points = cv2.convertPointsFromHomogeneous(np.transpose(triangulated_points))
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')

for point in Euclidean_points:
    print(point.reshape(-1))
    ax.scatter(point.reshape(-1)[0], point.reshape(-1)[1], point.reshape(-1)[2]) # plot the point (2,3,4) on the figure
plt.show()


extrMat12 = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0,0,0,1]], dtype=float)], axis=0)
print('extrMat12:\n', extrMat12)
print(view1_extrMat)
print()
print(extrMat12@view1_extrMat)
print()
print(view2_extrMat)

print(Rotation.from_matrix(view2_Rmat).as_euler('zyx',degrees=True), Rotation.from_matrix(view1_Rmat*R).as_euler('zyx',degrees=True))

rot_mat_rel = np.transpose(np.matmul(np.transpose(view1_Rmat), view2_Rmat)) # calculate relative rotation

zyx_rel = Rotation.from_matrix(rot_mat_rel).as_euler('zyx',degrees=True)
print('Relative euler angles zyx from file:\n',zyx_rel)