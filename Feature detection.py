import math
import cv2 as cv
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import numpy as np
import matplotlib.pyplot as plt
# img1 = cv.imread('structure-from-motion-master/SfM_quality_evaluation-master/Benchmarking_Camera_Calibration_2008/castle-P19/images/0000.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('structure-from-motion-master/SfM_quality_evaluation-master/Benchmarking_Camera_Calibration_2008/castle-P19/images/0001.jpg',cv.IMREAD_GRAYSCALE) # trainImage

img1 = cv.imread('structure-from-motion-master/images/CornerNW.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('structure-from-motion-master/images/WallW.jpg',cv.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv.DrawMatchesFlags_DEFAULT)
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
# #plt.imshow(img3,),plt.show()

def siftMatching(img1, img2):
    # Input : image1 and image2 in opencv format
    # Output : corresponding keypoints for source and target images
    # Output Format : Numpy matrix of shape: [No. of Correspondences X 2]

    #surf = cv.xfeatures2d.SURF_create(100)
    surf = cv.xfeatures2d.SIFT_create()

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    # Ransac
    model, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform, min_samples=4,
            residual_threshold=8, max_trials=10000
        )

    n_inliers = np.sum(inliers)

    inlier_keypoints_left = [cv.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv.DMatch(idx, idx, 1) for idx in range(n_inliers)]

    print(inlier_keypoints_left, inlier_keypoints_right)
    image3 = cv.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)

    dim_im3 = image3.shape[0:2]
    # print(dim_im3)
    # print(math.floor(dim_im3[0]/2), math.floor(dim_im3[1]/2))
    # image3 = cv.resize(image3,(math.floor(dim_im3[0]/2), math.floor(dim_im3[1]/2)), interpolation=cv.INTER_AREA)

    cv.namedWindow("Matches", 0)
    cv.resizeWindow("Matches",math.floor(dim_im3[0]/2), math.floor(dim_im3[1]/2) )

    cv.imshow('Matches', image3)
    cv.waitKey(0)

    src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

    return src_pts, dst_pts



siftMatching(img1, img2)


cv.waitKey()
cv.destroyAllWindows()