import json
import numpy as np
from scipy.spatial.transform import Rotation
from math import pi as PI
from rotation import angles2R

VALID_DISTORTION_PARAM_COUNTS = (0, 4, 5, 8, 12, 14)

def getprojMat(camMat, extrMat):
    # return projection matrix 3x4
    if camMat is None:
        camMat = np.eye(3, dtype=float)
    # if R  is None:
    #     R = np.identity(3, dtype=float)
    # if t is None:
    #     t = np.zeros((3, 1), dtype=float)
    print(extrMat)
    V = np.linalg.inv(extrMat)[:3,:]
    # projMat = camMat@np.linalg.inv(np.concatenate((R, t.reshape(3,1)), axis=1))
    projMat = camMat @ V
    return projMat


def readCameraMatrix(jsonfile):
    K = json.load(open(jsonfile))
    C_mat = np.array([[K['fx'], 0, K['cx']],[0, K['fy'], K['cy']], [0, 0, 1]]).astype(float)
    d_coeffs = np.array(K['dParams']).astype(float)[0:int(K['nDParams'])]
    Intrinsics = {'C_mat' : C_mat, 'd_coeffs': d_coeffs}
    return Intrinsics


def readExtrinsics(jsonfile):
    E = json.load(open(jsonfile))
    Euler = Rotation.from_euler('zyx', [E['yaw'], E['pitch'], E['roll']],
                                     degrees=True)
    R_mat = Euler.as_matrix()

    R_mat = angles2R(yaw=float(E['yaw']),pitch=float(E['pitch']), roll=float(E['roll']), order='ypr')
    T = np.array([E['posx'], E['posy'],E['posz']], dtype=float)
    
    extrMat = np.concatenate([np.concatenate([R_mat, T[:,None]], axis=1), np.array([[0,0,0,1]], dtype=float)], axis=0)
    Extrinsics = {'Euler': Euler, 'R_mat': R_mat, 'T': T, 'extrMat': extrMat}
    return Extrinsics




def extractRT(pts_l, pts_r, K_l, K_r, dParams_l, dParams_r):
    pts_l_norm = cv2.undistortPoints(pts_l, cameraMatrix=K_l, distCoeffs=dParams_l)
    pts_r_norm = cv2.undistortPoints(pts_r, cameraMatrix=K_r, distCoeffs=dParams_r)

    E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
    points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)

    return points, R, t, mask

class Camera():
    def __init__(self, camMat=np.eye(3, dtype=float), distortion=np.array([], dtype=float), transform=np.eye(4, dtype=float)):
        self.camMat = np.eye(3, dtype=float)
        self.distortion = np.zeros((14,), dtype=float)
        self.nDistortionParams = 0
        self.T = np.eye(4, dtype=float)
        self.V = np.linalg.inv(self.T)
        self.P = self.camMat @ self.T[:3, :]  # Projection matrix

        self.setIntrinsicParams(camMat, distortion)
        self.setExtrinsicParams(transform)

    
    def setIntrinsicParams(self, camMat=None, distortion=None):
        '''
        Change intrinsic calibration
        camMat: matrix of the form
                [[fx, 0,  cx],
                 [0,  fy, cy],
                 [0,  0,  1 ]]
        distortion: array of the form [k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]]] (length 0, 4, 5, 8, 12 or 14)
        '''
        if camMat is not None and camMat.shape == (3, 3):
            self.camMat = camMat
            self.P = self.camMat @ self.T[:3, :]
        if distortion is not None and len(distortion.shape) == 1 and distortion.shape[
            0] in VALID_DISTORTION_PARAM_COUNTS:
            self.nDistortionParams = distortion.shape[0]
            self.distortion[:self.nDistortionParams] = distortion
    
    
    def setExtrinsicParams(self, transform=None):
        '''
        Change extrinsic calibration
        transform: extrinsic transformation matrix of form
                   [[rx, 0,  0,  tx],
                    [0,  ry, 0,  ty],
                    [0,  0,  rz, tz],
                    [0,  0,  0,   1]] 
        '''
        if transform is not None and transform.shape == (4,4):
            self.T = transform # Transformation from world to cam coordinates
            self.V = np.linalg.inv(self.T) # Transformation from cam to world coordinates
            self.P = self.camMat@self.T[:3,:]
    
    
    def setExtrinsicParamsInv(self, view=None):
        if view is not None and view.shape == (4,4):
            self.V = view
            self.T = np.linalg.inv(self.V)
            self.P = self.camMat@self.T[:3,:]

    def get3DLine(self, imgPoint, format='TdL'):
        '''Map an image point [x,y] to a line in 3D space

           returns:
           if format is 'TdL':  translation, dL
           if format is 'abcd': [a,b,c,d] from equation ax+by+cz=d'''
        if self.T is not None and self.V is not None and imgPoint is not None:
            translation = self.V[:3, 3]
            dL = self.T[:3, :3].T @ (np.linalg.inv(self.camMat) @ np.concatenate([np.array(imgPoint), [1., ]], axis=0))

            if format == 'TdL':
                dL /= np.linalg.norm(dL)
                return translation, dL
            elif format == 'abcd':
                abc = np.array([1 / dL[0], -2 / dL[1], 1 / dL[2]])
                abc /= np.linalg.norm(abc)
                d = np.dot(abc, translation)
                return np.concatenate([abc, [d, ]], axis=0)
            else:
                return None

        else:
            return None, None

    def toDict(self):
        return {
            'camMat': self.camMat.tolist(),
            'distortion': self.distortion.tolist(),
            'nDistortionParams': self.nDistortionParams,
            'T': self.T.tolist(),
            'V': self.V.tolist()
        }

            
    @classmethod
    def fromDict(cls, d):
        cam = cls(np.array(d['camMat'], dtype=float), np.array(d['distortion'], dtype=float), np.array(d['T'], dtype=float))
        cam.nDistortionParams = int(d['nDistortionParams'])
        return cam
    