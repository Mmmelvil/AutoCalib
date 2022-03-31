import numpy as np
from math import pi as PI

def angles2R(yaw=0, pitch=0, roll=0, order='ypr'):
    """
    Transform yaw, pitch and roll angles in degrees into a rotation matrix
    In a cartesian coordinate system, yaw->pitch->roll corresponds to a rotation around z -> around y' -> around x''

    order: order in which to apply yaw, pitch and roll
    """
    yawRad, pitchRad, rollRad = [angle*2*PI/360 for angle in (yaw, pitch, roll)] # Transform to radians
    cosYaw, sinYaw, cosPitch, sinPitch, cosRoll, sinRoll = np.cos(yawRad), np.sin(yawRad), np.cos(pitchRad), np.sin(pitchRad), np.cos(rollRad), np.sin(rollRad)

    yawMat   = np.asarray([[cosYaw, -sinYaw, 0.], # Positive yaw = rotation from +x to +y
                           [sinYaw,  cosYaw, 0.],
                           [    0.,      0., 1.]])

    pitchMat = np.asarray([[cosPitch, 0., -sinPitch], # Positive pitch = rotation from +x to +z
                           [      0., 1.,        0.],
                           [sinPitch, 0.,  cosPitch]])

    rollMat  = np.asarray([[1.,      0.,       0.], # Positive roll = rotation from +z to +y
                           [0.,  cosRoll, sinRoll],
                           [0., -sinRoll, cosRoll]])

    mats = {'y': yawMat, 'p': pitchMat, 'r': rollMat}

    # rollMat@pitchMat =
    # np.array([[        cosPitch,                0.,         -sinPitch],
    #           [sinPitch*sinRoll,           cosRoll,  cosPitch*sinRoll],
    #           [sinPitch*cosRoll,          -sinRoll,  cosPitch*cosRoll]])

    # rollMat@yawMat =
    # np.array([[          cosYaw,           -sinYaw,                0.],
    #           [  sinYaw*cosRoll,    cosYaw*cosRoll,           sinRoll],
    #           [ -sinYaw*sinRoll,   -cosYaw*sinRoll,           cosRoll]])

    # pitchMat@rollMat =
    # np.array([[        cosPitch,  sinPitch*sinRoll, -sinPitch*cosRoll],
    #           [              0.,           cosRoll,           sinRoll],
    #           [        sinPitch, -cosPitch*sinRoll,  cosPitch*cosRoll]])

    # pitchMat@yawMat =
    # np.array([[ cosYaw*cosPitch,  -sinYaw*cosPitch,         -sinPitch],
    #           [          sinYaw,            cosYaw,                0.],
    #           [ cosYaw*sinPitch,  -sinYaw*sinPitch,          cosPitch]])

    # yawMat@rollMat =
    # np.array([[          cosYaw,   -sinYaw*cosRoll,   -sinYaw*sinRoll],
    #           [          sinYaw,    cosYaw*cosRoll,    cosYaw*sinRoll],
    #           [              0.,          -sinRoll,           cosRoll]])

    # yawMat@pitchMat =
    # np.array([[ cosYaw*cosPitch,           -sinYaw,  -cosYaw*sinPitch],
    #           [ sinYaw*cosPitch,            cosYaw,  -sinYaw*sinPitch],
    #           [        sinPitch,                0.,          cosPitch]])


    # ypr:
    # rollMat@pitchMat@yawMat =
    # np.array([[                            cosYaw*cosPitch,                            -sinYaw*cosPitch,                                   -sinPitch],
    #           [   sinYaw*cosRoll + cosYaw*sinPitch*sinRoll,    cosYaw*cosRoll - sinYaw*sinPitch*sinRoll,                            cosPitch*sinRoll],
    #           [  -sinYaw*sinRoll + cosYaw*sinPitch*cosRoll,   -cosYaw*sinRoll - sinYaw*sinPitch*cosRoll,                            cosPitch*cosRoll]])

    # pyr:
    # rollMat@yawMat@pitchMat =
    # np.array([[                            cosYaw*cosPitch,                                     -sinYaw,                            -cosYaw*sinPitch],
    #           [ sinYaw*cosPitch*cosRoll + sinPitch*sinRoll,                              cosYaw*cosRoll, -sinYaw*sinPitch*cosRoll + cosPitch*sinRoll],
    #           [-sinYaw*cosPitch*sinRoll + sinPitch*cosRoll,                             -cosYaw*sinRoll,  sinYaw*sinPitch*sinRoll + cosPitch*cosRoll]])

    # yrp:
    # pitchMat@rollMat@yawMat =
    # np.array([[  cosYaw*cosPitch + sinYaw*sinPitch*sinRoll,  -sinYaw*cosPitch + cosYaw*sinPitch*sinRoll,                           -sinPitch*cosRoll],
    #           [                             sinYaw*cosRoll,                              cosYaw*cosRoll,                                     sinRoll],
    #           [  cosYaw*sinPitch - sinYaw*cosPitch*sinRoll,  -sinYaw*sinPitch - cosYaw*cosPitch*sinRoll,                            cosPitch*cosRoll]])

    # ryp:
    # pitchMat@yawMat@rollMat =
    # np.array([[                            cosYaw*cosPitch, -sinYaw*cosPitch*cosRoll + sinPitch*sinRoll, -sinYaw*cosPitch*sinRoll - sinPitch*cosRoll],
    #           [                                     sinYaw,                              cosYaw*cosRoll,                              cosYaw*sinRoll],
    #           [                            cosYaw*sinPitch, -sinYaw*sinPitch*cosRoll - cosPitch*sinRoll, -sinYaw*sinPitch*sinRoll + cosPitch*cosRoll]])

    # pry:
    # yawMat@rollMat@pitchMat =
    # np.array([[  cosYaw*cosPitch - sinYaw*sinPitch*sinRoll,                             -sinYaw*cosRoll,  -cosYaw*sinPitch - sinYaw*cosPitch*sinRoll],
    #           [  sinYaw*cosPitch + cosYaw*sinPitch*sinRoll,                              cosYaw*cosRoll,  -sinYaw*sinPitch + cosYaw*cosPitch*sinRoll],
    #           [                           sinPitch*cosRoll,                                    -sinRoll,                            cosPitch*cosRoll]])

    # rpy:
    # yawMat@pitchMat@rollMat =
    # np.array([[                            cosYaw*cosPitch, cosYaw*sinPitch*sinRoll - sinYaw*cosRoll,      -cosYaw*sinPitch*cosRoll - sinYaw*sinRoll],
    #           [                            sinYaw*cosPitch, sinYaw*sinPitch*sinRoll + cosYaw*cosRoll,      -sinYaw*sinPitch*cosRoll + cosYaw*sinRoll],
    #           [                                   sinPitch,                        -cosPitch*sinRoll,                               cosPitch*cosRoll]])

    if len(order) != 3 or 'y' not in order or 'p' not in order or 'r' not in order: raise ValueError("angles2R: Error: order parameter must have 'y', 'p' and 'r' in it exactly once each.")
    return mats[order[2]]@mats[order[1]]@mats[order[0]]

def R2Angles(R, order='ypr'):
    """
    Transform a rotation matrix into yaw, pitch and roll angles in degrees
    Assume that yaw was applied first, then pitch, then roll

    order: order in which R was assembled (see angles2R) -> it will be disassembled in reverse order
    """
    if order=='ypr':
        yawRad   = np.arctan2(-R[0][1], R[0][0])
        pitchRad = np.arctan2(-R[0][2], np.sqrt(R[0][0]**2+R[0][1]**2))
        rollRad  = np.arctan2( R[1][2], R[2][2])
    elif order=='pyr':
        yawRad   = np.arctan2(-R[0][1], np.sqrt(R[0][0]**2+R[0][2]**2))
        pitchRad = np.arctan2(-R[0][2], R[0][0])
        rollRad  = np.arctan2(-R[2][1], R[1][1])
    elif order=='yrp':
        yawRad   = np.arctan2( R[1][0], R[1][1])
        pitchRad = np.arctan2(-R[0][2], R[2][2])
        rollRad  = np.arctan2( R[1][2], np.sqrt(R[1][0]**2+R[1][1]**2))
    elif order=='ryp':
        yawRad   = np.arctan2( R[1][0], np.sqrt(R[1][1]**2+R[1][2]**2))
        pitchRad = np.arctan2( R[2][0], R[0][0])
        rollRad  = np.arctan2( R[1][2], R[1][1])
    elif order=='pry':
        yawRad   = np.arctan2(-R[0][1], R[1][1])
        pitchRad = np.arctan2( R[2][0], R[2][2])
        rollRad  = np.arctan2(-R[2][1], np.sqrt(R[2][0]**2+R[2][2]**2))
    elif order=='rpy':
        yawRad   = np.arctan2( R[1][0], R[0][0])
        pitchRad = np.arctan2( R[2][0], np.sqrt(R[2][1]**2+R[2][2]**2))
        rollRad  = np.arctan2(-R[2][1], R[2][2])

    return (yawRad*360/(2*PI), pitchRad*360/(2*PI), rollRad*360/(2*PI))

