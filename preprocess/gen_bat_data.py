import numpy as np
import math
def main():
    pass

def calc_pose(left_hand, bat_sweet_spot):
    axis = bat_sweet_spot - left_hand
    axis = axis / (np.sqrt(np.dot(axis, axis)))
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])
    proj_yz = proj(axis, y, z)
    proj_yz = np.array([proj_yz[1], proj_yz[2]])
    proj_xz = proj(axis, x, z)
    proj_xz = np.array([proj_xz[0], proj_xz[2]])
    z_2d = np.array([0,1])
    multiplier = 1
    if cross_prod(proj_yz, z_2d) > 0:
        multiplier = -1
    angle_x = multiplier * cos_deg(z_2d, proj_yz)
    if cross_prod(proj_xz, z_2d) < 0:
        multiplier = -1
    else:
        multiplier = 1
    angle_y = multiplier * cos_deg(z_2d, proj_xz)
    return angle_x, angle_y, 0





def proj(axis, axis_1, axis_2):
    a1 = (np.dot(axis, axis_1) / np.dot(axis_1, axis_1)) * axis_1
    a2 = (np.dot(axis, axis_2) / np.dot(axis_2, axis_2)) * axis_2
    return  a1 + a2
    #stationary = np.array([0,0,1])
    #return (180 / math.pi) * np.arccos(np.dot(a, stationary) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(stationary,stationary)))) 

def cos_deg(a,b):
    return (180 / math.pi) * np.arccos(np.dot(a, b) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b))))

def cross_prod(a,b):
    return (a[0] * b[1]) - (a[1] * b[0])



if __name__ == "__main__":
    main()