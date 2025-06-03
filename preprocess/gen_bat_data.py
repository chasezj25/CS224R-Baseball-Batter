import numpy as np
import math
import pickle
import csv
def main():
    session_swings = set()
    with open("../eligible_swings.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            if first:
                first = False
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            session_swings.add(line[keys["session_swing"]])
    count = 0
    final_data = {}
    with open("../data/data/full_sig/landmarks.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            #print(line)
            if count % 10000 == 0:
                print(count)
            count += 1
            if first:
                first = False
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            sess = line[keys["session_swing"]]
            if sess in session_swings:
                if not sess in final_data.keys():
                    final_data[sess] = []
                time = float(line[keys["time"]])
                sweet_spot_x = try_float(final_data, sess, line[keys["sweet_spot_x"]], "x")
                sweet_spot_y = try_float(final_data, sess, line[keys["sweet_spot_y"]], "y")
                sweet_spot_z = try_float(final_data, sess, line[keys["sweet_spot_z"]], "z")
                
                lhjc_x = try_float(final_data, sess,line[keys["lhjc_x"]], "lhjc_x") # left hand joint center
                lhjc_y = try_float(final_data, sess,line[keys["lhjc_y"]], "lhjc_y")
                lhjc_z = try_float(final_data, sess,line[keys["lhjc_z"]], "lhjc_z")

                lajc_x = try_float(final_data, sess,line[keys["lajc_x"]], "lajc_x")
                lajc_y = try_float(final_data, sess,line[keys["lajc_y"]], "lajc_y")
                lajc_z = try_float(final_data, sess,line[keys["lajc_z"]], "lajc_z")
                rajc_x = try_float(final_data, sess,line[keys["rajc_x"]], "rajc_x")
                rajc_y = try_float(final_data, sess,line[keys["rajc_y"]], "rajc_y")
                rajc_z = try_float(final_data, sess,line[keys["rajc_z"]], "rajc_z")

                x_ang, y_ang, z_ang = calc_pose(np.array([lhjc_x, lhjc_y, lhjc_z]), np.array([sweet_spot_x, sweet_spot_y, sweet_spot_z]))
                x_vel = 0
                y_vel = 0
                z_vel = 0
                x_ang_vel = 0
                y_ang_vel = 0
                z_ang_vel = 0

                if(len(final_data[sess]) != 0):
                    delta_t = time - final_data[sess][-1]["time"]
                    x_vel = (sweet_spot_x - final_data[sess][-1]["x"]) / delta_t
                    y_vel = (sweet_spot_y - final_data[sess][-1]["y"]) / delta_t
                    z_vel = (sweet_spot_z - final_data[sess][-1]["z"]) / delta_t

                    x_ang_vel = (x_ang - final_data[sess][-1]["x_ang"]) / delta_t
                    y_ang_vel = (y_ang - final_data[sess][-1]["y_ang"]) / delta_t
                    z_ang_vel = (z_ang - final_data[sess][-1]["z_ang"]) / delta_t
                next_entry = {
                    "time": time,
                    "x": sweet_spot_x,
                    "y": sweet_spot_y,
                    "z": sweet_spot_z,
                    "x_ang": x_ang,
                    "y_ang": y_ang,
                    "z_ang": z_ang,
                    "x_vel": x_vel,
                    "y_vel": y_vel,
                    "z_vel": z_vel,
                    "x_ang_vel": x_ang_vel,
                    "y_ang_vel": y_ang_vel,
                    "z_ang_vel": z_ang_vel,
                    "lhjc_x": lhjc_x,
                    "lhjc_y": lhjc_y,
                    "lhjc_z": lhjc_z,
                    "lajc_x": lajc_x,
                    "lajc_y": lajc_y,
                    "lajc_z": lajc_z,
                    "rajc_x": rajc_x,
                    "rajc_y": rajc_y,
                    "rajc_z": rajc_z
                }
                final_data[sess].append(next_entry)
    with open('../bat_data.pkl', 'wb') as file:
        pickle.dump(final_data, file)

#In case there is gaps in the data
def try_float(final_data, sess, str, key):
    try:
        ret = float(str)
    except:
        ret = final_data[sess][-1][key]
    return ret

def calc_pose(left_hand, bat_sweet_spot):
    axis = bat_sweet_spot - left_hand
    axis = axis / (np.sqrt(np.dot(axis, axis)))
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])
    proj_yz = proj(axis, y, z)
    proj_yz = np.array([proj_yz[1], proj_yz[2]])
    #proj_xz = proj(axis, x, z)
    #proj_xz = np.array([proj_xz[0], proj_xz[2]])
    proj_yx = proj(axis, x, y)
    proj_yx = np.array([proj_yx[0], proj_yx[1]])
    #z_2d = np.array([0,1])\
    y_2d_proj_yz = np.array([1,0])
    y_2d_proj_yx = np.array([0,1])
    multiplier = 1
    if cross_prod(proj_yz, y_2d_proj_yz) > 0:
        multiplier = -1
    angle_x = multiplier * cos_deg(y_2d_proj_yz, proj_yz)
    if cross_prod(proj_yx, y_2d_proj_yx) > 0:
        multiplier = -1
    else:
        multiplier = 1
    angle_z = multiplier * cos_deg(y_2d_proj_yx, proj_yx)
    angle_y = 0
    return angle_x, angle_y, angle_z

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