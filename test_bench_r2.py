# This version can simulate the HENS manipulator on the RPV surface
# Added scan sequence and max joint angular velocity given true scan parameters
# 08/20/2020
# timothy.clark@ge.com

from numpy import *
import vpython as vp
from PIL import ImageGrab
import time

# Global variables for manipulator arm lengths
pipe_r = 10
track_h = 1
tool_height_j2 = 2.959
a1 = pipe_r + track_h + tool_height_j2
a2 = 12.0
a3 = 12.0
a4 = 3.5
a5 = 4.0

# Global variables for RPV dimensions
rpv_offset = 15.093  # Offset in z-direction from base frame to rpv OD
rpv_r = (251.0 / 2)  # RPV radius
rpv_h = 90  # RPV height
h, k = rpv_offset + rpv_r, 0  # Center of RPV

# Global scan parameters
min_reach = 15.63  # Inches, was 15.5
scan_res = 0.1  # Inches
scan_vel = 4.0  # Inches/second
scan_length = 13.8  # Inches 13.8 default, max 19.0
scan_index_angle = 1.8  # Degrees
sim_true_scan_speed = False  # Use Visual Python's rate() function, True/False

# Remove scientific notation from numpy print statements
set_printoptions(precision=2, suppress=True)


# Calculate the joint velocities using the Jacobian matrix
def c_vel(start_pt, end_pt, cfg):
    t1, t2, t3, t4 = c_angles(start_pt, cfg)

    x1, x2 = start_pt[0], end_pt[0]
    y1, y2 = start_pt[1], end_pt[1]
    z1, z2 = start_pt[2], end_pt[2]

    x_dot = scan_vel * (x2 - x1) / (sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    y_dot = scan_vel * (y2 - y1) / (sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    z_dot = scan_vel * (z2 - z1) / (sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))

    v_vel = array([[x_dot], [y_dot], [z_dot]])

    jacobian = array(
        [[-a1 * sin(t1) - a2 * sin(t1) * cos(t2) + a3 * sin(t1) * sin(t2) * sin(t3) - a3 * sin(t1) * cos(t2) * cos(t3),
          (-a2 * sin(t2) - a3 * sin(t2) * cos(t3) - a3 * sin(t3) * cos(t2)) * cos(t1),
          (-a3 * sin(t2) * cos(t3) - a3 * sin(t3) * cos(t2)) * cos(t1)],
         [a1 * cos(t1) + a2 * cos(t1) * cos(t2) - a3 * sin(t2) * sin(t3) * cos(t1) + a3 * cos(t1) * cos(t2) * cos(t3),
          (-a2 * sin(t2) - a3 * sin(t2) * cos(t3) - a3 * sin(t3) * cos(t2)) * sin(t1),
          (-a3 * sin(t2) * cos(t3) - a3 * sin(t3) * cos(t2)) * sin(t1)], [0, -(
                a2 * sin(t1) * cos(t2) - a3 * sin(t1) * sin(t2) * sin(t3) + a3 * sin(t1) * cos(t2) * cos(t3)) * sin(
            t1) - (a2 * cos(t1) * cos(t2) - a3 * sin(t2) * sin(t3) * cos(t1) + a3 * cos(t1) * cos(t2) * cos(t3)) * cos(
            t1),
                                                                          -(-a3 * sin(t1) * sin(t2) * sin(
                                                                              t3) + a3 * sin(
                                                                              t1) * cos(t2) * cos(t3)) * sin(t1) - (
                                                                                  -a3 * sin(t2) * sin(t3) * cos(
                                                                              t1) + a3 * cos(t1) * cos(t2) * cos(
                                                                              t3)) * cos(t1)]])
    result = dot(linalg.inv(jacobian), v_vel)

    return result


# Calculate the joint angles using pt: (X, Y, Z) and inverse kinematics
def c_angles(pt, cfg):
    # Find theta 1
    t1 = arctan2(pt[1], pt[0])

    # Calculate major and minor axes of ellipse
    a = abs(rpv_r / cos(pi / 2 - t1))
    b = rpv_r
    u = rpv_offset + rpv_r

    # Rotation matrix to rotate elliptical scan path into x-z plane
    r_z = array([[cos(-t1), -sin(-t1), 0],
                 [sin(-t1), cos(-t1), 0],
                 [0, 0, 1]])

    new_pt = zeros(3)

    # New rotated scan point
    new_pt[0] = dot(r_z, array([[pt[0]], [pt[1]], [pt[2]]]))[0]
    new_pt[1] = dot(r_z, array([[pt[0]], [pt[1]], [pt[2]]]))[1]
    new_pt[2] = dot(r_z, array([[pt[0]], [pt[1]], [pt[2]]]))[2]

    # Find slope of ellipse at scan point, check for divide by zero or imaginary
    radicand = -new_pt[2] ** 2 + 2 * u * new_pt[2] - u ** 2 + b ** 2
    if radicand <= 0:
        slope = inf
    else:
        slope = - a * (new_pt[2] - u) / (b * sqrt(radicand))

    # Calculate extra rotation angle from slope
    te = arctan(1 / slope)

    if (pi / 2 - arctan2(a5, a4)) <= te <= 0:
        print("Theta_extra :", round(rad2deg(te), 2))
        input("ERROR: Theta_extra is outside of range. Press Enter to continue...")

    # Find parameters to calculate j4 location
    h4 = cos(pi / 2 - te) * a4
    tp1 = te
    tp2 = pi / 2 - tp1

    h5 = sin(tp2) * a5

    r1 = sin(pi / 2 - te) * a4
    r2 = cos(tp2) * a5

    xj4 = pt[0] - cos(t1) * (r1 - r2)
    yj4 = pt[1] - sin(t1) * (r1 - r2)
    zj4 = pt[2] - (h4 + h5)

    j4_loc = array([xj4, yj4, zj4])

    # Find joint 2 location
    j2_loc = array([cos(t1) * a1, sin(t1) * a1, 0])
    r0 = j4_loc - j2_loc
    r0_norm = linalg.norm(r0)

    # Find theta 3
    t3 = arccos((a3 ** 2 + a2 ** 2 - r0_norm ** 2) / (2 * a3 * a2)) - pi

    # Find theta 2
    phi_2 = arccos((a2 ** 2 + r0_norm ** 2 - a3 ** 2) / (2 * a2 * r0_norm))

    # Configuration zero where wrist is above shoulder
    if cfg == 0:
        gamma_1 = arcsin(j4_loc[2] / r0_norm)
        t2 = phi_2 - gamma_1
    # Configuration one where wrist is below shoulder
    elif cfg == 1:
        r4 = array([j4_loc[0], j4_loc[1]])
        r4_norm = linalg.norm(r4)
        beta_1 = arccos((a1 - r4_norm) / r0_norm)
        t2 = phi_2 + beta_1 - pi


    # Find theta 4
    r0_4 = array([[cos(t1), 0, -sin(t1)], [sin(t1), 0, cos(t1)], [0, -1, 0]])

    r0_3 = array([[-sin(t2) * sin(t3) * cos(t1) + cos(t1) * cos(t2) * cos(t3),
                   -sin(t2) * cos(t1) * cos(t3) - sin(t3) * cos(t1) * cos(t2), -sin(t1)],
                  [-sin(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t2) * cos(t3),
                   -sin(t1) * sin(t2) * cos(t3) - sin(t1) * sin(t3) * cos(t2), cos(t1)],
                  [-sin(t2) * cos(t3) - sin(t3) * cos(t2), sin(t2) * sin(t3) - cos(t2) * cos(t3), 0]])

    inv_r0_3 = linalg.inv(r0_3)

    # Find the rotation matrix between frames 3 and 4
    r3_4 = dot(inv_r0_3, r0_4)

    # Subtract the extra theta angle to press the end effector on the RPV
    t4 = arccos(r3_4[0][0]) - te

    return t1, t2, t3, t4


# Calculate joint positions in Cartesian space using forward kinematics and the homogeneous transformation matrix
def c_jpos(pt, cfg):
    t1, t2, t3, t4 = c_angles(pt, cfg)

    joint = array([[0], [0], [0], [1]])

    h0_1 = array([[cos(t1), 0, -sin(t1), a1 * cos(t1)],
                  [sin(t1), 0, cos(t1), a1 * sin(t1)],
                  [0, -1, 0, 0],
                  [0, 0, 0, 1]])

    h0_2 = array([[cos(t1) * cos(t2), -sin(t2) * cos(t1), -sin(t1), a1 * cos(t1) + a2 * cos(t1) * cos(t2)],
                  [sin(t1) * cos(t2), -sin(t1) * sin(t2), cos(t1), a1 * sin(t1) + a2 * sin(t1) * cos(t2)],
                  [-sin(t2), -cos(t2), 0, -a2 * sin(t2)], [0, 0, 0, 1]])

    h0_3 = array([[-sin(t2) * sin(t3) * cos(t1) + cos(t1) * cos(t2) * cos(t3),
                   -sin(t2) * cos(t1) * cos(t3) - sin(t3) * cos(t1) * cos(t2), -sin(t1),
                   a1 * cos(t1) + a2 * cos(t1) * cos(t2) - a3 * sin(t2) * sin(t3) * cos(t1) + a3 * cos(t1) * cos(
                       t2) * cos(t3)], [-sin(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t2) * cos(t3),
                                        -sin(t1) * sin(t2) * cos(t3) - sin(t1) * sin(t3) * cos(t2), cos(t1),
                                        a1 * sin(t1) + a2 * sin(t1) * cos(t2) - a3 * sin(t1) * sin(t2) * sin(
                                            t3) + a3 * sin(t1) * cos(t2) * cos(t3)],
                  [-sin(t2) * cos(t3) - sin(t3) * cos(t2), sin(t2) * sin(t3) - cos(t2) * cos(t3), 0,
                   -a2 * sin(t2) - a3 * sin(t2) * cos(t3) - a3 * sin(t3) * cos(t2)], [0, 0, 0, 1]])

    h0_4 = array([[(-sin(t2) * sin(t3) * cos(t1) + cos(t1) * cos(t2) * cos(t3)) * cos(t4) + (
            -sin(t2) * cos(t1) * cos(t3) - sin(t3) * cos(t1) * cos(t2)) * sin(t4),
                   -(-sin(t2) * sin(t3) * cos(t1) + cos(t1) * cos(t2) * cos(t3)) * sin(t4) + (
                           -sin(t2) * cos(t1) * cos(t3) - sin(t3) * cos(t1) * cos(t2)) * cos(t4), -sin(t1),
                   a1 * cos(t1) + a2 * cos(t1) * cos(t2) - a3 * sin(t2) * sin(t3) * cos(t1) + a3 * cos(t1) * cos(
                       t2) * cos(t3) + a4 * (-sin(t2) * sin(t3) * cos(t1) + cos(t1) * cos(t2) * cos(t3)) * cos(
                       t4) + a4 * (-sin(t2) * cos(t1) * cos(t3) - sin(t3) * cos(t1) * cos(t2)) * sin(t4)], [
                      (-sin(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t2) * cos(t3)) * cos(t4) + (
                              -sin(t1) * sin(t2) * cos(t3) - sin(t1) * sin(t3) * cos(t2)) * sin(t4),
                      -(-sin(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t2) * cos(t3)) * sin(t4) + (
                              -sin(t1) * sin(t2) * cos(t3) - sin(t1) * sin(t3) * cos(t2)) * cos(t4), cos(t1),
                      a1 * sin(t1) + a2 * sin(t1) * cos(t2) - a3 * sin(t1) * sin(t2) * sin(t3) + a3 * sin(t1) * cos(
                          t2) * cos(t3) + a4 * (-sin(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t2) * cos(t3)) * cos(
                          t4) + a4 * (-sin(t1) * sin(t2) * cos(t3) - sin(t1) * sin(t3) * cos(t2)) * sin(t4)], [
                      (sin(t2) * sin(t3) - cos(t2) * cos(t3)) * sin(t4) + (
                              -sin(t2) * cos(t3) - sin(t3) * cos(t2)) * cos(t4),
                      (sin(t2) * sin(t3) - cos(t2) * cos(t3)) * cos(t4) - (
                              -sin(t2) * cos(t3) - sin(t3) * cos(t2)) * sin(t4), 0,
                      -a2 * sin(t2) - a3 * sin(t2) * cos(t3) - a3 * sin(t3) * cos(t2) + a4 * (
                              sin(t2) * sin(t3) - cos(t2) * cos(t3)) * sin(t4) + a4 * (
                              -sin(t2) * cos(t3) - sin(t3) * cos(t2)) * cos(t4)], [0, 0, 0, 1]])

    j2_loc = dot(h0_1, joint)
    j3_loc = dot(h0_2, joint)
    j4_loc = dot(h0_3, joint)
    pvt_loc = dot(h0_4, joint)
    ee_loc = dot(h0_4, array([[0], [-a5], [0], [1]]))

    a1_chk = linalg.norm(j2_loc - joint)
    a2_chk = linalg.norm(j3_loc - j2_loc)
    a3_chk = linalg.norm(j4_loc - j3_loc)
    a4_chk = linalg.norm(pvt_loc - j4_loc)
    a5_chk = linalg.norm(ee_loc - pvt_loc)

    # Check that the link lengths are not longer/shorter than specified
    if abs(a1 - a1_chk) > 0.0001:
        print("ERROR: Check a1", a1, a1_chk)
    if abs(a2 - a2_chk) > 0.0001:
        print("ERROR: Check a2", a2, a2_chk)
    if abs(a3 - a3_chk) > 0.0001:
        print("ERROR: Check a3", a3, a3_chk)
    if abs(a4 - a4_chk) > 0.0001:
        print("ERROR: Check a4", a4, a4_chk)
    if abs(a5 - a5_chk) > 0.0001:
        print("ERROR: Check a5", a5, a5_chk)

    return j2_loc, j3_loc, j4_loc, pvt_loc, ee_loc


# Calculate scan point array with inputs scan length (in) and index angle (deg), CW scanning only, only 360 rotation
def calc_scan_pts(scan_length, index_angle):
    print("Making HENS Scan Points", end='')
    start_time = time.time()

    # Create empty array for saving scan points
    pa = array([[], [], []])

    total_pts = int(360 / index_angle)

    # Enter loop to create scan points, i*index_angle = scan angle in degrees
    for i in range(0, total_pts):

        a = deg2rad(i * index_angle)

        # Scan line in to out
        if i % 2 == 0:
            pos_start = min_reach
            m = 1
        # Scan line out to in
        else:
            pos_start = min_reach + scan_length
            m = -1

        for j in range(0, int(scan_length / scan_res)):
            # Add to the previous point the x increment based on the scan angle
            new_x = (pa[0, int(pa.size / len(pa)) - 1] + m * scan_res * cos(a) if j >= 1 else pos_start * cos(a))
            new_y = (pa[1, int(pa.size / len(pa)) - 1] + m * scan_res * sin(a) if j >= 1 else pos_start * sin(a))
            new_z = -(sqrt(rpv_r ** 2 - new_y ** 2) - h)
            pa = append(pa, array([[new_x], [new_y], [new_z]]), axis=1)

        # Outer arc
        if i % 2 == 0:
            pos_start = min_reach + scan_length
        # Inner arc
        else:
            pos_start = min_reach

        # Create vector from start of arc to end
        x_pt_a = pos_start * cos(a)
        y_pt_a = pos_start * sin(a)
        z_pt_a = -(sqrt(rpv_r ** 2 - y_pt_a ** 2) - h)

        next_angle = deg2rad((i + 1) * index_angle)
        x_pt_b = pos_start * cos(next_angle)
        y_pt_b = pos_start * sin(next_angle)
        z_pt_b = -(sqrt(rpv_r ** 2 - y_pt_b ** 2) - h)

        pt_a = array([[x_pt_a], [y_pt_a], [z_pt_a]])
        pt_b = array([[x_pt_b], [y_pt_b], [z_pt_b]])
        vec = pt_b - pt_a
        vec_mag = linalg.norm(vec)
        vec /= vec_mag  # create unit vector

        for j in range(0, int(vec_mag / scan_res)):
            new_x = (pa[0, int(pa.size / len(pa)) - 1] + vec[0, 0] * scan_res if j >= 1 else x_pt_a)
            new_y = (pa[1, int(pa.size / len(pa)) - 1] + vec[1, 0] * scan_res if j >= 1 else y_pt_a)
            new_z = (pa[2, int(pa.size / len(pa)) - 1] + vec[2, 0] * scan_res if j >= 1 else z_pt_a)
            pa = append(pa, array([[new_x], [new_y], [new_z]]), axis=1)

        if (i % int(total_pts * 0.1)) == 0:
            print(".", end='')

    print("\nScan Points Complete: %2.2f seconds" % float((time.time() - start_time)))

    return pa


# Check calculated point against scan point
def calc_pos_err_check(cp, sp):
    err_res = 3
    xcp = round(float(cp[0]), err_res)
    ycp = round(float(cp[1]), err_res)
    zcp = round(float(cp[2]), err_res)

    xsp = round(float(sp[0]), err_res)
    ysp = round(float(sp[1]), err_res)
    zsp = round(float(sp[2]), err_res)

    if xcp != xsp or ycp != ysp or zcp != zsp:
        return True
    else:
        return False


def print_joint_set(t1, t2, t3, t4):
    t1 = round(rad2deg(t1), 2)
    t2 = round(rad2deg(t2), 2)
    t3 = round(rad2deg(t3), 2)
    t4 = round(rad2deg(t4), 2)

    print("Joint Set: %2.2f %2.2f %2.2f %2.2f" % (t1, t2, t3, t4))
    return 0


# Set the stage for Visual Python: Create RPV and scan pipe for HENS
rpv = vp.cylinder(pos=vp.vector(-rpv_h / 2, 0, rpv_r + rpv_offset), axis=vp.vector(rpv_h, 0, 0), radius=rpv_r,
                  color=vp.color.cyan)
scan_pipe = vp.cylinder(pos=vp.vector(0, 0, 0.1), axis=vp.vector(0, 0, rpv_offset), radius=a1 * .8,
                        color=vp.color.red)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(2, 0, 0), shaftwidth=.4, color=vp.color.red)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 2, 0), shaftwidth=.4, color=vp.color.green)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 2), shaftwidth=.4, color=vp.color.blue)

vp.label(pos=vp.vector(a1 * 4, 0, rpv_offset), text='0')
vp.label(pos=vp.vector(0, a1 * 4, rpv_offset), text='90')
vp.label(pos=vp.vector(-a1 * 4, 0, rpv_offset), text='180')
vp.label(pos=vp.vector(0, -a1 * 4, rpv_offset), text='270')
vp.local_light(pos=vp.vector(0, 0, -10), color=vp.color.gray(0.2))

scan_pts = calc_scan_pts(scan_length, scan_index_angle)
num_pts = int(size(scan_pts) / len(scan_pts))
# vp.scene.camera.pos = vp.vector(scan_pts[0][num_pts - 1], scan_pts[1][num_pts - 1], scan_pts[2][num_pts - 1])

# Loop through scan points, assume configuration 1 first
config = 1
for j in range(num_pts):
    # Set the loop rate to be equivalent to the desired scan velocity.
    if sim_true_scan_speed:
        vp.rate(scan_vel / scan_res)

    j2, j3, j4, pvt, ee = c_jpos(scan_pts[:, j], config)
    t1, t2, t3, t4 = c_angles(scan_pts[:, j], config)

    #  Check if end effector is in the scan location
    if calc_pos_err_check(ee, scan_pts[:, j]):
        config = ~config
        j2, j3, j4, pvt, ee = c_jpos(scan_pts[:, j], 1)
        t1, t2, t3, t4 = c_angles(scan_pts[:, j], 1)

    if calc_pos_err_check(ee, scan_pts[:, j]):
        # If problems, print the joint set and leave a red curve of the manipulator
        print_joint_set(t1, t2, t3, t4)
        vp.curve(
            pos=[(0, 0, 0), (j2[0], j2[1], j2[2]), (j3[0], j3[1], j3[2]), (j4[0], j4[1], j4[2]),
                 (pvt[0], pvt[1], pvt[2]), (ee[0], ee[1], ee[2])], radius=0.25, color=vp.color.red)

    # Visual Python: Show scan points in green and the manipulator as a grey curve
    if j == 0:
        hens = vp.curve(
            pos=[(0, 0, 0), (j2[0], j2[1], j2[2]), (j3[0], j3[1], j3[2]), (j4[0], j4[1], j4[2]),
                 (pvt[0], pvt[1], pvt[2]), (ee[0], ee[1], ee[2])], radius=0.25)
        point = vp.sphere(pos=vp.vector(scan_pts[0, j], scan_pts[1, j], scan_pts[2, j]), color=vp.color.green,
                          radius=0.25, make_trail=True, retain=350)
        max_t1_vel = 0
        max_t2_vel = 0
        max_t3_vel = 0
        update_flag = 0

    else:
        # Update the position of the arm and the scan point
        hens.modify(1, x=j2[0], y=j2[1], z=j2[2])
        hens.modify(2, x=j3[0], y=j3[1], z=j3[2])
        hens.modify(3, x=j4[0], y=j4[1], z=j4[2])
        hens.modify(4, x=pvt[0], y=pvt[1], z=pvt[2])
        hens.modify(5, x=ee[0], y=ee[1], z=ee[2])

        point.pos = vp.vector(scan_pts[0, j], scan_pts[1, j], scan_pts[2, j])

        vel_vector = c_vel(scan_pts[:, j - 1], scan_pts[:, j], config)
        if abs(vel_vector[0]) > abs(max_t1_vel):
            max_t1_vel = vel_vector[0]
            update_flag = 1
        if abs(vel_vector[1]) > abs(max_t2_vel):
            max_t2_vel = vel_vector[1]
            update_flag = 1
        if abs(vel_vector[2]) > abs(max_t3_vel):
            max_t3_vel = vel_vector[2]
            update_flag = 1

        if update_flag:
            print(rad2deg(max_t1_vel), rad2deg(max_t2_vel), rad2deg(max_t3_vel))
            update_flag = 0
        # if j % 20 == 0:
        #     im = ImageGrab.grab((20, 278, 1620, 278 + 1000))
        #     im.save('filename' + str(num_pts-j) + '.jpg')
