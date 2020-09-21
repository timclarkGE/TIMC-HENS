# This version can simulate the HENS manipulator on the RPV surface
# 08/14/2020
# timothy.clark@ge.com

from numpy import *
import vpython as vp
import time
import matplotlib.pyplot as plt

# Global variables for manipulator arm lengths
a1 = 10.0
a2 = 12.0 + 5
a3 = 12.0 + 5
a4 = 5.5
a5 = 4.0
v = 2.0

# Global variables for RPV dimensions
rpv_offset = 15.093  # Offset in z-direction from base frame to rpv OD
rpv_r = (251.0 / 2)  # RPV radius
rpv_h = 90  # RPV height

set_printoptions(precision=2, suppress=True)


# Calculate the joint velocities using the Jacobian matrix
def c_vel(start_pt, end_pt):
    t1, t2, t3, t4 = c_angles(start_pt)

    x1, x2 = start_pt[0], end_pt[0]
    y1, y2 = start_pt[1], end_pt[1]
    z1, z2 = start_pt[2], end_pt[2]

    x_dot = v * (x2 - x1) / (sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    y_dot = v * (y2 - y1) / (sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    z_dot = v * (z2 - z1) / (sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))

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


# Calculate the joint angles using pt: (X, Y, Z)
def c_angles(pt):
    # Find theta 1
    t1 = arctan2(pt[1], pt[0])

    a = rpv_r / cos(pi/2-t1)
    b = rpv_r
    u = rpv_offset + rpv_r

    # Rotation matrix for the scan points
    r_z = array([[cos(-t1), -sin(-t1), 0],
                 [sin(-t1), cos(-t1), 0],
                 [0, 0, 1]])

    new_pt = zeros(3)

    new_pt[0] = dot(r_z, array([[pt[0]], [pt[1]], [pt[2]]]))[0]
    new_pt[1] = dot(r_z, array([[pt[0]], [pt[1]], [pt[2]]]))[1]
    new_pt[2] = dot(r_z, array([[pt[0]], [pt[1]], [pt[2]]]))[2]

    try:
        slope = - a * (new_pt[2] - u) / (b * sqrt(-new_pt[2] ** 2 + 2 * u * new_pt[2] - u ** 2 + b ** 2))
    except:
        print("Error with Slope")

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

    vp.sphere(pos=vp.vector(j4_loc[0], j4_loc[1], j4_loc[2]), color=vp.color.red, radius=0.1)

    # Find joint 2 location
    j2_loc = array([cos(t1) * a1, sin(t1) * a1, 0])
    r = j4_loc - j2_loc  # Technically j4 -j2
    r_norm = linalg.norm(r)
    gamma_1 = arcsin(j4_loc[2] / r_norm)
    phi_2 = arccos((a2 ** 2 + r_norm ** 2 - a3 ** 2) / (2 * a2 * r_norm))
    t2 = phi_2 - gamma_1

    # Find theta 3
    t3 = arccos((a3 ** 2 + a2 ** 2 - r_norm ** 2) / (2 * a3 * a2)) - pi

    # Find theta 4
    r0_4 = array([[cos(t1), 0, -sin(t1)], [sin(t1), 0, cos(t1)], [0, -1, 0]])

    r0_3 = array([[-sin(t2) * sin(t3) * cos(t1) + cos(t1) * cos(t2) * cos(t3),
                   -sin(t2) * cos(t1) * cos(t3) - sin(t3) * cos(t1) * cos(t2), -sin(t1)],
                  [-sin(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t2) * cos(t3),
                   -sin(t1) * sin(t2) * cos(t3) - sin(t1) * sin(t3) * cos(t2), cos(t1)],
                  [-sin(t2) * cos(t3) - sin(t3) * cos(t2), sin(t2) * sin(t3) - cos(t2) * cos(t3), 0]])

    inv_r0_3 = linalg.inv(r0_3)

    r3_4 = dot(inv_r0_3, r0_4)
    t4 = arccos(r3_4[0][0]) - te

    return t1, t2, t3, t4


# Calculate joint positions in Cartesian space using forward kinematics and the homogeneous transformation matrix
def c_jpos(pt):
    t1, t2, t3, t4 = c_angles(pt)

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


# Set the stage for Visual Python: Create RPV and scan pipe for HENS
rpv = vp.cylinder(pos=vp.vector(-rpv_h / 2, 0, rpv_r + rpv_offset), axis=vp.vector(rpv_h, 0, 0), radius=rpv_r,
                  color=vp.color.cyan)
scan_pipe = vp.cylinder(pos=vp.vector(0, 0, 0.1), axis=vp.vector(0, 0, rpv_offset), radius=a1 * .8,
                        color=vp.color.red)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(2, 0, 0), shaftwidth=.4, color=vp.color.red)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 2, 0), shaftwidth=.4, color=vp.color.green)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 2), shaftwidth=.4, color=vp.color.blue)
vp.label(pos=vp.vector(a1 * 12, 0, rpv_offset), text='RPV Top')
vp.label(pos=vp.vector(a1 * 4, 0, rpv_offset), text='0')
vp.label(pos=vp.vector(0, a1 * 4, rpv_offset), text='90')
vp.label(pos=vp.vector(-a1 * 4, 0, rpv_offset), text='180')
vp.label(pos=vp.vector(0, -a1 * 4, rpv_offset), text='270')
vp.local_light(pos=vp.vector(0, 0, -10), color=vp.color.gray(0.2))
# vp.box(pos=vp.vector(0, 0, rpv_offset), length=100, height=100, width=0.001)

# Initiate scan variables

scan_angle = deg2rad(89)

start_pt = array([[(a1 + 5.5) * cos(scan_angle)], [(a1 + 5.5) * sin(scan_angle)], [rpv_offset]])  # Scan starting point
scan_dist = 25
inc = .25  # scan point resolution
num_pts = int(scan_dist / inc)
h, k = rpv_offset + rpv_r, 0  # Center of RPV

scan_pts = zeros((3, num_pts))
scan_pts[:, 0] = start_pt[:, 0]

########################
# Generate scan points #
########################

for i in range(1, num_pts):
    scan_pts[0][i] = (inc * cos(scan_angle) + scan_pts[0][i - 1]) * 1
    scan_pts[1][i] = (inc * sin(scan_angle) + scan_pts[1][i - 1]) * 1
    scan_pts[2][i] = -(sqrt(rpv_r ** 2 - scan_pts[1][i] ** 2) - h)

vp.scene.camera.pos = vp.vector(scan_pts[0][num_pts - 1], scan_pts[1][num_pts - 1], scan_pts[2][num_pts - 1])

# Loop through scan points
for j in range(num_pts):
    j2, j3, j4, pvt, ee = c_jpos(scan_pts[:, j])
    t1, t2, t3, t4 = c_angles(scan_pts[:, j])

    # Check if end effector is in the scan location
    if round(float(scan_pts[0, j]), 4) != round(float(ee[0]), 4) or round(float(scan_pts[1, j]), 4) != round(
            float(ee[1]), 4) or round(float(scan_pts[2, j]), 4) != round(float(ee[2]), 4):
        print("ERROR POSITION %2.2f %2.2f %2.2f" % (ee[0], ee[1], ee[2]))

    # Print the calculated results
    print("%2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f " % (
        scan_pts[0, j], scan_pts[1, j], scan_pts[2, j], rad2deg(t1), rad2deg(t2),
        rad2deg(t3), rad2deg(t4), ee[0], ee[1], ee[2]))

    # Visual Python: Show scan points in green and the manipulator as a grey curve
    vp.sphere(pos=vp.vector(scan_pts[0, j], scan_pts[1, j], scan_pts[2, j]), color=vp.color.green, radius=0.1)

    vp.curve(
        pos=[(0, 0, 0), (j2[0], j2[1], j2[2]), (j3[0], j3[1], j3[2]), (j4[0], j4[1], j4[2]),
             (pvt[0], pvt[1], pvt[2]), (ee[0], ee[1], ee[2])], radius=0.05)
    time.sleep(.02)
