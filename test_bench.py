# This code is for the popsicle stick manipulator and moving in from point to point with interpolation between
# 07/30/2020
# timothy.clark@ge.com

from numpy import *
import vpython as vp
import time

# Global variables for manipulator arm lengths
a1 = 1.107
a2 = 1.668
a3 = 1.662
a4 = 0.655
a5 = 0.208
v = 2


# Inverse kinematics to calculate rotational position for the first three joints: t1, t2, t3
def c_t1(pt):
    theta_1 = arctan2(pt[1], pt[0])
    return theta_1


def c_t2(pt):
    theta_1 = c_t1(pt)
    temp = pt.copy()

    # Define location of joint 4 relative to the input point of the ee location
    temp[0] -= a4 * cos(theta_1)
    temp[1] -= a4 * sin(theta_1)
    temp[2] -= a5

    j2_loc = array([cos(theta_1) * a1, sin(theta_1) * a1, 0])
    r = temp - j2_loc
    r_norm = linalg.norm(r)

    gamma_1 = arcsin(temp[2] / r_norm)
    phi_2 = arccos((a2 ** 2 + r_norm ** 2 - a3 ** 2) / (2 * a2 * r_norm))
    theta_2 = phi_2 - gamma_1

    return theta_2


def c_t3(pt):
    theta_1 = c_t1(pt)
    temp = pt.copy()

    # Define location of joint 4 relative to the input point of the ee location
    temp[0] -= a4 * cos(theta_1)
    temp[1] -= a4 * sin(theta_1)
    temp[2] -= a5

    j2_loc = array([cos(theta_1) * a1, sin(theta_1) * a1, 0])
    r = temp - j2_loc
    r_norm = linalg.norm(r)

    theta_3 = arccos((a3 ** 2 + a2 ** 2 - r_norm ** 2) / (2 * a3 * a2)) - pi

    return theta_3


def c_t4(pt):
    t1 = c_t1(pt)
    t2 = c_t2(pt)
    t3 = c_t3(pt)

    r0_4 = array([[cos(t1), 0, -sin(t1)], [sin(t1), 0, cos(t1)], [0, -1, 0]])
    r0_3 = array([[-sin(t2) * sin(t3) * cos(t1) + cos(t1) * cos(t2) * cos(t3),
                   -sin(t2) * cos(t1) * cos(t3) - sin(t3) * cos(t1) * cos(t2), -sin(t1)],
                  [-sin(t1) * sin(t2) * sin(t3) + sin(t1) * cos(t2) * cos(t3),
                   -sin(t1) * sin(t2) * cos(t3) - sin(t1) * sin(t3) * cos(t2), cos(t1)],
                  [-sin(t2) * cos(t3) - sin(t3) * cos(t2), sin(t2) * sin(t3) - cos(t2) * cos(t3), 0]])

    inv_r0_3 = linalg.inv(r0_3)

    r3_4 = dot(inv_r0_3, r0_4)
    theta_4 = arccos(r3_4[0][0])

    return theta_4


def c_vel(start_pt, end_pt):
    t1 = c_t1(start_pt)
    t2 = c_t2(start_pt)
    t3 = c_t3(start_pt)
    t4 = c_t3(start_pt)

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


def c_jpos(pt):
    t1 = c_t1(pt)
    t2 = c_t2(pt)
    t3 = c_t3(pt)
    t4 = c_t4(pt)

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

    if abs(a1 - a1_chk) > 0.0001:
        print("Check a1", a1, a1_chk)
    if abs(a2 - a2_chk) > 0.0001:
        print("Check a2", a2, a2_chk)
    if abs(a3 - a3_chk) > 0.0001:
        print("Check a3", a3, a3_chk)
    if abs(a4 - a4_chk) > 0.0001:
        print("Check a4", a4, a4_chk)
    if abs(a5 - a5_chk) > 0.0001:
        print("Check a5", a5, a5_chk)

    return j2_loc, j3_loc, j4_loc, pvt_loc, ee_loc


# Create starting point 1 and ending point 2

x1 = 0.848 + 0.5
y1 = 0
z1 = 1.4

x2 = 0.848 + 1.9
y2 = 0
z2 = 1.4

pt1 = array([x1, y1, z1])
pt2 = array([x2, y2, z2])

# Create a list of x,y,z points which represent a line
x = [x1]
y = [y1]
z = [z1]

resolution = 10
p = pt2 - pt1
norm = linalg.norm(p)
u = p / norm
delta = norm / resolution

x_inc = u[0] * delta
y_inc = u[1] * delta
z_inc = u[2] * delta

# Interpolate points between the start and end points
for i in range(1, resolution + 1):
    x.append(x[i - 1] + x_inc)
    y.append(y[i - 1] + y_inc)
    z.append(z[i - 1] + z_inc)
pt = array([x[0], y[0], z[0]])

vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(1, 0, 0), shaftwidth=.1, color=vp.color.red)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 1, 0), shaftwidth=.1, color=vp.color.green)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 1), shaftwidth=.1, color=vp.color.blue)
vp.label(pos=vp.vector(6, 0, z1), text='RPV Top 0')
vp.label(pos=vp.vector(0, 6, z1), text='90')
vp.label(pos=vp.vector(-6 * 4, 0, z1), text='180')
vp.label(pos=vp.vector(0, -6 * 4, z1), text='270')
vp.local_light(pos=vp.vector(0, 0, -10), color=vp.color.gray(0.2))

for j in range(len(x)):
    pt = array([x[j], y[j], z[j]])
    if j != 0:
        pt_old = array([x[j - 1], y[j - 1], z[j - 1]])
        temp = c_vel(pt_old, pt)
    else:
        temp = array([0, 0, 0])

    # Calculate position of j2, j3, j4, end effector
    j2, j3, j4, pvt, ee = c_jpos(pt)
    vp.curve(
        pos=[(0, 0, 0), (j2[0], j2[1], j2[2]), (j3[0], j3[1], j3[2]), (j4[0], j4[1], j4[2]),
             (pvt[0], pvt[1], pvt[2]), (ee[0], ee[1], ee[2])],
        radius=0.01)
    if round(float(x[j]), 4) != round(float(ee[0]), 4) or round(float(y[j]), 4) != round(float(ee[1]), 4) or round(
            float(z[j]), 4) != round(float(ee[2]), 4):
        print("ERROR")
    print("%2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.2f %2.5f %2.5f %2.5f %2.2f %2.5f %2.5f" % (
        x[j], y[j], z[j], rad2deg(c_t1(pt)), rad2deg(c_t2(pt)), rad2deg(c_t3(pt)),
        rad2deg(c_t4(pt)), rad2deg(temp[0]), rad2deg(temp[1]), rad2deg(temp[2]), ee[0], ee[1], ee[2]))
    time.sleep(.1)
