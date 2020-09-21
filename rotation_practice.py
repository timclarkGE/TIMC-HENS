import vpython as vp
from numpy import *

set_printoptions(precision=2, suppress=True)

a4 = 2
a5 = 1
j4 = (0, 0, 0)
pvt = (a4, 0, 0)
ee = (a4, 0, a5)

# Setup origin
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(2, 0, 0), shaftwidth=.04, color=vp.color.red)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 2, 0), shaftwidth=.04, color=vp.color.green)
vp.arrow(pos=vp.vector(0, 0, 0), axis=vp.vector(0, 0, 2), shaftwidth=.04, color=vp.color.blue)

vp.sphere(pos=vp.vector(ee[0], ee[1], ee[2]), color=vp.color.red, radius=0.1)
vp.curve(pos=[j4, pvt, ee], radius=0.05)

ty = pi / 4
tz = pi / 2

ry = array([[cos(ty), 0, sin(ty)],
            [0, 1, 0],
            [-sin(ty), 0, cos(ty)]])

ryz = array([[cos(tz) * cos(ty), -sin(tz), sin(ty) * cos(tz)],
             [sin(tz) * cos(ty), cos(tz), sin(tz) * sin(ty)],
             [-sin(ty), 0, cos(ty)]])

rzy = array([[cos(tz) * cos(ty), -sin(tz) * cos(ty), sin(ty)],
             [sin(tz), cos(tz), 0],
             [-sin(ty) * cos(tz), sin(tz) * sin(ty), cos(ty)]])

rb2e = array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

rrz = array([[cos(tz), 0, -sin(tz)], [sin(tz), 0, cos(tz)], [0, -1, 0]])

# # rotate about the y axis
# vp.curve(pos=[(dot(ry, j4)[0], dot(ry, j4)[1], dot(ry, j4)[2]), (dot(ry, pvt)[0], dot(ry, pvt)[1], dot(ry, pvt)[2]),
#               (dot(ry, ee)[0], dot(ry, ee)[1], dot(ry, ee)[2])], radius=0.05)
#
# # rotate about the y then z axis
# vp.curve(
#     pos=[(dot(rzy, j4)[0], dot(rzy, j4)[1], dot(rzy, j4)[2]), (dot(rzy, pvt)[0], dot(rzy, pvt)[1], dot(rzy, pvt)[2]),
#          (dot(rzy, ee)[0], dot(rzy, ee)[1], dot(rzy, ee)[2])], radius=0.05)
#
# # rotate about the z then y axis
# vp.sphere(pos=vp.vector(dot(ryz, ee)[0], dot(ryz, ee)[1], dot(ryz, ee)[2]), color=vp.color.green, radius=0.1)

# rotate by transforming the axis
# vp.curve(
#     pos=[(dot(rb2e, j4)[0], dot(rb2e, j4)[1], dot(rb2e, j4)[2]), (dot(rb2e, pvt)[0], dot(rb2e, pvt)[1], dot(rb2e, pvt)[2]),
#          (dot(rb2e, ee)[0], dot(rb2e, ee)[1], dot(rb2e, ee)[2])], radius=0.05)
#
# vp.sphere(pos=vp.vector(dot(rb2e, ee)[0], dot(rb2e, ee)[1], dot(rb2e, ee)[2]), color=vp.color.green, radius=0.1)
#
# vp.curve(
#     pos=[(dot(rrz, j4)[0], dot(rrz, j4)[1], dot(rrz, j4)[2]), (dot(rrz, pvt)[0], dot(rrz, pvt)[1], dot(rrz, pvt)[2]),
#          (dot(rrz, ee)[0], dot(rrz, ee)[1], dot(rrz, ee)[2])], radius=0.05)
#
# vp.sphere(pos=vp.vector(dot(rrz, ee)[0], dot(rrz, ee)[1], dot(rrz, ee)[2]), color=vp.color.blue, radius=0.1)

# Setup for calculation of J4 location given end effector location. The math should calculate that J4 is at the origin

t4 = pi / 4
t1 = pi / 4

r1 = cos(t4) * a4  # good
tp1 = pi / 2 - t4
tp2 = pi / 2 - tp1

r2 = sin(tp2) * a5  # good

h0 = cos(tp2) * a5
h1 = sin(t4) * a4 - h0

print("h1, calc:", h1, r1 - r2)

# if 0 > t4 > pi:
#     h1 *= -1

x = (r1 + r2) * cos(t1)
y = (r1 + r2) * sin(t1)
z = -(r1 - r2)

vp.sphere(pos=vp.vector(x, y, z), color=vp.color.green, radius=0.1)
print("norm scan pt:", round(linalg.norm(array([x, y, z])), 2))
print("norm of arm:", round(linalg.norm(array([a4, 0, a5])), 2))

xj4 = x - cos(t1) * (r1 + r2)
yj4 = y - sin(t1) * (r1 + r2)
zj4 = z + h1

print(xj4, yj4, zj4)
