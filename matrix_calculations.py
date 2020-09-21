from sympy import *
import time

start_time = time.time()
# Joint Angles
t1, t2, t3, t4 = symbols('t1 t2 t3 t4')

# Link Lengths
a1, a2, a3, a4 = symbols('a1 a2 a3 a4')

# Joint angular velocity
t1_dot, t2_dot, t3_dot, t4_dot = symbols('t1_dot t2_dot t3_dot t4_dot')

R0_1 = Matrix([[cos(t1), 0, -sin(t1)],
               [sin(t1), 0, cos(t1)],
               [0, -1, 0]])

R1_2 = Matrix([[cos(t2), -sin(t2), 0],
               [sin(t2), cos(t2), 0],
               [0, 0, 1]])

R2_3 = Matrix([[cos(t3), -sin(t3), 0],
               [sin(t3), cos(t3), 0],
               [0, 0, 1]])

R0_2 = R0_1 * R1_2
R0_3 = R0_2 * R2_3

# Define the Homogeneous Transformation Matrices
H0_1 = Matrix([[cos(t1), 0, -sin(t1), a1 * cos(t1)],
               [sin(t1), 0, cos(t1), a1 * sin(t1)],
               [0, -1, 0, 0],
               [0, 0, 0, 1]])

H1_2 = Matrix([[cos(t2), -sin(t2), 0, a2 * cos(t2)],
               [sin(t2), cos(t2), 0, a2 * sin(t2)],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

H2_3 = Matrix([[cos(t3), -sin(t3), 0, a3 * cos(t3)],
               [sin(t3), cos(t3), 0, a3 * sin(t3)],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

H3_4 = Matrix([[cos(t4), -sin(t4), 0, a4 * cos(t4)],
               [sin(t4), cos(t4), 0, a4 * sin(t4)],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

H0_2 = H0_1 * H1_2
H0_3 = H0_1 * H1_2 * H2_3
H0_4 = H0_1 * H1_2 * H2_3 * H3_4

print("H0_1:\n", H0_1)
print("H0_2:\n", H0_2)
print("H0_3:\n", H0_3)
print("H0_4:\n", H0_4)

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

# Calculate the displacement vectors
d0_1 = Matrix([[H0_1[0, 3]],
               [H0_1[1, 3]],
               [H0_1[2, 3]]])

d0_2 = Matrix([[H0_2[0, 3]],
               [H0_2[1, 3]],
               [H0_2[2, 3]]])

d0_3 = Matrix([[H0_3[0, 3]],
               [H0_3[1, 3]],
               [H0_3[2, 3]]])

d0_4 = Matrix([[H0_4[0, 3]],
               [H0_4[1, 3]],
               [H0_4[2, 3]]])

# Create temporary matrices for calculating the jacobian
vec = Matrix([[0],
              [0],
              [1]])
c1_a = Matrix([[(eye(3) * vec).cross(d0_4)[0]],
               [(eye(3) * vec).cross(d0_4)[1]],
               [(eye(3) * vec).cross(d0_4)[2]]])

c2_a = Matrix([[(R0_1 * vec).cross(d0_4 - d0_1)[0]],
               [(R0_1 * vec).cross(d0_4 - d0_1)[1]],
               [(R0_1 * vec).cross(d0_4 - d0_1)[2]]])

c2_b = Matrix([[(R0_1 * vec)[0]],
               [(R0_1 * vec)[1]],
               [(R0_1 * vec)[2]]])

c3_a = Matrix([[(R0_2 * vec).cross(d0_4 - d0_2)[0]],
               [(R0_2 * vec).cross(d0_4 - d0_2)[1]],
               [(R0_2 * vec).cross(d0_4 - d0_2)[2]]])

c3_b = Matrix([[(R0_2 * vec)[0]],
               [(R0_2 * vec)[1]],
               [(R0_2 * vec)[2]]])

c4_a = Matrix([[(R0_3 * vec).cross(d0_4 - d0_3)[0]],
               [(R0_3 * vec).cross(d0_4 - d0_3)[1]],
               [(R0_3 * vec).cross(d0_4 - d0_3)[2]]])

c4_b = Matrix([[(R0_3 * vec)[0]],
               [(R0_3 * vec)[1]],
               [(R0_3 * vec)[2]]])

# Create the Jacobian
jac = Matrix([[c1_a[0], c2_a[0], c3_a[0], c4_a[0]],
              [c1_a[1], c2_a[1], c3_a[1], c4_a[1]],
              [c1_a[2], c2_a[2], c3_a[2], c4_a[2]],
              [0, c2_b[0], c3_b[0], c4_b[0]],
              [0, c2_b[1], c3_b[1], c4_b[1]],
              [1, c2_b[2], c3_b[2], c4_b[2]]])

# Create theta dot variable matrix
t_dot = Matrix([[t1_dot],
                [t2_dot],
                [t3_dot],
                [t4_dot]])
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("x_dot:\n", (jac * t_dot)[0])
print("y_dot:\n", (jac * t_dot)[1])
print("z_dot:\n", (jac * t_dot)[2])
print("w_x:\n", (jac * t_dot)[3])
print("w_y:\n", (jac * t_dot)[4])
print("w_z:\n", (jac * t_dot)[5])

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

c1_a = Matrix([[(eye(3) * vec).cross(d0_3)[0]],
               [(eye(3) * vec).cross(d0_3)[1]],
               [(eye(3) * vec).cross(d0_3)[2]]])

c2_a = Matrix([[(R0_1 * vec).cross(d0_3 - d0_1)[0]],
               [(R0_1 * vec).cross(d0_3 - d0_1)[1]],
               [(R0_1 * vec).cross(d0_3 - d0_1)[2]]])

c2_b = Matrix([[(R0_1 * vec)[0]],
               [(R0_1 * vec)[1]],
               [(R0_1 * vec)[2]]])

c3_a = Matrix([[(R0_2 * vec).cross(d0_3 - d0_2)[0]],
               [(R0_2 * vec).cross(d0_3 - d0_2)[1]],
               [(R0_2 * vec).cross(d0_3 - d0_2)[2]]])

c3_b = Matrix([[(R0_2 * vec)[0]],
               [(R0_2 * vec)[1]],
               [(R0_2 * vec)[2]]])

# Create a smaller square version of the Jacobian in order to find the inverse
jac = Matrix([[c1_a[0], c2_a[0], c3_a[0]],
              [c1_a[1], c2_a[1], c3_a[1]],
              [c1_a[2], c2_a[2], c3_a[2]]])

# Create a cartesian velocity matrix
cart_dot = Matrix([[0.5],
                   [0],
                   [0]])

print("x_dot:\n", (jac * cart_dot)[0])
print("y_dot:\n", (jac * cart_dot)[1])
print("z_dot:\n", (jac * cart_dot)[2])

print((jac).subs([(t1, 0), (t2, pi / 4), (t3, -pi / 2), (t4, pi / 4), (a1, 1), (a2, 1), (a3, 1), (a4, 1)]).inv())
print("t_dot: \n",
      (jac).subs([(t1, 0), (t2, pi / 4), (t3, -pi / 2), (t4, pi / 4), (a1, 1), (a2, 1), (a3, 1),
                  (a4, 1)]).inv().evalf() * cart_dot)

# Calculate distances between transducer and joint4
ty, t1, a4, a5, x, y, z = symbols('ty t1 a4 a5 x y z')
ry = Matrix([[cos(ty), 0, sin(ty)],
             [0, 1, 0],
             [-sin(ty), 0, cos(ty)]])

rz = Matrix([[cos(t1), -sin(t1), 0],
             [sin(t1), cos(t1), 0],
             [0, 0, 1]])

pvt = Matrix([[a4],
              [0],
              [a5]])
pt = Matrix([[x], [y], [z]])

print("Location RPV:", pt - (rz * ry * pvt))

# Calculate new R0_4 matrix
ry = Matrix([[cos(ty), 0, sin(ty)],
             [0, 1, 0],
             [-sin(ty), 0, cos(ty)]])

rz = Matrix([[cos(t1), -sin(t1), 0],
             [sin(t1), cos(t1), 0],
             [0, 0, 1]])

r = Matrix([[1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]])

print("Rrz:", ( rz * r))

print("Location rpv:", pt - (ry.inv() * rz * pvt))
print("ry inv", ry.inv())