import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

# 3D animation 

import time

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make the X, Y meshgrid.
xs = np.linspace(-1, 1, 50)
ys = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(xs, ys)

# Set the z axis limits, so they aren't recalculated each frame.
ax.set_zlim(-1, 1)

# Begin plotting.
wframe = None
tstart = time.time()
for phi in np.linspace(0, 180. / np.pi, 100):
    # If a line collection is already remove it before drawing.
    if wframe:
        wframe.remove()
    # Generate data.
    Z = np.cos(2 * np.pi * X + phi) * (1 - np.hypot(X, Y))
    # Plot the new wireframe and pause briefly before continuing.
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    plt.pause(.001)

print('Average FPS: %f' % (100 / (time.time() - tstart)))

# end 3D animation

# Demonstrate the use of "from mpl_toolkits.mplot3d import Axes3D" for 3D plotting

# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Define data for a 3D plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

# Show 3D plot for unit sphere with origin (0, 0, 0) and radius 1 

# Define the sphere
phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Plot the sphere using mpl_toolkits.mplot3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.6)

# Plot the origin point (0, 0, 0)
ax.scatter(0, 0, 0, color='r', label='Origin (0, 0, 0)')
ax.text(0, 0, 0, "(0, 0, 0)", color='r')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjust perspective (elevation and azimuth)
ax.view_init(elev=40, azim=60)  # Higher elevation and rotated azimuth

# Show plot
plt.show()

## End of 3D shape plotting

# Show vectors in 3D plot

# Clear the plot

# Clear the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Define the vectors

# Origin
O = np.array([0, 0, 0])
A = np.array([1, 2, 3])
B = np.array([3, 2, 1])
C = np.array([2, 3, 1])
D = np.array([1, 3, 2])

# Plot vectors (with arrows) OA, OB, OC, OD, and OA -> AB -> BC -> CD -> DA -> DO on 3D plane 

# Plot vectors with arrows
ax.quiver(O[0], O[1], O[2], A[0], A[1], A[2], color='r')
ax.quiver(O[0], O[1], O[2], B[0], B[1], B[2], color='g')
ax.quiver(O[0], O[1], O[2], C[0], C[1], C[2], color='b')
ax.quiver(O[0], O[1], O[2], D[0], D[1], D[2], color='m')

# Plot path OA -> AB -> BC -> CD -> DA -> DO
ax.quiver(A[0], A[1], A[2], B[0] - A[0], B[1] - A[1], B[2] - A[2], color='r', linestyle='dashed')
ax.quiver(B[0], B[1], B[2], C[0] - B[0], C[1] - B[1], C[2] - B[2], color='g', linestyle='dashed')
ax.quiver(C[0], C[1], C[2], D[0] - C[0], D[1] - C[1], D[2] - C[2], color='b', linestyle='dashed')
ax.quiver(D[0], D[1], D[2], O[0] - D[0], O[1] - D[1], O[2] - D[2], color='m', linestyle='dashed')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

# Show wireframe plot for a 3D surface

# Define the surface
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Plot the wireframe
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, color='k')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

# Demo use of Axes3D.plot_surface(X, Y, Z, *args, **kwargs)
# Define the surface
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
Z = np.cos(np.sqrt(X**2 + Y**2))

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

# Show a triangular prism in 3D plot
# TODO: 

# Show a pyramid in 3D plot
# Define the vertices of the pyramid
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]])

# Define the faces of the pyramid
faces = [[vertices[j] for j in [0, 1, 4]],
         [vertices[j] for j in [1, 2, 4]],
         [vertices[j] for j in [2, 3, 4]],
         [vertices[j] for j in [3, 0, 4]],
         [vertices[j] for j in [0, 1, 2, 3]]]

# Plot the pyramid
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each face
for face in faces:
    face = np.array(face)
    ax.plot_trisurf(face[:, 0], face[:, 1], face[:, 2], color='cyan', alpha=0.6, edgecolor='r')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()


# Show a cylinder in 3D plot
# Define the cylinder
theta = np.linspace(0, 2.0 * np.pi, 100)
z = np.linspace(0, 1, 100)
theta, z = np.meshgrid(theta, z)
x = np.cos(theta)
y = np.sin(theta)

# Plot the cylinder
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='orange', alpha=0.6)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

# Show a cone in 3D plot
# Define the cone
theta = np.linspace(0, 2.0 * np.pi, 100)
z = np.linspace(0, 1, 100)
theta, z = np.meshgrid(theta, z)
x = z * np.cos(theta)
y = z * np.sin(theta)

# Plot the cone
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='purple', alpha=0.6)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

# Show a sphere in 3D plot
# Define the sphere
phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Plot the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.6)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()

# Show a semi-sphere in 3D plot
# Define the semi-sphere
phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi/2:50j]
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Plot the semi-sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.6)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()


""" 
Demonstrate 3D Geometry

1. Defines points A, B, C, D in 3D space (with coords of form (x, y, z))
3. Find distance from origin O to A
2. Find distance between two points 
4. Find midpoint between A and B

Show on 3D plane: 
5. Show A, B, C, D on plane 
6. Show unit vectors i, j, k on the plan 

"""

# Define points A, B, C, D in 3D space
A = np.array([1, 2, 0])
B = np.array([2, 1, 0])
C = np.array([0, 0, 1])
D = np.array([2, 0, 0])

# Find distance from origin O to A
O = np.array([0, 0, 0])
distance_OA = np.linalg.norm(A - O)
print(f"Distance from O to A: {distance_OA}")

# Find distance between two points
distance_AB = np.linalg.norm(B - A)
print(f"Distance between A and B: {distance_AB}")

# Find midpoint between A and B
midpoint_AB = (A + B) / 2
print(f"Midpoint between A and B: {midpoint_AB}")

# Show A, B, C, D on plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*O, color='r', label='O')
ax.scatter(*A, color='r', label='A')
ax.scatter(*B, color='g', label='B')
ax.scatter(*C, color='b', label='C')
ax.scatter(*D, color='m', label='D')

# Show unit vectors i, j, k on the plane
i = np.array([1, 0, 0])
j = np.array([0, 1, 0])
k = np.array([0, 0, 1])

ax.quiver(O[0], O[1], O[2], i[0], i[1], i[2], color='r', label='i')
ax.quiver(O[0], O[1], O[2], j[0], j[1], j[2], color='g', label='j')
ax.quiver(O[0], O[1], O[2], k[0], k[1], k[2], color='b', label='k')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show plot
plt.show()

""" 

Show on 3D plane: Quadric surfaces

Ellipsoid
Hyperboloid of one sheet
Hyperboloid of two sheets
Elliptic cone
Elliptic paraboloid
Hyperbolic paraboloid

on separated diagrams, one by one 

"""

# Show an ellipsoid in 3D plot
phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
x = 2 * np.sin(theta) * np.cos(phi)
y = 1 * np.sin(theta) * np.sin(phi)
z = 3 * np.cos(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='c', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Ellipsoid')
plt.show()

# Show a hyperboloid of one sheet in 3D plot
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(-1, 1, 100)
u, v = np.meshgrid(u, v)
x = np.cosh(v) * np.cos(u)
y = np.cosh(v) * np.sin(u)
z = np.sinh(v)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='m', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Hyperboloid of One Sheet')
plt.show()

# Show a hyperboloid of two sheets in 3D plot
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(-1, 1, 100)
u, v = np.meshgrid(u, v)
x = np.sqrt(1 + v**2) * np.cos(u)
y = np.sqrt(1 + v**2) * np.sin(u)
z = v

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='y', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Hyperboloid of Two Sheets')
plt.show()

# Show an elliptic cone in 3D plot
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(-1, 1, 100)
u, v = np.meshgrid(u, v)
x = v * np.cos(u)
y = v * np.sin(u)
z = v

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='g', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Elliptic Cone')
plt.show()

# Show an elliptic paraboloid in 3D plot
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 1, 100)
u, v = np.meshgrid(u, v)
x = v * np.cos(u)
y = v * np.sin(u)
z = v**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Elliptic Paraboloid')
plt.show()

# Show a hyperbolic paraboloid in 3D plot
u = np.linspace(-1, 1, 100)
v = np.linspace(-1, 1, 100)
u, v = np.meshgrid(u, v)
x = u
y = v
z = u**2 - v**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='r', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Hyperbolic Paraboloid')
plt.show()

""" 
Define 2 vectors on 3D plane, A and B
Find the dot product, angle, and length of the vectors
"""

# Define vectors A and B
A = np.array([1, 0, 1])
B = np.array([0, 1, 1])

# Find the cross product of A and B
cross_product = np.cross(A, B)
print(f"Cross product of A and B: {cross_product}")

# Find the dot product
dot_product = np.dot(A, B)
print(f"Dot product of A and B: {dot_product}")

# Find the lengths of the vectors
length_A = np.linalg.norm(A)
length_B = np.linalg.norm(B)
print(f"Length of vector A: {length_A}")
print(f"Length of vector B: {length_B}")

# Find the angle between the vectors in radians
angle_radians = np.arccos(dot_product / (length_A * length_B))
print(f"Angle between A and B in radians: {angle_radians}")

# Convert the angle to degrees
angle_degrees = np.degrees(angle_radians)
print(f"Angle between A and B in degrees: {angle_degrees}")

# Show vectors A and B on 3D plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Origin
O = np.array([0, 0, 0])

# Plot vectors A and B
ax.quiver(O[0], O[1], O[2], A[0], A[1], A[2], color='r', label='A')
ax.quiver(O[0], O[1], O[2], B[0], B[1], B[2], color='g', label='B')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show plot
plt.show()

"""
Solve the system of three equations in the three unknowns

x + y - z = 4 
x - 2*y + 3*z = -6
2*x + 3*y + z = 7 

using matrix 

"""

# Define the coefficients matrix
A = np.array([[1, 1, -1],
              [1, -2, 3],
              [2, 3, 1]])

# Define the constants vector
b = np.array([4, -6, 7])

# Solve the system of equations
solution = np.linalg.solve(A, b)
print(f"Solution: x = {solution[0]}, y = {solution[1]}, z = {solution[2]}") # Should be x = 1, y = -2, z = -1

"""
Solve the system of three equations in the three unknowns

x + y - z = 4 
x - 2*y + 3*z = -6
2*x + 3*y + z = 7 

using sympy 
"""

# Define the symbols
x, y, z = sp.symbols('x y z')

# Define the equations
eq1 = sp.Eq(x + y - z, 4)
eq2 = sp.Eq(x - 2*y + 3*z, -6)
eq3 = sp.Eq(2*x + 3*y + z, 7)

# Solve the system of equations
solution = sp.solve((eq1, eq2, eq3), (x, y, z))
print(f"Solution: x = {solution[x]}, y = {solution[y]}, z = {solution[z]}") # Should be x = 1, y = -2, z = -1

"""
Given the point Original point A = (1, 1, 1)

Demonstrate 

i) Reflect A over plane xy 
ii) Reflect A over plane xy 
iii) Reflect A over plane xy 

On the same 3D diagram 

"""

# Define the original point A
A = np.array([1, 1, 1])

# Reflect A over plane xy (z = -z)
A_reflect_xy = np.array([A[0], A[1], -A[2]])

# Reflect A over plane yz (x = -x)
A_reflect_yz = np.array([-A[0], A[1], A[2]])

# Reflect A over plane zx (y = -y)
A_reflect_zx = np.array([A[0], -A[1], A[2]])

# Show the original and reflected points on the same 3D diagram
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original point A
ax.scatter(*A, color='r', label='A (1, 1, 1)')

# Plot the coordinate axes
ax.quiver(O[0], O[1], O[2], 1, 0, 0, color='k', linestyle='dotted', label='Ox')
ax.quiver(O[0], O[1], O[2], 0, 1, 0, color='k', linestyle='dotted', label='Oy')
ax.quiver(O[0], O[1], O[2], 0, 0, 1, color='k', linestyle='dotted', label='Oz')

# Connect the origin with the original point A and the reflected points using dotted lines
ax.plot([O[0], A[0]], [O[1], A[1]], [O[2], A[2]], 'r--')
ax.plot([O[0], A_reflect_xy[0]], [O[1], A_reflect_xy[1]], [O[2], A_reflect_xy[2]], 'g--')
ax.plot([O[0], A_reflect_yz[0]], [O[1], A_reflect_yz[1]], [O[2], A_reflect_yz[2]], 'b--')
ax.plot([O[0], A_reflect_zx[0]], [O[1], A_reflect_zx[1]], [O[2], A_reflect_zx[2]], 'm--')

# Draw planes xy, yz, and zx using faint colors 
ax.plot_surface(np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]]), np.array([[0, 0], [0, 0]]), color='r', alpha=0.1)
ax.plot_surface(np.array([[0, 0], [0, 0]]), np.array([[0, 1], [0, 1]]), np.array([[0, 0], [1, 1]]), color='g', alpha=0.1)
ax.plot_surface(np.array([[0, 0], [1, 1]]), np.array([[0, 0], [0, 0]]), np.array([[0, 1], [0, 1]]), color='b', alpha=0.1)
                
# Plot the reflected points
ax.scatter(*A_reflect_xy, color='g', label='Reflect over xy (1, 1, -1)')
ax.scatter(*A_reflect_yz, color='b', label='Reflect over yz (-1, 1, 1)')
ax.scatter(*A_reflect_zx, color='m', label='Reflect over zx (1, -1, 1)')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show plot
plt.show()

""" 

Given the point Original point A = (1, 1, 1)

Demonstrate 

i) Reflect A over line Ox
ii) Reflect A over line Oy
iii) Reflect A over line Oz
iv) Reflect A over the origin point O

On the same 3D diagram 

"""

# Define the original point A
A = np.array([1, 1, 1])

# Reflect A over line Ox (y = -y, z = -z)
A_reflect_Ox = np.array([A[0], -A[1], -A[2]])

# Reflect A over line Oy (x = -x, z = -z)
A_reflect_Oy = np.array([-A[0], A[1], -A[2]])

# Reflect A over line Oz (x = -x, y = -y)
A_reflect_Oz = np.array([-A[0], -A[1], A[2]])

# Reflect A over the origin point O (x = -x, y = -y, z = -z)
A_reflect_O = np.array([-A[0], -A[1], -A[2]])

# Show the original and reflected points on the same 3D diagram
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original point A
ax.scatter(*A, color='r', label='A (1, 1, 1)')

# Plot the coordinate axes
ax.quiver(O[0], O[1], O[2], 1, 0, 0, color='k', linestyle='dotted', label='Ox')
ax.quiver(O[0], O[1], O[2], 0, 1, 0, color='k', linestyle='dotted', label='Oy')
ax.quiver(O[0], O[1], O[2], 0, 0, 1, color='k', linestyle='dotted', label='Oz')

# Connect the origin with the original point A and the reflected points using dotted lines
ax.plot([O[0], A[0]], [O[1], A[1]], [O[2], A[2]], 'r--')
ax.plot([O[0], A_reflect_Ox[0]], [O[1], A_reflect_Ox[1]], [O[2], A_reflect_Ox[2]], 'g--')
ax.plot([O[0], A_reflect_Oy[0]], [O[1], A_reflect_Oy[1]], [O[2], A_reflect_Oy[2]], 'b--')
ax.plot([O[0], A_reflect_Oz[0]], [O[1], A_reflect_Oz[1]], [O[2], A_reflect_Oz[2]], 'm--')
ax.plot([O[0], A_reflect_O[0]], [O[1], A_reflect_O[1]], [O[2], A_reflect_O[2]], 'c--')

# Plot the reflected points
ax.scatter(*A_reflect_Ox, color='g', label='Reflect over Ox (1, -1, -1)')
ax.scatter(*A_reflect_Oy, color='b', label='Reflect over Oy (-1, 1, -1)')
ax.scatter(*A_reflect_Oz, color='m', label='Reflect over Oz (-1, -1, 1)')
ax.scatter(*A_reflect_O, color='c', label='Reflect over O (-1, -1, -1)')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show plot
plt.show()

""" 

Given the point Original point A = (1, 1, 1)
And an angle theta = 45 degree to rotate

Demonstrate 

i) Positive rotate A around the x-axis with angle theta 
ii) Positive rotate A around the y-axis with angle theta 
iii) Positive rotate A around the z-axis with angle theta 

Define the rotation matrix R_x(theta), R_y(theta), R_z(theta)

Demonstrate General 3D rotations 
R = R_z(alpha) * R_y(beta) * R_x(gamma)
for the point A 
where 
alpha = 45 degree
beta = 45 degree
gamma = 45 degree

On the same 3D diagram 

"""

# Define the original point A
A = np.array([1, 1, 1])

# Define the angle theta in radians
theta = np.radians(45)

# Define the rotation matrices
R_x = np.array([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])

R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])

R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

# Rotate A around the x-axis
A_rot_x = R_x @ A

# Rotate A around the y-axis
A_rot_y = R_y @ A

# Rotate A around the z-axis
A_rot_z = R_z @ A

# Define the general rotation angles
alpha = beta = gamma = np.radians(45)

# Define the general rotation matrix
R_general = R_z @ R_y @ R_x

# Rotate A using the general rotation matrix
A_rot_general = R_general @ A

# Show the original and rotated points on the same 3D diagram
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original point A
ax.scatter(*A, color='r', label='A (1, 1, 1)')

# Plot the coordinate axes
ax.quiver(O[0], O[1], O[2], 1, 0, 0, color='k', linestyle='dotted', label='Ox')
ax.quiver(O[0], O[1], O[2], 0, 1, 0, color='k', linestyle='dotted', label='Oy')
ax.quiver(O[0], O[1], O[2], 0, 0, 1, color='k', linestyle='dotted', label='Oz')

# Plot the rotated points
ax.scatter(*A_rot_x, color='g', label='Rotate around x-axis')
ax.scatter(*A_rot_y, color='b', label='Rotate around y-axis')
ax.scatter(*A_rot_z, color='m', label='Rotate around z-axis')
ax.scatter(*A_rot_general, color='c', label='General rotation')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show plot
plt.show()
