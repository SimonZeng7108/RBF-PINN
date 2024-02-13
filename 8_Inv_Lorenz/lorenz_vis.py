import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#def lorenz(xyz, *, s=10, r=28, b=8/3):
def lorenz(x, y, z, a=10.0, r=15.0, b=8/3):
    
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """

    x_dot = a*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


def lorenz_pred(x, y, z, a=10.0, r=15.0, b=8/3):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """

    x_dot = a*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

alpha = 10
beta = 8/3
rho = 15

#rbf pol 20
test_alpha=  9.983504380278167
test_beta= 2.6577673151887657
test_rho=  15.00396374139246


#random sample
dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)
ts = np.empty(num_steps + 1)
# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)
ts[0] = 0
# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], a = alpha, r = rho, b = beta)
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)
    ts[i + 1] = ts[i] + dt

# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xs, ys, zs, '.', lw=2, markersize=0.2, c='blue', alpha=1, label="Ground Truth")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.dist = 7.5
ax.azim = -66   # z rotation (default=270)
ax.elev = 4    # x rotation (default=0)
plt.axis('scaled')
plt.autoscale(tight=True)
plt.tight_layout()
plt.legend(prop = { "size": 14 }, bbox_to_anchor=(0.8,1))


dt = 0.01
num_steps = 10000
test_xs = np.empty(num_steps + 1)
test_ys = np.empty(num_steps + 1)
test_zs = np.empty(num_steps + 1)
test_ts = np.empty(num_steps + 1)
# Set initial values
test_xs[0], test_ys[0], test_zs[0] = (0., 1., 1.05)
test_ts[0] = 0
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(test_xs[i], test_ys[i], test_zs[i], a = test_alpha, r = test_rho, b = test_beta)
    test_xs[i + 1] = test_xs[i] + (x_dot * dt)
    test_ys[i + 1] = test_ys[i] + (y_dot * dt)
    test_zs[i + 1] = test_zs[i] + (z_dot * dt)
    test_ts[i + 1] = test_ts[i] + dt

# ax = plt.figure().add_subplot(projection='3d')
ax.plot(test_xs, test_ys, test_zs,'.', lw=2, markersize=0.2, c='red', alpha=1, label="Prediction")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.dist = 7.5
ax.azim = -66   # z rotation (default=270)
ax.elev = 4    # x rotation (default=0)
plt.legend(prop = { "size": 14 }, bbox_to_anchor=(0.8,1))
plt.axis('scaled')
plt.autoscale(tight=True)
plt.tight_layout()
# plt.savefig('rbf_pol.png', dpi= 300, bbox_inches='tight', pad_inches=0.4)
plt.show()


plt.figure()
# ax = plt.figure().add_subplot(projection='3d')
print(test_xs.shape)
print((xs-test_xs).shape)
print(test_xs-xs)
ax.plot(test_xs-xs, test_ys-ys, test_zs-zs,'.', lw=2, markersize=0.2, c='red', alpha=1, label="Prediction")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
# ax.dist = 7.5
# ax.azim = -66   # z rotation (default=270)
# ax.elev = 4    # x rotation (default=0)
plt.legend(prop = { "size": 14 }, bbox_to_anchor=(0.8,1))
plt.axis('scaled')
plt.autoscale(tight=True)
plt.tight_layout()
# plt.savefig('rbf_pol.png', dpi= 300, bbox_inches='tight', pad_inches=0.4)
plt.show()
