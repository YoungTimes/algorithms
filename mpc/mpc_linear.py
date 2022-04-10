from operator import ne
from re import U
from xxlimited import Null
import numpy as np
import math
from scipy import sparse
from scipy.spatial import KDTree
import osqp
import matplotlib.pyplot as plt
import time

import sys

v_d = 10.0
dt = 0.1
sim_steps = 200

def load_ref_traj():
    ref_traj = np.zeros((sim_steps, 5))

    for i in range(sim_steps):
        ref_traj[i, 0] = 5 * i * dt
        ref_traj[i, 1] = 5.0
        ref_traj[i, 2] = 0.0
        ref_traj[i, 3] = v_d
        ref_traj[i, 4] = 0.0

    return ref_traj

class UGV_model:
    def __init__(self, x0, y0, theta0, L, T): # L:wheel base
        self.x = x0 # X
        self.y = y0 # Y
        self.theta = theta0 # headding
        self.l = L  # wheel base
        self.dt = T  # decision time periodic

    def update(self, vt, deltat):  # update ugv's state
        self.v = vt

        dx = self.v * np.cos(self.theta)
        dy = self.v * np.sin(self.theta)
        dtheta = self.v * np.tan(deltat) / self.l

        self.x += dx * self.dt
        self.y += dy * self.dt
        self.theta += dtheta * self.dt
        
    def plot_duration(self):
        plt.scatter(self.x, self.y, color='r')   
        plt.axis([-5, 100, -6, 6])

        plt.pause(0.008)


class MPCController():
    def __init__(self, L, dt):
        self.L = L

        self.Nx = 3
        self.Nu = 2

        self.Nc = 30
        self.Np = 60

        self.T = dt

    def Solve(self, x, u_pre, ref_traj):

        tree = KDTree(ref_traj[:, :2])
        nearest_ref_info = tree.query(x[:2])
        nearest_ref_x = ref_traj[nearest_ref_info[1]]

        a = np.array([
            [1.0,  0,   -nearest_ref_x[3] * math.sin(nearest_ref_x[2]) * self.T],
            [0,   1.0,   nearest_ref_x[3] * math.cos(nearest_ref_x[2]) * self.T],
            [0,   0,     1]
        ])

        b = np.array([
            [math.cos(nearest_ref_x[2]) * self.T,             0],
            [math.sin(nearest_ref_x[2]) * self.T,             0],
            [math.tan(nearest_ref_x[4]) * self.T / self.L,     nearest_ref_x[3] * self.T / (self.L * math.pow(math.cos(nearest_ref_x[4]), 2.0))]
        ])

        A = np.zeros([self.Nx + self.Nu, self.Nx + self.Nu])
        A[0 : self.Nx, 0 : self.Nx] = a
        A[0 : self.Nx, self.Nx : ] =  b
        A[self.Nx :, self.Nx :] = np.eye(self.Nu)

        B = np.zeros([self.Nx + self.Nu, self.Nu])
        B[0 : self.Nx, :] = b
        B[self.Nx :, : ] = np.eye(self.Nu)

        C = np.array([[1, 0, 0, 0, 0], [0, 1, 0 ,0 ,0], [0, 0, 1, 0, 0]])

        theta = np.zeros([self.Np * self.Nx, self.Nc * self.Nu])
        phi = np.zeros([self.Np * self.Nx, self.Nu + self.Nx])
        tmp = C

        for i in range(1, self.Np + 1):
            phi[self.Nx * (i - 1) : self.Nx * i] = np.dot(tmp, A)

            tmp_c = np.zeros([self.Nx, self.Nc * self.Nu])
            tmp_c[ :, 0 : self.Nu] = np.dot(tmp, B)

            if i > 1:
                tmp_c[ :, self.Nu :] = theta[self.Nx * (i - 2) : self.Nx * (i - 1), 0 : -self.Nu]

            theta[self.Nx * (i - 1) : self.Nx * i, :] = tmp_c

            tmp = np.dot(tmp, A)


        Q = np.eye(self.Nx * self.Np)

        R = 5.0 * np.eye(self.Nu * self.Nc)

        rho = 10

        H = np.zeros((self.Nu * self.Nc + 1, self.Nu * self.Nc + 1))
        H[0 : self.Nu * self.Nc, 0 : self.Nu * self.Nc] = np.dot(np.dot(theta.transpose(), Q), theta) + R
        H[-1 : -1] = rho

        kesi = np.zeros((self.Nx + self.Nu, 1))
        diff_x = x - nearest_ref_x[:3]
        diff_x = diff_x.reshape(-1, 1)
        kesi[: self.Nx, :] = diff_x
        diff_u = u_pre.reshape(-1, 1)
        kesi[self.Nx :, :] = diff_u

        F = np.zeros((1, self.Nu * self.Nc + 1))
        F_1 = 2 * np.dot(np.dot(np.dot(phi, kesi).transpose(), Q), theta)
        F[ 0,  0 : self.Nu * self.Nc] = F_1

        # constraints
        umin = np.array([[-0.2], [-0.54]])
        umax = np.array([[0.2], [0.332]])

        delta_umin = np.array([[-0.05], [-0.0082]])
        delta_umax = np.array([[0.05], [0.0082]])

        A_t = np.zeros((self.Nc, self.Nc))
        for row in range(self.Nc):
            for col in range(self.Nc):
                if row >= col:
                    A_t[row, col] = 1.0


        A_I = np.kron(A_t, np.eye(self.Nu))

        A_cons = np.zeros((self.Nc * self.Nu, self.Nc * self.Nu + 1))
        A_cons[0 : self.Nc * self.Nu, 0 : self.Nc * self.Nu] = A_I

        U_t = np.kron(np.ones((self.Nc, 1)), u_pre.reshape(-1, 1))

        U_min = np.kron(np.ones((self.Nc, 1)), umin)
        U_max = np.kron(np.ones((self.Nc, 1)), umax)

        LB = U_min - U_t
        UB = U_max - U_t

        delta_Umin = np.kron(np.ones((self.Nc, 1)), delta_umin)
        delta_Umax = np.kron(np.ones((self.Nc, 1)), delta_umax)

        delta_Umin = np.vstack((delta_Umin, [0]))
        delta_Umax = np.vstack((delta_Umax, [10]))

        A_1_cons = np.eye(self.Nc * self.Nu + 1, self.Nc * self.Nu + 1)

        A_cons = np.vstack((A_cons, A_1_cons))

        LB = np.vstack((LB, delta_Umin))
        UB = np.vstack((UB, delta_Umax))

        # Create an OSQP object
        prob = osqp.OSQP()
    
        H = sparse.csc_matrix(H)
        A_cons = sparse.csc_matrix(A_cons)

        # Setup workspace
        prob.setup(H, F.transpose(), A_cons, LB, UB)

        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        u_cur = u_pre + res.x[0 : self.Nu]

        return u_cur, res.x[0 : self.Nu]


#u_0 = [3.0, 0.0]
x = np.array([0.0, 0.0, 0.0])
pre_u = np.array([0.0, 0.0])
L = 2.6

ref_traj = load_ref_traj()

plt.figure(figsize=(18, 3))
plt.plot(ref_traj[:,0], ref_traj[:,1], '-.b', linewidth=5.0)

history_us = np.array([])
history_delta_us = np.array([])

ugv = UGV_model(x[0], x[1], x[2], L, dt)
controller = MPCController(L, dt)
for k in range(sim_steps):

    if k == 1:
        time.sleep(20)

    u_cur, delta_u_cur = controller.Solve(x, pre_u, ref_traj)
    abs_u = [v_d, 0.0] + u_cur

    print(abs_u)

    ugv.update(abs_u[0], abs_u[1])
    ugv.plot_duration()

    # history_us = np.append(history_us, abs_u)
    if len(history_delta_us) == 0:
        history_delta_us = np.array([u_cur])
    else:
        history_delta_us = np.vstack((history_delta_us, u_cur))

    x = x + np.array([abs_u[0] * np.math.cos(x[2]) * dt,
                    abs_u[0] * np.math.sin(x[2]) * dt,
                    abs_u[0] * np.math.tan(abs_u[1]) / L * dt])

    pre_u = u_cur

plt.show()


# plt.figure(num = 1, figsize = (3,5))
# plt.plot(np.linspace(0, v_d * sim_steps * dt, 50), [0.2] * 50, color='green', linewidth=3.0, linestyle='--')
# plt.plot(np.linspace(0, v_d * sim_steps * dt, 50), [-0.2] * 50, color='green', linewidth=3.0, linestyle='--')
# plt.plot(np.linspace(0, v_d * sim_steps * dt, len(history_delta_us)), history_delta_us[:, 0], color='orange', linewidth=3.0, linestyle='--')
# plt.show()
