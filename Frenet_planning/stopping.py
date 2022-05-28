from ast import Mod
from enum import Enum
from turtle import color
import numpy as np
import sys
import random

sys.path.append(".")

from quintic_poly import QuinticPolynomial
from quartic_poly import QuarticPolynomial
from cubic_spline import *
import copy
from coordinate import frenet_to_cartesian3D, frenet_to_cartesian1D

import matplotlib.pyplot as plt
import matplotlib.patches as mp


SIM_LOOP = 500

# Parameter
MAX_SPEED = 50 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]

MIN_ROAD_WIDTH = -1.7
MAX_ROAD_WIDTH = 8.5  # maximum road width [m]

D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 10 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# Parameters for constant distance and constant time law.
D0 = 1 # static distance required between cars at speed 0
tau = 1 # time distance between cars

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.

show_animation = True

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.a = []
        self.c = []

        self.mode = None

def calc_following_path(c_d, c_d_d, c_d_dd, s0, s0_d, s0_dd, s1, s1_d, s1_dd):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(MIN_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(2, 5, 0.2):
            fp = FrenetPath()

            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]


            for delta_s in [s1 - s0]:

                s_target_dd =  s1_dd
                s_target_d =  s1_d
                s_target = s0 + delta_s

                tfp = copy.deepcopy(fp)
                lon_qp = QuinticPolynomial(s0, s0_d, s0_dd, s_target, s_target_d, s_target_dd, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk
                ds = (tfp.s[-1] - s0) ** 2


                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(MIN_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, csp):
    for fp in fplist:
        # calc global positions
        for i in range(len(fp.s)):
            rx, ry = csp.calc_position(fp.s[i])
            if rx is None:
                break

            rtheta = csp.calc_yaw(fp.s[i])
            rkappa = csp.calc_curvature(fp.s[i])
            rdkappa = csp.calc_d_curvature(fp.s[i])

            s_condition = np.array([fp.s[i], fp.s_d[i], fp.s_dd[i]])
            d_condition = np.array([fp.d[i], fp.d_d[i], fp.d_dd[i]])

            x, y, v, a, theta, kappa = frenet_to_cartesian3D(fp.s[i], rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition)

            fp.x.append(x)
            fp.y.append(y)
            fp.v.append(v)
            fp.a.append(a)
            fp.yaw.append(theta)

            fp.c.append(kappa)

    return fplist


def check_collision(fp, ob):
    if len(ob) == 0:
        return True

    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True

def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                    fplist[i].c]):  # Max curvature check
           continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]

def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path, fplist

def frenet_following_optimal_planning(csp, s0, s0_d, s0_dd, c_d, c_d_d, c_d_dd, s1, s1_d, s1_dd, ob):
    fplist = calc_following_path(c_d, c_d_d, c_d_dd, s0, s0_d, s0_dd, s1, s1_d, s1_dd)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path, fplist


def generate_target_course(x, y):
    csp = Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def create_lane_border(ref_line, width):
    tx, ty, tyaw, tc, csp = generate_target_course(ref_line[:, 0], ref_line[:, 1])

    border = []

    s = np.arange(0, csp.s[-1], 0.1)
    # print("s:" + str(s[300]))
    for i_s in range(len(s)):
        s_condition = [s[i_s]]
        d_condition = [width]
        lx, ly = frenet_to_cartesian1D(s[i_s], tx[i_s], ty[i_s], tyaw[i_s], s_condition, d_condition)
        border.append([lx, ly])

    return np.array(border)

from enum import Enum

class Mode(Enum):
    VELOCITY_KEEPING = 1
    STOPPING = 2

def decision(s):
    if s < 20.0:
        return Mode.VELOCITY_KEEPING

    return Mode.STOPPING


def main():
    print(__file__ + " start!!")

    center_lines = []
    borders = []

    center_line = np.array([[0.0, 1.0], [10.0, 0.0], [20.5, 5.0], [35.0, 6.5], [70.5, 0.0]])

    tx, ty, _, _, csp = generate_target_course(center_line[:, 0], center_line[:, 1])
    border_l = [-1.7, 1.7, 5.1, 8.5]
    center_l = [3.4, 6.8]
    for i in range(len(border_l)):
        border = create_lane_border(center_line, border_l[i])
        borders.append(border)
    for i in range(len(center_l)):
        center = create_lane_border(center_line, center_l[i])
        center_lines.append(center)

    stop_line = np.array([[28.70262896, 4.95465598], [28.25703523, 15.1449183]])
    stop_line_s = 30.0

    ob = np.array([])


    # initial state
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]

    s0 = 0.0  # current course position
    s0_d = 10.0 / 3.6  # current speed [m/s]
    c_speed = s0_d
    s0_dd = 0.0

    s1 = stop_line_s
    s1_d = 0.0
    s1_dd = 0.0


    area = 20.0  # animation area length [m]

    for i in range(SIM_LOOP):
        if decision(s0) == Mode.STOPPING:
            path, all_paths = frenet_following_optimal_planning(
                csp, s0, s0_d, s0_dd, c_d, c_d_d, c_d_dd, s1, s1_d, s1_dd, ob)
        else:
            path, all_paths = frenet_optimal_planning(
                csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        s0_d = path.s_d[1]
        s0_dd = path.s_dd[1]

        c_speed = path.s_d[1]

        if abs(s0 - stop_line_s) < 0.2:
            print("stop")
            break

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.gca().set_aspect(1.0)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(tx, ty, linestyle = '--')
            for border in borders:
                plt.plot(border[:, 0],  border[:, 1], color = '#333333')

            plt.plot(stop_line[:, 0], stop_line[:, 1], linestyle = '-', color = '#333333')

            for center in center_lines:
                plt.plot(center[:, 0],  center[:, 1], linestyle = '--', color = '#777777')

            if len(ob) > 0:
                plt.plot(ob[:, 0], ob[:, 1], "xk")
            for obi in range(len(ob)):
                obstacle = plt.Circle((ob[obi, 0], ob[obi, 1]), 2.0, color = '#444444')
                plt.gca().add_patch(obstacle)

            for all_path in all_paths:
                plt.plot(all_path.x[1:], all_path.y[1:], linestyle = '-', marker = 'o', markersize = 3.0, color = '#228800')

            plt.plot(path.x[1:], path.y[1:], linestyle = '', marker = 'o', markersize = 3.0, color = '#FF3300')
            plt.plot(path.x[1], path.y[1], linestyle = '', marker = 'o', markersize = 5.0, color = '#00EE00')

            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
