import numpy as np
import scipy
import scipy.optimize as opt
from scipy import integrate 
import sys
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt

class Path:
    def __init__(self):
        self._s0 = 0.0

        self._integrate_steps = 8

    # 辛普森积分法
    def simpson_rule(self, f, a, b, n):  
        """  
        f: 要积分的函数  
        a: 积分下限  
        b: 积分上限  
        n: 迭代次数  
        """  
        h = (b - a) / n  # 计算步长  
        x = [a + i * h for i in range(n + 1)]  # 生成分割点  
        y = [f(xi) for xi in x]  # 计算每个分割点处的函数值  
    
        # 根据辛普森公式进行计算  
        I = h / 3 * (y[0] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]) + y[-1])  
    
        return I

    def curvaturef(self, a, b, c, d, s):
        curvatures = a + b * s + c * s**2 + d * s**3

        return curvatures

    def curvature(self, s, p):
        a0 = p[0]
        a1 = (-5.5 * p[0] + 9.0 * p[1] - 4.5 * p[2] + p[3]) / p[4]
        a2 = (9.0 * p[0] - 22.5 * p[1] + 18 * p[2] - 4.5 * p[3]) / (p[4]**2)
        a3 = (-4.5 * p[0] + 13.5 * p[1] - 13.5 * p[2] + 4.5 * p[3])  / (p[4]**3)

        value = a0 + a1 * s + a2 * s**2 + a3 * s**3
        
        return value

    def thetaf(self, a, b, c, d, s):
        # Remember that a, b, c, d and s are lists
        thetas = self._theta0 + a * s + (b / 2) * (s**2) + (c / 3) * (s**3) + (d / 4) * (s**4)
        
        return thetas

    def theta(self, s, p):
        a0 = p[0]
        a1 = (-5.5 * p[0] + 9 * p[1] - 4.5 * p[2] + p[3]) / p[4]
        a2 = (9 * p[0] - 22.5 * p[1] + 18 * p[2] - 4.5 * p[3]) / (p[4]**2)
        a3 = (-4.5 * p[0] + 13.5 * p[1] - 13.5 * p[2] + 4.5 * p[3])  / (p[4]**3)

        thetas = self._theta0 + a0 * s + (a1 / 2) * (s**2) + (a2 / 3) * (s**3) + (a3 / 4) * (s**4)
        
        return thetas

    def objective(self, p):
        """
        The optimizer can freely move 3 of the spiral parameter variables.
        The other two are fixed due to boundary conditions.
        """
        p = [0.0, p[0], p[1], 0.0, p[2]]
        
        return self.fbe(p)  +  25.0 * (self.fxf(p)**2.0 + self.fyf(p)**2.0) + 30.0 * self.ftf(p)**2.0

    def objective_grad(self, p):
        """
        The optimizer can freely move 3 of the spiral parameter variables.
        The other two are fixed due to boundary conditions.
        """
        p = [0.0, p[0], p[1], 0.0, p[2]]

        return np.add(np.add(np.add(self.fbe_grad(p), np.multiply(25, self.fxf_grad(p))), \
            np.multiply(25, self.fyf_grad(p))), np.multiply(30, self.ftf_grad(p)))


    def integral_x(self, s, p):
        return np.cos(self.theta(s, p))

    def fxf(self, p):
        # value = self._x0 + self.simpson_rule(self.integral_x, self._s0, p[4], 2)
        # value = (value - self._xf)**2

        num = self._integrate_steps

        delta = (p[4] - self._s0) / num

        value = 0.0
        for i in range(num + 1):
            s = self._s0 + i * delta
            tmp_value = self.integral_x(s, p)

            param = delta / 3.0 
            if (i > 0 and i != num and i % 2 == 0):
                param = param * 2

            elif (i > 0 and i != num and i % 2 == 1):
                param = param * 4

            value += tmp_value * param

        return (value + self._x0 - self._xf)


    def fxf_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        num = self._integrate_steps

        delta = (p[4] - self._s0) / num

        grad_2_1 = 0.0
        grad_2_2 = 0.0

        grad_2_param = 1.0


        for i in range(num + 1):
            s = self._s0 + i * delta

            param = delta / 3.0 
            if (i > 0 and i != num and i % 2 == 0):
                param = param * 2
                grad_2_param = 2.0

            elif (i > 0 and i != num and i % 2 == 1):
                param = param * 4
                grad_2_param = 4.0

            grad[0] += (-9.0 / p[4] * (s**2 / 2.0) \
                + 22.5 / (p[4]**2) * (s**3 / 3.0) \
                - 13.5 / (p[4]**3) * (s**4 / 4.0)) * np.sin(self.theta(s, p)) * self.fxf(p) * 2 * param

            grad[1] += (4.5 / p[4] * (s**2 / 2.0) \
                - 18.0 / (p[4]**2) * (s**3 / 3.0) \
                + 13.5 / p[4]**3 * (s**4 / 4.0)) * np.sin(self.theta(s, p)) * self.fxf(p) * 2 * param


            grad_2_1 += (-1.0 / (p[4]**2) * (5.5 * p[0] - 9.0 * p[1] + 4.5 * p[2] -  p[3]) * (s**2 / 2.0) \
                + (2.0 / (p[4]**3)) * (9 * p[0] - 22.5 * p[1] + 18.0 * p[2] - 4.5 * p[3]) * (s**3 / 3.0) \
                - (3.0 / (p[4]**4)) * (4.5 * p[0] - 13.5 * p[1] + 13.5 * p[2] - 4.5 * p[3]) * (s**4 / 4.0))  * np.sin(self.theta(s, p)) * param


            grad_2_2 += 1.0 / (3.0 * num) * np.cos(self.theta(s, p)) * grad_2_param

        grad[2] = (grad_2_1 + grad_2_2) * self.fxf(p) * 2 


        return grad


    def integral_y(self, s, p):
        return np.sin(self.theta(s, p))

    def fyf(self, p):

        num = self._integrate_steps

        delta = (p[4] - self._s0) / num

        value = 0.0
        for i in range(num + 1):
            s = self._s0 + i * delta
            tmp_value = self.integral_y(s, p)

            param = delta / 3.0 
            if (i > 0 and i != num and i % 2 == 0):
                param = param * 2

            elif (i > 0 and i != num and i % 2 == 1):
                param = param * 4

            value += tmp_value * param

        return (value + self._y0 - self._yf)


    def fyf_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        num = self._integrate_steps

        delta = (p[4] - self._s0) / num

        # tst = 0.0

        grad_2_1 = 0.0
        grad_2_2 = 0.0

        for i in range(num + 1):
            s =  self._s0 + i * delta

            # print("i="+str(i))

            param = delta / 3.0 
            grad_2_param = 1.0
            if (i > 0 and i != num and i % 2 == 0):
                param = param * 2
                grad_2_param = 2.0

            elif (i > 0 and i != num and i % 2 == 1):
                param = param * 4
                grad_2_param = 4.0

            grad[0] += (9.0 / p[4] * (s**2 / 2.0) \
                - 22.5 / (p[4]**2) * (s**3 / 3.0) \
                + 13.5 / (p[4]**3) * (s**4 / 4.0)) * np.cos(self.theta(s, p)) * 2 * self.fyf(p) * param

            grad[1] += (-4.5 / p[4] * (s**2 / 2.0) \
                + 18.0 / p[4]**2 * (s**3 / 3.0) \
                - 13.5 / p[4]**3 * (s**4 / 4.0)) * np.cos(self.theta(s, p)) * 2 * self.fyf(p) * param

            grad_2_1 += ((1.0 / p[4]**2.0) * (11.0 / 2.0 * p[0] - 18.0 / 2.0 * p[1] + 9.0 / 2.0 * p[2] - p[3]) * (s**2 / 2.0) \
                - (2.0 / p[4]**3.0) * (9.0 * p[0] - 22.5 * p[1] + 18.0 * p[2] - 4.5 * p[3]) * (s**3 / 3.0) \
                + (3.0 / p[4]**4.0) * (4.5 * p[0] - 13.5 * p[1] + 13.5 * p[2] - 4.5 * p[3]) * (s**4 / 4.0)) * np.cos(self.theta(s, p))  * param \

            grad_2_2 += 1.0 / (3.0 * num) * np.sin(self.theta(s, p)) * grad_2_param


        grad[2] = (grad_2_1 + grad_2_2) * 2 * self.fyf(p)


        return grad


    def ftf(self, p):

        value  =  self.theta(p[4], p)
        value = value - self._thetaf

        return value

    def ftf_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        s = p[4]

        grad[0] += 9.0 / p[4] * (s**2 /2.0) * 2 * self.ftf(p) \
                - 22.5 / (p[4]**2) * (s**3 / 3.0) * 2 * self.ftf(p) \
                + 13.5 / (p[4]**3) * (s**4 / 4.0) * 2 * self.ftf(p)

        grad[1] += -9.0 / (2 * p[4]) * (s**2 / 2.0) * 2 * self.ftf(p) \
                + 18.0 / (p[4]**2) * (s**3 / 3.0) * 2 * self.ftf(p) \
                - 27.0 / (2 * p[4]**3) * (s**4 / 4.0) * 2 * self.ftf(p)

        grad[2] += 1.0 / (p[4]**2) * (5.5 * p[0] - 9.0 * p[1] + 4.5 * p[2] + p[3]) * (s**2 / 2.0) * 2 * self.ftf(p) \
                - 2.0 / (p[4]**3) * (9 * p[0] - 22.5 * p[1] + 18 * p[2] - 4.5 * p[3]) * (s**3 / 3.0) *  2 * self.ftf(p) \
                + 3.0 / (p[4]**4) * (4.5 * p[0] - 13.5 * p[1] + 13.5 * p[2] - 4.5 * p[3]) * (s**4 / 4.0) *  2 * self.ftf(p)

        return grad
        
    def int_fbe(self, s, p):
        a0 = p[0]
        a1 = (-5.5 * p[0] + 9 * p[1] - 4.5 * p[2] + p[3]) / p[4]
        a2 = (9 * p[0] - 22.5 * p[1] + 18 * p[2] - 4.5 * p[3]) / (p[4]**2)
        a3 = (-4.5 * p[0] + 13.5 * p[1] - 13.5 * p[2] + 4.5 * p[3])  / (p[4]**3)

        # fbe = \int (a0 + a1 * s + a2 * s**2 + a3 * s**3)
        return a0 + a1 * s + a2 * s**2 + a3 * s**3


    def fbe(self, p):

        num = self._integrate_steps

        delta = (p[4] - self._s0) / num

        value = 0.0
        for i in range(num + 1):
            s = self._s0 + i * delta
            tmp_value = self.int_fbe(s, p)

            param = delta / 3.0 
            if (i > 0 and i != num and i % 2 == 0):
                param = param * 2

            elif (i > 0 and i != num and i % 2 == 1):
                param = param * 4

            value += tmp_value**2.0 * param


        return value

    def fbe_grad(self, p):
        grad = [0.0, 0.0, 0.0]

        num = self._integrate_steps

        delta = (p[4] - self._s0) / num

        for i in range(num + 1):
            s = self._s0 + i * delta

            param = delta / 3.0 
            if (i > 0 and i != num and i % 2 == 0):
                param = param * 2

            elif (i > 0 and i != num and i % 2 == 1):
                param = param * 4

            # print(self.curvature(s, p))

            grad[0] += (9.0 / p[4] * s \
                - 22.5 / (p[4]**2) * s**2 \
                + 13.5 / (p[4]**3) * s**3) * self.curvature(s, p) * 2.0 * param

            grad[1] += (-9.0 / (2 * p[4]) * s \
                + 18.0 / (p[4]**2) * (s**2) \
                - 27.0 / (2 * p[4]**3) * (s**3)) * self.curvature(s, p) * 2.0 * param

            grad[2] += (1.0 / (p[4]**2) * (5.5 * p[0] -9.0 * p[1] + 4.5 * p[2] - p[3]) * s \
                - 2.0 / (p[4]**3) * (9 * p[0] - 22.5 * p[1] + 18 * p[2] - 4.5 * p[3]) * (s**2) \
                + 3.0 / (p[4]**4) * (4.5 * p[0] - 13.5 * p[1] + 13.5 * p[2] - 4.5 * p[3]) * (s**3)) * self.curvature(s, p) * 2.0 * param


        return grad


    # Sets up the optimization problem to compute a spiral to a given
    # goal point, (xf, yf, tf).
    def optimize_spiral(self, x0, y0, theta0, xf, yf, thetaf):
        self._x0 = x0
        self._y0 = y0
        self._theta0 = theta0 

        # Save the terminal x, y, and theta.
        self._xf = xf
        self._yf = yf
        self._thetaf = thetaf
        # The straight line distance serves as a lower bound on any path's
        # arc length to the goal.
        sf_0 = np.linalg.norm([xf - x0, yf - y0])
        max = sys.maxsize
        # The initial variables correspond to a straight line with arc length
        # sf_0.  Recall that p here is defined as:
        #    [p1, p2, sf]
        #, where p1 and p2 are the curvatures at points p1 and p2
        #, and sf is the final arc length for the spiral.
        # Since we already set p0 and p4 (being the curvature of
        # the initial and final points) to be zero.
        p0 = [0.0, 0.0, sf_0]
        # Here we will set the bounds [lower, upper] for each optimization 
        # variable.
        # The first two variables correspond to the curvature 1/3rd of the
        # way along the path and 2/3rds of the way along the path, respectively.
        # As a result, their curvature needs to lie within [-0.5, 0.5].
        # The third variable is the arc length, it has no upper limit, and it
        # has a lower limit of the straight line arc length.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        bounds = [[-0.5, 0.5], [-0.5, 0.5], [sf_0, max]]
        # ------------------------------------------------------------------

        # Here we will call scipy.optimize.minimize to optimize our spiral.
        # The objective and gradient are given to you by self.objective, and
        # self.objective_grad. The bounds are computed above, and the inital
        # variables for the optimizer are set by p0. You should use the L-BFGS-B
        # optimization methods.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        res = opt.minimize(fun = self.objective, x0 = p0, method = 'L-BFGS-B', jac = self.objective_grad, bounds= bounds)
        # ------------------------------------------------------------------

        spiral = self.sample_spiral(res.x)

        return spiral

    # This function samples the spiral along its arc length to generate
    # a discrete set of x, y, and theta points for a path.
    def sample_spiral(self, p):
        """Samples a set of points along the spiral given the optimization
        parameters.

        args:
            p: The resulting optimization parameters that minimizes the
                objective function given a goal state.
                Format: [p1, p2, sf], Unit: [1/m, 1/m, m]
                , where p1 and p2 are the curvatures at points p1 and p2
                  and sf is the final arc length for the spiral.
        returns:
            [x_points, y_points, t_points]:
                x_points: List of x values (m) along the spiral
                y_points: List of y values (m) along the spiral
                t_points: List of yaw values (rad) along the spiral
        """
        # These equations map from the optimization parameter space
        # to the spiral parameter space.   
        p = [0.0, p[0], p[1], 0.0, p[2]]    # recall p0 and p3 are set to 0
                                            # and p4 is the final arc length
        # print("p:" + str(p))

        a = p[0]
        b = -(11.0 * p[0] / 2.0 - 9.0 * p[1] + 9.0 * p[2] / 2.0 - p[3]) / p[4]
        c = (9.0 * p[0] - 45.0 * p[1] / 2.0 + 18.0 * p[2] - 9.0 * p[3] / 2.0) / p[4]**2
        d = -(9.0 * p[0] / 2.0 - 27.0 * p[1] / 2.0 + 27.0 * p[2] / 2.0 - 9.0 * p[3] / 2.0) / p[4]**3

        # print([a, b, c, d])

        # Set the s_points (list of s values along the spiral) to be from 0.0
        # to p[4] (final arc length)
        s_points = np.linspace(0.0, p[4])

        # Compute the theta, x, and y points from the uniformly sampled
        # arc length points s_points (p[4] is the spiral arc length).
        # Use self.thetaf() to compute the theta values from the s values.
        # Recall that x = integral cos(theta(s)) ds and
        #             y = integral sin(theta(s)) ds.
        # You will find the scipy.integrate.cumtrapz() function useful.
        # Try to vectorize the code using numpy functions for speed if you can.

        # Try to vectorize the code using numpy functions for speed if you can.
        # TODO: INSERT YOUR CODE BETWEEN THE DASHED LINES
        # ------------------------------------------------------------------
        t_points = self.thetaf(a, b, c, d, s_points)

        x_points = integrate.cumtrapz(np.cos(t_points), s_points, initial=None)
        y_points = integrate.cumtrapz(np.sin(t_points), s_points, initial=None)

        x_points = x_points + self._x0
        y_points = y_points + self._y0
        t_points = t_points + self._theta0

        return [x_points, y_points, t_points]


def viz():
    road_1_boundary = np.array([[-30.0, 9.0], [10.0, 9.0], [20.0, 10.0], [30.0, 10.0], [45.0, 30.0], [45.0, 70.0]])
    road_2_boundary = np.array([[70.0, 70.0], [70.0, -50.0]])
    road_3_boundary = np.array([[-30.0, -9.0], [10.0, -9.0], [20.0, -10.0], [30.0, -10.0], [45.0, -30.0], [45.0, -50.0]])


    lane_1_center = np.array([[-30.0 + i * 10.0, 6.0] for i in range(0, 7)])
    lane_2_center = np.array([[-30.0 + i * 10.0, 0.0] for i in range(0, 7)])
    lane_3_center = np.array([[-30.0 + i * 10.0, -6.0] for i in range(0, 7)])


    lane_4_center = np.array([[50.0, 30.0 + i * 10.0] for i in range(0, 5)]) 
    lane_5_center = np.array([[56.0, 30.0 + i * 10.0] for i in range(0, 5)])
    lane_6_center = np.array([[62.0, 30.0 + i * 10.0] for i in range(0, 5)])

    plt.axis('equal')

    plt.plot(lane_1_center[:, 0], lane_1_center[:, 1], ls="-.",color="r",marker =",", lw=2)
    plt.plot(lane_2_center[:, 0], lane_2_center[:, 1], ls="-.",color="r",marker =",", lw=2)
    plt.plot(lane_3_center[:, 0], lane_3_center[:, 1], ls="-.",color="r",marker =",", lw=2)

    plt.plot(lane_4_center[:, 0], lane_4_center[:, 1], ls="-.",color="r",marker =",", lw=2)
    plt.plot(lane_5_center[:, 0], lane_5_center[:, 1], ls="-.",color="r",marker =",", lw=2)
    plt.plot(lane_6_center[:, 0], lane_6_center[:, 1], ls="-.",color="r",marker =",", lw=2)

    # plt.plot(lane_7_center[:, 0], lane_7_center[:, 1], ls="-.",color="r",marker =",", lw=2)

    plt.plot(road_1_boundary[:, 0], road_1_boundary[:, 1], ls="-",color="black", lw = 4.0)
    plt.plot(road_2_boundary[:, 0], road_2_boundary[:, 1], ls="-",color="black", lw = 4.0)
    plt.plot(road_3_boundary[:, 0], road_3_boundary[:, 1], ls="-",color="black", lw = 4.0)

    # plt.plot(ref_line[:, 0], ref_line[:, 1], color="g", marker ="o", lw=2)

    end_states = [(30.0, 6.0, 0.0), (30.0, 4.0, 0.0), (30.0, 2.0, 0.0), (30.0, 0.0, 0.0), (30.0, -2.0, 0.0), (30.0, -4.0, 0.0), (30.0, -6.0, 0.0)]
    # end_states = [(100.0, 100.0, 0.0)]

    sprial_path = Path()

    for end_state in end_states:
        sample_pts = sprial_path.optimize_spiral(-10.0, 0.0, 0.0, end_state[0], end_state[1], end_state[2])

        plt.plot(sample_pts[0], sample_pts[1], color="c", marker ="o", lw=2)

    plt.show()


# sprial_path = Path()
# sample_pts = sprial_path.optimize_spiral(0.0, 0.0, 0.0, 500.0, 5.0, 1.50)

# print(sample_pts[0])
# print(sample_pts[1])
# print(sample_pts[2])

# plt.plot(sample_pts[0], sample_pts[1])  
  
# # 显示图像  
# plt.show()

viz()