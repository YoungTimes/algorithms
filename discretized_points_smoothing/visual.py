from weakref import ref
import matplotlib.pyplot as plt
import numpy as np

from cubic_spline import *

from shapely.geometry import Point, LineString

road_1_boundary = np.array([[-30.0, 9.0], [10.0, 9.0], [20.0, 10.0], [30.0, 10.0], [45.0, 30.0], [45.0, 70.0]])
road_2_boundary = np.array([[70.0, 70.0], [70.0, -50.0]])
road_3_boundary = np.array([[-30.0, -9.0], [10.0, -9.0], [20.0, -10.0], [30.0, -10.0], [45.0, -30.0], [45.0, -50.0]])


lane_1_center = np.array([[-30.0 + i * 10.0, 6.0] for i in range(0, 7)])
lane_2_center = np.array([[-30.0 + i * 10.0, 0.0] for i in range(0, 7)])
lane_3_center = np.array([[-30.0 + i * 10.0, -6.0] for i in range(0, 7)])


lane_4_center = np.array([[50.0, 30.0 + i * 10.0] for i in range(0, 5)]) 
lane_5_center = np.array([[56.0, 30.0 + i * 10.0] for i in range(0, 5)])
lane_6_center = np.array([[62.0, 30.0 + i * 10.0] for i in range(0, 5)])


ds = 2.0  # [m] distance of each interpolated points
sp = Spline2D(np.array([30.0, 32.0, 56.0, 56.0]), np.array([0.0, 0.0, 28.0, 30.0]))
s = np.arange(0, sp.s[-1], ds)
new_x = []
for i_s in s:
    rx, ry = sp.calc_position(i_s)
    rtheta = sp.calc_yaw(i_s)
    new_x.append(np.array([rx, ry]))
lane_7_center = np.array(new_x)

ref_line = lane_3_center
ref_line = np.append(ref_line, lane_7_center, axis=0)
ref_line = np.append(ref_line, lane_6_center, axis=0)

road_1_linestring = LineString(road_1_boundary.tolist())
road_2_linestring = LineString(road_2_boundary.tolist())
road_3_linestring = LineString(road_3_boundary.tolist())

l_distances = []
for i in range(len(ref_line)):
    l_distance = []
    left_dis = Point(ref_line[i]).distance(road_1_linestring)

    right_dis = Point(ref_line[i]).distance(road_2_linestring)
    dis = Point(ref_line[i]).distance(road_3_linestring)
    


plt.axis('equal')

plt.plot(lane_1_center[:, 0], lane_1_center[:, 1], ls="-.",color="r",marker =",", lw=2)
plt.plot(lane_2_center[:, 0], lane_2_center[:, 1], ls="-.",color="r",marker =",", lw=2)
plt.plot(lane_3_center[:, 0], lane_3_center[:, 1], ls="-.",color="r",marker =",", lw=2)

plt.plot(lane_4_center[:, 0], lane_4_center[:, 1], ls="-.",color="r",marker =",", lw=2)
plt.plot(lane_5_center[:, 0], lane_5_center[:, 1], ls="-.",color="r",marker =",", lw=2)
plt.plot(lane_6_center[:, 0], lane_6_center[:, 1], ls="-.",color="r",marker =",", lw=2)

plt.plot(lane_7_center[:, 0], lane_7_center[:, 1], ls="-.",color="r",marker =",", lw=2)

plt.plot(road_1_boundary[:, 0], road_1_boundary[:, 1], ls="-",color="black", lw = 4.0)
plt.plot(road_2_boundary[:, 0], road_2_boundary[:, 1], ls="-",color="black", lw = 4.0)
plt.plot(road_3_boundary[:, 0], road_3_boundary[:, 1], ls="-",color="black", lw = 4.0)

plt.plot(ref_line[:, 0], ref_line[:, 1], color="g", marker ="o", lw=2)

print(ref_line)

smoothed_ref_line = np.array([[-30, -6],[-19.0421, -5.46179],[-8.36261, -4.91931],[1.75987, -4.36829],[11.0469, -3.8043],[19.22, -3.22281],[26.0007, -2.61907],[31.1105, -1.98812],[34.9013, -1.32474],[37.7251, -0.623417],[39.9338, 0.121637],[41.8792, 0.91651],[43.664, 1.76759],[45.3001, 2.68153],[46.7995, 3.66526],[48.1743, 4.72594],[49.4362, 5.87093],[50.5968, 7.10776],[51.6672, 8.4441],[52.6584, 9.88771],[53.5805, 11.4464],[54.4432, 13.128],[55.2553, 14.9404],[56.0251, 16.8915],[56.7598, 18.989],[57.466, 21.3532],[58.1493, 24.3794],[58.8147, 28.4628],[59.4664, 33.9987],[60.108, 41.3823],[60.7424, 50.1516],[61.3723, 59.8448],[62, 70]])
plt.plot(smoothed_ref_line[:, 0], smoothed_ref_line[:, 1], color="c", marker ="o", lw=2)

plt.show()