import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import chi2

def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    """
    Normalizes a value, which can be a scalar or a NumPy array, to a new range.
    The original and new ranges can also be scalars or NumPy arrays.
    """
    value = np.asanyarray(value)
    min_val = np.asanyarray(min_val)
    max_val = np.asanyarray(max_val)
    new_min = np.asanyarray(new_min)
    new_max = np.asanyarray(new_max)

    original_range = max_val - min_val

    original_range = np.where(original_range == 0, 1e-6, original_range)
    
    new_range = new_max - new_min

    result = (((value - min_val) * new_range) / original_range) + new_min
    
    if clip:
        return np.clip(result, new_min, new_max)
    else:
        return result


def get_distance_from_target(robot_node, target_node):
    distance_from_target = np.linalg.norm(robot_node[:, np.newaxis, :] - target_node, axis=2)
    return distance_from_target


def get_angle_from_target(robot_node, target_node, epuck_angle, is_abs=False):

    angle_between = np.arctan2(target_node[:, np.newaxis, 1] - robot_node[:, 1], target_node[:, np.newaxis, 0] - robot_node[:, 0])
    angle_diff = np.fmod(angle_between.T - epuck_angle,math.tau)

    return abs(angle_diff) if is_abs else angle_diff


def update_R(R_, angular_velocity, dt):
    # 将角速度向量转换为四元数形式
    omega = angular_velocity * dt
    incremental_rotation_matrix = R.from_rotvec(omega).as_matrix()
    return np.matmul(R_, incremental_rotation_matrix)

def quaternion_to_euler(rotation):
    angle = rotation[:, 3] / 2.0
    x = rotation[:, 0] * np.sin(angle)
    y = rotation[:, 1] * np.sin(angle)
    z = rotation[:, 2] * np.sin(angle)
    w = np.cos(angle)
    quaternion = np.array([w, x, y, z]).T
    return quaternion


def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def fun_resample(xarr, z, N, weight):
    # 协方差矩阵
    Covariance_matrix = np.cov(xarr[:2, :])

    eig_vals, V_ = np.linalg.eig(Covariance_matrix)
    largest_idx = np.argmax(eig_vals)
    largest_c = V_[:, largest_idx]
    largest_v = eig_vals[largest_idx]
    smallest_v = eig_vals[1 - largest_idx]

    # 椭圆倾斜角度
    th = np.arctan2(largest_c[1], largest_c[0])
    # import pdb;pdb.set_trace()
    # 置信度为 r1，r2, 自由度为 2 时椭圆的规模
    r1 = chi2.ppf(0.8, 1)
    r2 = chi2.ppf(0.125, 2)

    # 重采样
    Nl, Nd, Nh, sw, count = 0, 0, 0, 0, 0
    x = np.zeros((2, 2 * N))
    xweight = np.zeros(N)
    xweightrem = np.zeros(2 * N)
    xpart = np.zeros((2, N))
    # print(largest_v)
    # print(smallest_v)
    # import pdb;pdb.set_trace()

    for i in range(N):
        d = ((xarr[0, i] - z[0]) * np.cos(th) + (xarr[1, i] - z[1]) * np.sin(th)) ** 2 / largest_v + \
            ((xarr[1, i] - z[1]) * np.cos(th) - (xarr[0, i] - z[0]) * np.sin(th)) ** 2 / smallest_v
        if d > r1:
            Nl += 1
        elif r2 <= d <= r1:
            count += 1
            xpart[:, count - 1] = xarr[:, i]
            xweight[count - 1] = weight[i]
            sw += weight[i]
        else:
            Nh += 1
            x[:, Nh - 1] = xarr[:, i]
            xweightrem[Nh - 1] = weight[i]

    if Nl == N:
        xpart[0, :] = z[0] + np.random.normal(0.001, 0.0002, N)
        xpart[1, :] = z[1] + np.random.normal(0.001, 0.0002, N)
        xweight = np.ones(N)  # normrnd(10,0.5)
    else:
        qw = (1 - sw) / (Nl + Nh)
        Nt = Nl - (Nl // Nh) * Nh
        for i in range(Nh):
            if i < Nt:
                copytimes = (Nl // Nh) + 2
            else:
                copytimes = (Nl // Nh) + 1
            for j in range(copytimes):
                count += 1
                xpart[:, count - 1] = x[:, i]
                xweight[count - 1] = xweightrem[i]

    xweight = xweight / np.sum(xweight)
    xpart = xpart.T
    weight = xweight
    # print(Nl)
    return np.sum(xpart * weight[:, np.newaxis], axis=0)