import numpy as np
import open3d as o3d
import math
import torch
from scipy.spatial.transform import Rotation as R

def read_xyz_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                coordinates = line.split()[:3]
                points.append([float(coord) for coord in coordinates])
    return np.array(points)

def rgbd_to_pcd_Art(color,depth):
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=640,
                                                      fx=914, fy=914,
                                                      cx=320, cy=320)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color,
                                                              depth=depth,
                                                              depth_trunc=7.0,
                                                              convert_rgb_to_intensity=False
                                                              )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd,
                                                         intrinsic=o3d_intrinsic,
                                                         )

    points=np.array(pcd.points)
    colors = np.array(pcd.colors)
    return points, colors

def rgbd_to_pcd_ReArt(color,depth):
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=640,
                                                      fx=914, fy=914,
                                                      cx=320, cy=320)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color,
                                                              depth=depth,
                                                              depth_trunc=7.0,
                                                              convert_rgb_to_intensity=False
                                                              )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd,
                                                         intrinsic=o3d_intrinsic,
                                                         )

    points=np.array(pcd.points)
    colors = np.array(pcd.colors)
    return points, colors

def RotateAnyAxis(v1, v2, step): #(xyz, xyz+rpy, state)
    ROT = np.identity(4)

    axis = v2 - v1
    axis = axis / math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)

    step_cos = math.cos(step)
    step_sin = math.sin(step)

    ROT[0][0] = axis[0] * axis[0] + (axis[1] * axis[1] + axis[2] * axis[2]) * step_cos
    ROT[0][1] = axis[0] * axis[1] * (1 - step_cos) + axis[2] * step_sin
    ROT[0][2] = axis[0] * axis[2] * (1 - step_cos) - axis[1] * step_sin
    ROT[0][3] = 0

    ROT[1][0] = axis[1] * axis[0] * (1 - step_cos) - axis[2] * step_sin
    ROT[1][1] = axis[1] * axis[1] + (axis[0] * axis[0] + axis[2] * axis[2]) * step_cos
    ROT[1][2] = axis[1] * axis[2] * (1 - step_cos) + axis[0] * step_sin
    ROT[1][3] = 0

    ROT[2][0] = axis[2] * axis[0] * (1 - step_cos) + axis[1] * step_sin
    ROT[2][1] = axis[2] * axis[1] * (1 - step_cos) - axis[0] * step_sin
    ROT[2][2] = axis[2] * axis[2] + (axis[0] * axis[0] + axis[1] * axis[1]) * step_cos
    ROT[2][3] = 0

    ROT[3][0] = (v1[0] * (axis[1] * axis[1] + axis[2] * axis[2]) - axis[0] * (v1[1] * axis[1] + v1[2] * axis[2])) * (1 - step_cos) + \
                (v1[1] * axis[2] - v1[2] * axis[1]) * step_sin

    ROT[3][1] = (v1[1] * (axis[0] * axis[0] + axis[2] * axis[2]) - axis[1] * (v1[0] * axis[0] + v1[2] * axis[2])) * (1 - step_cos) + \
                (v1[2] * axis[0] - v1[0] * axis[2]) * step_sin

    ROT[3][2] = (v1[2] * (axis[0] * axis[0] + axis[1] * axis[1]) - axis[2] * (v1[0] * axis[0] + v1[1] * axis[1])) * (1 - step_cos) + \
                (v1[0] * axis[1] - v1[1] * axis[0]) * step_sin
    ROT[3][3] = 1

    return ROT.T

def swap(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]

def rotate_pts(source, target):
    source = source - torch.mean(source, dim=0, keepdim=True)
    target = target - torch.mean(target, dim=0, keepdim=True)
    M = torch.matmul(target.t(), source)
    U, D, Vh = torch.svd(M)
    d = (torch.det(U) * torch.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    R = torch.matmul(U, Vh)
    return R


def scale_pts(source, target):
    pdist_s = source.unsqueeze(1) - source.unsqueeze(0)
    A = torch.sqrt(torch.sum(pdist_s ** 2, dim=2)).view(-1)
    pdist_t = target.unsqueeze(1) - target.unsqueeze(0)
    b = torch.sqrt(torch.sum(pdist_t ** 2, dim=2)).view(-1)
    scale = torch.dot(A, b) / (torch.dot(A, A) + 1e-6)
    return scale

def transform_pts(source, target):
    # source: [N x 3], target: [N x 3]
    source_centered = source - torch.mean(source, dim=0, keepdim=True)
    target_centered = target - torch.mean(target, dim=0, keepdim=True)
    rotation = rotate_pts(source_centered, target_centered)

    scale = scale_pts(source_centered, target_centered)

    translation = torch.mean(target.t() - scale * torch.matmul(rotation, source.t()), 1)
    return rotation, scale, translation

def check_nocs(nocs1, nocs2, points, threshold=10):
    assert nocs1.shape == nocs2.shape, "Arrays must have the same shape"

    r1, _, translation1 = transform_pts(points, nocs1)
    r2, _, translation2 = transform_pts(points, nocs2)

    results = []
    errors = []
    translations = []
    trace_value1 = r1.trace()
    trace_value2 = r2.trace()
    angle1 = torch.arccos(torch.clamp((trace_value1 - 1) / 2, -1.0 + 1e-6, 1.0 - 1e-6))
    angle2 = torch.arccos(torch.clamp((trace_value2 - 1) / 2, -1.0 + 1e-6, 1.0 - 1e-6))

    error = torch.abs(torch.rad2deg(angle1) -  torch.rad2deg(angle2))
    error = torch.clamp(error, max=50)

    if nocs1.numel() == 0 or nocs2.numel() == 0 or points.numel() == 0:
        translation = torch.tensor(0.0)
    else:
        translation = torch.norm(translation1 - translation2)

    diff_exceeds_threshold = torch.abs(torch.rad2deg(angle1) - torch.rad2deg(angle2)) > threshold or translation>0.05

    results.append(not diff_exceeds_threshold)
    errors.append(error)
    translations.append(translation)

    return results, errors, translations

def get_anchor(initial_vector,arbitrary_vector):
    quats = []
    new_action_direction_cam = initial_vector
    # new_forward = np.random.randn(3).astype(np.float32)
    new_forward = np.array([0,1,0])
    # angles = [-180, -160, -140, -120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    # angles = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    # angles = [-180,-170, -160,-150, -140,-130,-120,-110, -100,-90, -80,-70, -60,-50, -40,-30, -20,-10, 0,10, 20, 30,40,50, 60,70, 80,90, 100,110, 120,130, 140,150, 160,170, 180] #旋转角度范围
    angles = [-180, -90, 0, 90, 180]
    for i in range(5):
        action_direction_cam = new_action_direction_cam
        action_direction_cam /= np.linalg.norm(action_direction_cam)

        rotation_axis = find_rotation_axis(action_direction_cam,arbitrary_vector)
        rotation_angle = np.radians(angles[i])
        action_direction_cam = rotate_vector(action_direction_cam, rotation_axis, rotation_angle)

        f1 = 0
        l1 = 0
        l2 = 0
        # compute final pose
        for j in range(4):
            up = np.array(action_direction_cam, dtype=np.float32)
            # forward = np.random.randn(3).astype(np.float32)
            if j == 0:
                forward = new_forward
            elif j == 1:
                forward = -f1
            elif j == 2:
                forward = l1
            else:
                forward = l2
            while abs(up @ forward) > 0.99:
                forward = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            left = np.cross(up, forward)
            if j == 0:
                f1 = forward
                l1 = left
            elif j == 1:
                l2 = left
            left /= np.linalg.norm(left)
            forward = np.cross(left, up)
            forward /= np.linalg.norm(forward)
            rotmat = np.eye(3).astype(np.float32)
            rotmat[:3, 0] = forward  # z
            rotmat[:3, 1] = left  # x
            rotmat[:3, 2] = up  # y

            r = R.from_matrix(rotmat)

            quat = r.as_quat()
            quats.append(quat)
    return quats

def find_rotation_axis(initial_vector,arbitrary_vector):
    rotation_axis = np.cross(initial_vector, arbitrary_vector)

    rotation_axis /= np.linalg.norm(rotation_axis)

    return rotation_axis

def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)

    rotation_matrix = np.array([
        [np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle)),
         axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
         axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],

        [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
         np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle)),
         axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],

        [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
         axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
         np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle))]
    ])

    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector


