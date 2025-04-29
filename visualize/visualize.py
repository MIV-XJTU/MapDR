# Official implementation of MapDR
import os
import cv2
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Change the visualization color here
COLORS = {
    '0': (0, 255, 0),  # divider
    '1': (255, 0, 255),  # functional
    '2': (0, 0, 0),  # boundary
    '3': (0, 255, 255),  # centerline
    '4': (255, 0, 0),  # crosswalk
    'sign': (0, 0, 255),
    'semantic_polygon': (0, 0, 255),
    'asso': (0, 0, 255),
}  # BGR

LOCAL_MAP_SIZE = 50
PV_IMAGE_SIZE = (1240, 1920)


def load_info(args):
    """
    load data.json and label.json
    """
    data_json_path = os.path.join(args.data_dir, 'data.json')
    label_json_path = os.path.join(args.data_dir, 'label.json')
    with open(data_json_path, 'r') as f:
        data_info = json.load(f)
    with open(label_json_path, 'r') as f:
        label_info = json.load(f)

    return data_info, label_info


def calculate_pmat(rvec, tvec, intrinsic_matrix):
    """
    calculate pop matrix for transfer coordinate system
    """
    # camera2world
    rot_matrix_inv = R.from_quat(np.array(rvec)).as_matrix()
    # world2camera
    rot_matrix = np.linalg.inv(rot_matrix_inv)
    # world2camera
    tvec = -np.dot(rot_matrix, np.array(tvec))
    # world2camera
    pmat = np.zeros((3, 4), dtype=np.float64)
    # camera2pixel
    pmat_rot = np.dot(intrinsic_matrix, rot_matrix)
    pmat_t = np.dot(intrinsic_matrix, np.array(tvec).reshape(3, 1))
    # camera2pixel
    pmat[:, :3] = pmat_rot
    pmat[:, -1] = pmat_t.reshape(-1)

    return pmat


def project_to_image_use_pmat(xyzs, pmat):
    """
    project enu points to pixel points
    """
    xyzs = np.array(xyzs, dtype=np.float64).reshape(-1, 4)
    xyzs = np.transpose(xyzs)
    pts = np.dot(pmat, xyzs)
    pts = np.transpose(pts)
    pts[:, 0] /= pts[:, 2]
    pts[:, 1] /= pts[:, 2]
    return pts[:, :3]


def judge_points(point1, point2, h, w):
    """
    judge whether the point is projected into the pixel scope

    """
    draw_point1 = point1[:2].astype(int)
    draw_point2 = point2[:2].astype(int)
    if (
        np.min(point1[-1]) < 0
        or np.min(point2[-1]) < 0
        or np.max(point1[-1]) > 50
        or np.max(point2[-1]) > 50
    ):
        return False
    elif (
        np.any(draw_point1 < 0)
        or np.any(draw_point1 >= (w, h))
        or np.any(draw_point2 < 0)
        or np.any(draw_point2 >= (w, h))
    ):
        return False

    return True


def find_nearest_vec_point(asso_centerline_vec_list, attr_bottom_center_point):
    """
    find the the nearest vector point from senmantic polygon
    """

    distances = [
        np.linalg.norm(point - np.asarray(attr_bottom_center_point))
        for point in asso_centerline_vec_list
    ]
    min_distance_index = np.argmin(distances)
    nearest_point = asso_centerline_vec_list[min_distance_index]

    return nearest_point


def create_single_image(raw_img_path, timestamp, data_info, label_info):
    """
    create an frame of visualization from annnotation
    """
    raw_img = cv2.imread(raw_img_path)
    h, w, _ = raw_img.shape
    camera_intrinsic_matrix = data_info['camera_intrinsic_matrix']
    camera_pose = data_info['camera_pose']
    rvec, tvec = camera_pose[timestamp]['rvec_enu'], camera_pose[timestamp]['tvec_enu']
    pmat = calculate_pmat(rvec, tvec, camera_intrinsic_matrix)

    # project vector
    all_vector = data_info['vector']
    for vector_index in all_vector:
        vec_geo = np.asarray(all_vector[vector_index]['vec_geo'])
        vec_type = all_vector[vector_index]['type']
        vec_color = COLORS[vec_type]
        line_thickness = 2 if vec_type == '3' else 3  # centerline
        # skip empty
        if len(vec_geo) == 0:
            continue

        # transfer enu coordinate to pixel points
        cur_s = np.ones([vec_geo.shape[0], 1])
        cur_xyzs = np.concatenate([vec_geo, cur_s], axis=1)
        if len(cur_xyzs) < 2:
            continue
        interpolation_points = []
        for i in range(1, len(cur_xyzs)):
            _xyz = np.linspace(cur_xyzs[i - 1], cur_xyzs[i], 5)
            # if i == 0:
            interpolation_points.append(_xyz)
            # else:
            #     insert_cur_xyzs.append(c_xyz[1:])

        cur_xyzs = np.concatenate(interpolation_points, 0)
        pixel_points = project_to_image_use_pmat(cur_xyzs, pmat)

        # draw vectors
        for j in range(len(pixel_points) - 1):
            point1, point2 = pixel_points[j], pixel_points[j + 1]
            if not judge_points(point1, point2, h, w):
                continue
            draw_point1 = point1[:2].astype(int)
            draw_point2 = point2[:2].astype(int)
            cv2.line(raw_img, draw_point1, draw_point2, vec_color, line_thickness)

    # # project traffic sign
    # traffic_board_pose = np.asarray(data_info['traffic_board_pose'])
    # sign_color = COLORS['sign']
    # cur_s = np.ones([traffic_board_pose.shape[0], 1])
    # cur_xyzs = np.concatenate([traffic_board_pose, cur_s], axis=1)
    # pixel_points = project_to_image_use_pmat(cur_xyzs, pmat)
    # draw_points = np.array(pixel_points[:, :2], dtype=int)
    # cv2.drawContours(raw_img, [draw_points], -1, sign_color, 2)

    # project semantic polygon
    semantic_color = COLORS['semantic_polygon']
    for rule_idx, rule in label_info.items():
        centerline_list = rule['centerline']
        semantic_polygon = np.asarray(rule['semantic_polygon'])
        cur_s = np.ones([semantic_polygon.shape[0], 1])
        cur_xyzs = np.concatenate([semantic_polygon, cur_s], axis=1)
        coord_pos = project_to_image_use_pmat(cur_xyzs, pmat)
        if np.min(coord_pos[:, -1]) < 0:
            continue
        draw_points = np.array(coord_pos[:, :2], dtype=int)
        cv2.drawContours(raw_img, [draw_points], -1, semantic_color, 2)

        # project association
        polygon_bottom_center_point = (
            np.average(semantic_polygon[:, 0]),
            np.average(semantic_polygon[:, 1]),
            min(semantic_polygon[:, 2]),
        )
        for centerline in centerline_list:
            centerline_geo_points = all_vector[str(centerline)]['vec_geo']
            nearest_point = find_nearest_vec_point(
                centerline_geo_points, polygon_bottom_center_point
            )
            asso_line_points = np.asarray([polygon_bottom_center_point, nearest_point])
            cur_s = np.ones([asso_line_points.shape[0], 1])
            cur_xyzs = np.concatenate([asso_line_points, cur_s], axis=1)
            coord_pos = project_to_image_use_pmat(cur_xyzs, pmat)
            if not judge_points(coord_pos[0], coord_pos[1], h, w):
                continue
            cv2.line(
                raw_img,
                coord_pos[0, :2].astype(int),
                coord_pos[1, :2].astype(int),
                COLORS['asso'],
                2,
            )

    return raw_img


def create_bev_image(data_info):
    """
    create local bev map visualization
    """
    canvas_bound = (-LOCAL_MAP_SIZE, LOCAL_MAP_SIZE, 0.1)
    all_vectors = data_info['vector']
    traffic_board_pose = np.asarray(data_info['traffic_board_pose'])
    board_center = np.mean(traffic_board_pose, 0).tolist()

    # draw local map
    canvas = [
        int((canvas_bound[1] - canvas_bound[0]) / canvas_bound[2]),
        int((canvas_bound[1] - canvas_bound[0]) / canvas_bound[2]),
        3,
    ]
    offset = np.array(
        [canvas_bound[0] + board_center[0], canvas_bound[0] + board_center[1]]
    )
    scale = np.array([canvas_bound[2], canvas_bound[2]])
    bev_img = np.ones(canvas, dtype=np.uint8) * 255

    for vector_idx, vector in all_vectors.items():
        polyline = np.asarray(vector['vec_geo'])[:, :2]
        polyline = (polyline - offset) / scale
        polyline = polyline.astype(np.int16)
        vector_type = vector['type']
        color = COLORS[vector_type]
        thichness = 2

        for i in range(polyline.shape[0] - 1):
            cv2.arrowedLine(
                bev_img,
                tuple(polyline[i]),
                tuple(polyline[i + 1]),
                color,
                thickness=thichness,
                tipLength=0.05,
            )

    # draw grid
    img_grid = np.ones_like(bev_img) * 255
    grid = np.arange(0, img_grid.shape[0], 100)
    bev_img[grid, :] = [0, 0, 0]
    bev_img[:, grid] = [0, 0, 0]
    bev_img = cv2.addWeighted(bev_img, 1, img_grid, 0.3, 0).astype(np.uint8)

    # draw sign
    img_center = (int(bev_img.shape[0] / 2), int(bev_img.shape[1] / 2))
    cv2.circle(bev_img, img_center, 10, COLORS['sign'], 2)

    bev_img = cv2.flip(bev_img, 0)

    return bev_img


def create_rule_visualization(label_info, image_size):
    """
    visualize the rule annotation
    """

    canvas = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 0, 0)
    thickness = 2
    start_x, start_y = 50, 50
    line_height = 50

    text = []
    for rule_idx, rule in label_info.items():
        text.append(f"Rule {rule_idx} : {json.dumps(rule['attr_info'])}")

    for line in text:
        cv2.putText(
            canvas, line, (start_x, start_y), font, font_scale, color, thickness
        )
        start_y += line_height

    return canvas


def create_video(args):
    """
    create an video from annotation
    """
    data_info, label_info = load_info(args)
    img_dir = os.path.join(args.data_dir, 'img')
    all_img = sorted(os.listdir(img_dir))

    # create local map bev visualization
    bev_img = create_bev_image(data_info)
    bev_img = cv2.resize(bev_img, (PV_IMAGE_SIZE[0], PV_IMAGE_SIZE[0]))
    # cv2.imwrite(os.path.join(args.save_dir, f'bev_{uid}.jpg'), bev_img)

    # create rule visualization
    image_size = (300, bev_img.shape[1] + PV_IMAGE_SIZE[1])
    rule_img = create_rule_visualization(label_info, image_size)
    # cv2.imwrite(os.path.join(args.save_dir, f'rule_{uid}.jpg'), rule_img)

    # create pv project visualization
    frames = []

    for img_name in all_img:
        raw_img_path = os.path.join(img_dir, img_name)
        timestamp = img_name.replace('.jpg', '')
        project_img = create_single_image(
            raw_img_path, timestamp, data_info, label_info
        )
        bev_pv_img = cv2.hconcat([bev_img, project_img])
        final_img = cv2.vconcat([rule_img, bev_pv_img])
        frames.append(final_img)
        # cv2.imwrite(os.path.join(args.save_dir, f'final_{uid}_{timestamp}.jpg'), final_img)

    # create video
    default_uid = args.data_dir.split('/')[-1]
    output_file = os.path.join(args.save_dir, f'video_{default_uid}.mp4')
    frame_width = final_img.shape[1]
    frame_height = final_img.shape[0]
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()

    return


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(description="get path to datset and save dir")
    parser.add_argument("data_dir", type=str, help="target uid directory")
    parser.add_argument("save_dir", type=str, help="directory to save visualization")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # create visualization
    os.makedirs(args.save_dir)
    create_video(args)
