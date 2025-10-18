import json
import os
import os.path as osp
import queue
import random
import re
import subprocess
import threading
from copy import deepcopy
from glob import glob

import numpy as np
import scipy
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from utils import common, point_cloud


class platte:
    def __init__(self) -> None:
        self.color_map = {0: [0, 0, 0]}

    def getcolor(self, id):
        if id in self.color_map:
            return self.color_map[id]
        else:

            while True:
                is_new_color = True
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                for exist_id in self.color_map:
                    exist_color = self.color_map[exist_id]
                    if (
                        r != exist_color[0]
                        or g != exist_color[1]
                        or b != exist_color[2]
                    ):
                        pass
                    else:
                        is_new_color = False
                if is_new_color:
                    self.color_map[id] = [r, g, b]
                    return [r, g, b]


def is_pose_valid(pose):
    is_nan = np.isnan(pose).any() or np.isinf(pose).any()
    if is_nan:
        return False
    R_matrix = pose[:3, :3]
    I = np.identity(3)
    is_rotation_valid = (
        np.isclose(np.matmul(R_matrix, R_matrix.T), I, atol=1e-3)
    ).all and np.isclose(np.linalg.det(R_matrix), 1, atol=1e-3)
    if not is_rotation_valid:
        return False
    return True


def load_frame_idxs(data_split_dir, scan_id, skip=None):
    num_frames = len(
        glob(osp.join(data_split_dir, "scenes", scan_id, "data", "color", "*.jpg"))
    )

    if skip is None:
        frame_idxs = ["{:d}".format(frame_idx) for frame_idx in range(0, num_frames)]
    else:
        frame_idxs = [
            "{:d}".format(frame_idx) for frame_idx in range(0, num_frames, skip)
        ]
    return frame_idxs


def load_frame_paths(data_split_dir, scan_id, skip=None):
    frame_idxs = load_frame_idxs(data_split_dir, scan_id, skip)
    img_folder = osp.join(data_split_dir, "scenes", scan_id, "data", "color")
    img_paths = {}
    for frame_idx in frame_idxs:
        img_name = "{}.jpg".format(frame_idx)
        img_path = osp.join(img_folder, img_name)
        img_paths[frame_idx] = img_path
    return img_paths


def load_mesh_path(data_split_dir, scan_id):
    mesh_folder = osp.join(data_split_dir, "scenes", scan_id)
    mesh_path = osp.join(mesh_folder, "{}_vh_clean_2.labels.ply".format(scan_id))
    return mesh_path


def load_frame_poses(data_split_dir, scan_id, skip=None, type="matrix"):
    frame_idxs = load_frame_idxs(data_split_dir, scan_id, skip)
    pose_folder = osp.join(data_split_dir, "scenes", scan_id, "data", "pose")
    poses_W_C = {}
    for frame_idx in frame_idxs:
        pose_file = osp.join(pose_folder, "{}.txt".format(frame_idx))
        pose = np.loadtxt(pose_file)
        if not is_pose_valid(pose):
            continue
        if type == "matrix":
            poses_W_C[frame_idx] = pose
        elif type == "quat_trans":
            T_pose = pose
            quaternion = R.from_matrix(T_pose[:3, :3]).as_quat()
            translation = T_pose[:3, 3]
            pose = np.concatenate([quaternion, translation])
            poses_W_C[frame_idx] = pose
        else:
            raise ValueError("Invalid type")
    return poses_W_C


def load_masks(data_dir, scan_id):
    mask_file = osp.join(data_dir, "files/gt_projection/obj_id_pkl", f"{scan_id}.pkl")
    mask = common.load_pkl_data(mask_file)
    return mask


def load_frame_intrinsics(data_split_dir, scan_id, sensor="color"):
    if sensor == "color":
        intrinsic_file = osp.join(
            data_split_dir,
            "scenes",
            scan_id,
            "data",
            "intrinsic",
            "intrinsic_color.txt",
        )
        rgb_intrinsic = np.loadtxt(intrinsic_file)[:3, :3]
        # some color intrinsic matrix is messed up with depth intrin
        if rgb_intrinsic[0, 0] < 1000:
            ## if so, assign an approx intrinsic matrix, sorry for the hard code, it's Scannet's fault
            rgb_intrinsic = np.array(
                [
                    [1170.187988, 0.000000, 647.750000],
                    [0.000000, 1170.187988, 483.750000],
                    [0.000000, 0.000000, 1.000000],
                ]
            )
        return {"intrinsic_mat": np.array(rgb_intrinsic), "width": 1296, "height": 968}
    elif sensor == "depth":
        intrinsic_file = osp.join(
            data_split_dir,
            "scenes",
            scan_id,
            "data",
            "intrinsic",
            "intrinsic_depth.txt",
        )
        depth_intrinsic = np.loadtxt(intrinsic_file)[:3, :3]
        return depth_intrinsic
    else:
        raise ValueError("sensor should be color or depth")


def load_plydata_npy(
    file_path, data_dir=None, scan_id=None, obj_ids=None, return_ply_data=False
):
    ply_data = np.load(file_path)
    points = np.stack([ply_data["x"], ply_data["y"], ply_data["z"]]).transpose((1, 0))
    obj_ids_pc = ply_data["label"]

    if obj_ids is not None:
        if type(obj_ids) == np.ndarray:
            obj_ids_pc_mask = np.isin(obj_ids_pc, obj_ids)
            points = points[np.where(obj_ids_pc_mask is True)[0]]  # Shape (N, 3)
        else:
            points = points[np.where(obj_ids_pc == obj_ids)[0]]

    if return_ply_data:
        return points, ply_data, obj_ids_pc
    else:
        return points, obj_ids_pc


def scenegraphfusion2scan3r(scan_id, prediction_folder, edge2idx, class2idx, cfg):

    inseg_ply_file = osp.join(prediction_folder, "inseg.ply")
    pred_file = osp.join(prediction_folder, "predictions.json")
    ply_save_file = osp.join(prediction_folder, "inseg_filtered.ply")
    data_npy_file = osp.join(prediction_folder, "data.npy")
    cloud_pd, points_pd, segments_pd = point_cloud.load_inseg(inseg_ply_file)
    filter_seg_size = cfg.data.preprocess.filter_segment_size
    # get num of segments
    segment_ids = np.unique(segments_pd)
    segment_ids = segment_ids[segment_ids != 0]
    segments_pd_filtered = list()
    for seg_id in segment_ids:
        pts = points_pd[np.where(segments_pd == seg_id)]
        if len(pts) > filter_seg_size:
            segments_pd_filtered.append(seg_id)
    segment_ids = segments_pd_filtered
    sgfusion_pred = common.load_json(pred_file)[scan_id]
    rel_obj_data_dict = get_pred_obj_rel(sgfusion_pred, edge2idx, class2idx)

    # fuse pred file info and inseg.ply info
    relationships = []
    objects = []
    filtered_segments_ids = []
    ## get segments in both inseg.ply and pred file
    for object_data in rel_obj_data_dict["objects"]:
        if int(object_data["id"]) in segment_ids:
            filtered_segments_ids.append(int(object_data["id"]))
    segment_ids = filtered_segments_ids
    ## get relationships in both inseg.ply and pred file
    for rel in rel_obj_data_dict["relationships"]:
        if int(rel[0]) in segment_ids and int(rel[1]) in segment_ids:
            relationships.append(rel)
    ## get objects in both inseg.ply and pred file
    for seg_id in segment_ids:
        obj_data = [
            object_data
            for object_data in rel_obj_data_dict["objects"]
            if seg_id == int(object_data["id"])
        ]
        if len(obj_data) == 0:
            continue
        objects.append(obj_data[0])
    assert len(segment_ids) == len([object_data["id"] for object_data in objects])
    ## get points
    points_pd_mask = np.isin(segments_pd, segment_ids)
    points_pd = points_pd[np.where(points_pd_mask == True)[0]]
    segments_pd = segments_pd[np.where(points_pd_mask == True)[0]]
    ## filter and merge same part point cloud
    segment_ids, segments_pd, objects, relationships = filter_merge_same_part_pc(
        segments_pd, objects, relationships
    )
    assert len(segment_ids) == len([object_data["id"] for object_data in objects])
    ## get relationship and object data
    relationship_data_dict = {"relationships": relationships}
    object_data_fict = {"objects": objects}
    ## create inseg_filter ply file
    segments_ids_pc_mask = np.isin(segments_pd, segment_ids)
    points_pd = points_pd[np.where(segments_ids_pc_mask == True)[0]]
    segments_pd = segments_pd[np.where(segments_ids_pc_mask == True)[0]]
    verts = []
    for idx, v in enumerate(points_pd):
        vert = (v[0], v[1], v[2], segments_pd[idx])
        verts.append(vert)

    verts = np.asarray(
        verts, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("label", "u2")]
    )
    print(f"Saving file to {data_npy_file}")
    np.save(data_npy_file, verts)

    plydata = PlyData([PlyElement.describe(verts, "vertex", comments=["vertices"])])
    with open(ply_save_file, mode="wb") as f:
        PlyData(plydata).write(f)
    print(f"Saving file to {ply_save_file}")

    # create scan3r data
    x = verts["x"]
    y = verts["y"]
    z = verts["z"]
    object_id = verts["label"]
    vertices = np.empty(
        x.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("objectId", "h")]
    )
    vertices["x"] = x.astype("f4")
    vertices["y"] = y.astype("f4")
    vertices["z"] = z.astype("f4")
    vertices["objectId"] = object_id.astype("h")
    ply_data_pred = vertices
    data_dict = process_ply_data2scan3r(
        scan_id, ply_data_pred, relationship_data_dict, object_data_fict, edge2idx, cfg
    )
    return data_dict


def read_segmentation(filename):
    assert os.path.isfile(filename)
    with open(filename) as f:
        data = json.load(f)
        segsIndices = np.array(data["segIndices"])
    return segsIndices


def load_frame_idxs_per_obj(obj_id, data_dir, scan_id):
    save_pkl_dir = osp.join(data_dir, "files/gt_projection/obj_id_pkl")
    obj_id_imgs = common.load_pkl_data(osp.join(save_pkl_dir, f"{scan_id}.pkl"))

    frames = []
    masks = []

    for frame_idx, obj_id_map in obj_id_imgs.items():
        mask = np.where(obj_id_map == int(obj_id), obj_id_map, 0)
        if mask.sum() > 0:
            frames.append(frame_idx)
            masks.append(mask)
    return frames, masks


def load_ply_data(data_dir, scan_id):
    """Load ply data for a specific object id in a scan"""
    # load ply file inseg_filtered
    ply_data_file = osp.join(
        data_dir, "scene_graph_fusion", scan_id, "inseg_filtered.ply"
    )
    ply_data = PlyData.read(ply_data_file)
    return ply_data


def load_obj_annotations(data_dir, scan_id, obj_id):
    """Load object annotations for a specific object id in a scan"""
    # load ply file inseg_filtered
    ply_data_file = osp.join(
        data_dir, "scene_graph_fusion", scan_id, "inseg_filtered.ply"
    )
    ply_data = PlyData.read(ply_data_file)
    obj_ids = ply_data["vertex"]["label"]
    indices = np.where(obj_ids == obj_id)[0]
    return indices


def get_scannet_path(path: str):
    scan3r_path = re.search(r"(.*/3RScan)", path)
    scan3r_path = scan3r_path.group(1)
    return scan3r_path


def get_scan_id(path: str):
    # e.g. scene0011_00
    scan_id = re.search(r"(scene\d{4}_\d{2})", path)
    scan_id = scan_id.group(1)
    return scan_id


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    import csv

    assert os.path.isfile(filename)

    def represents_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_obj_info_nyu40(filename, label_map):
    assert os.path.isfile(filename)
    objects_info = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = data["segGroups"][i]["objectId"] + 1  # 0 is background
            semantic_label_raw = data["segGroups"][i]["label"]
            semantic_label_nyu40 = label_map[semantic_label_raw]
            segs = data["segGroups"][i]["segments"]
            objects_info[object_id] = {
                "segs": segs,
                "label": semantic_label_nyu40,
                "inst_id": object_id,
            }
    return objects_info


def scannetGtSeg2scan3r(scan_id, scene_folder, label_map, cfg, filter_segment_size=512):
    import open3d as o3d

    color_map = platte()
    # output file
    out_folder = osp.join(scene_folder, "gt_scan3r")
    common.ensure_dir(out_folder)
    data_npy_file = osp.join(out_folder, "data.npy")
    ply_file = osp.join(out_folder, "gt_scan3r.ply")

    # input file
    segment_file = osp.join(
        scene_folder, "{}_vh_clean_2.0.010000.segs.json".format(scan_id)
    )
    label_file = osp.join(scene_folder, "{}_vh_clean.aggregation.json".format(scan_id))
    mesh_file = osp.join(scene_folder, "{}_vh_clean_2.ply".format(scan_id))

    # load data
    segsIndices = read_segmentation(segment_file)
    objects_info = read_obj_info_nyu40(label_file, label_map)
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    assert len(points) == len(segsIndices)
    ## object id indexs
    filter_seg_size = filter_segment_size
    objects_info_filt = {}
    obj_Ids = np.zeros(len(points), dtype=np.uint8)
    sem_Ids = np.zeros(len(points), dtype=np.uint8)
    for obj_id, obj_info in objects_info.items():
        sem_id = obj_info["label"]
        segs = obj_info["segs"]
        num_points_obj = 0
        obj_masks = []
        for seg_id in segs:
            seg_mask = segsIndices == seg_id
            obj_masks.append(seg_mask)
            num_points_seg = np.sum(seg_mask)
            num_points_obj += num_points_seg
        # if object is too small, ignore
        if num_points_obj < filter_seg_size:
            continue

        for mask in obj_masks:
            obj_Ids[mask] = obj_id
            sem_Ids[mask] = sem_id
            num_points_seg = np.sum(mask)
            obj_color = (
                np.array(color_map.getcolor(obj_id)).reshape(1, 3).astype(np.float32)
                / 255
            )
            obj_colors = obj_color.repeat(num_points_seg, axis=0)
            colors[mask] = obj_colors

        objects_info_filt[obj_id] = obj_info
    ## filter out unlabelled points
    valid_mask = sem_Ids != 0
    points = points[valid_mask]
    obj_Ids = obj_Ids[valid_mask]
    sem_Ids = sem_Ids[valid_mask]
    colors = colors[valid_mask]
    # save data
    ## save pcl to ply file
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(points)
    pointcloud_o3d.colors = o3d.utility.Vector3dVector(colors)
    ## save ply file
    o3d.io.write_point_cloud(ply_file, pointcloud_o3d)

    # create scan3r data
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    vertices = np.empty(
        x.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("objectId", "h"),
            ("global_id", "h"),
        ],
    )
    vertices["x"] = x.astype("f4")
    vertices["y"] = y.astype("f4")
    vertices["z"] = z.astype("f4")
    vertices["objectId"] = obj_Ids.astype("h")
    vertices["global_id"] = sem_Ids.astype("h")
    ply_data_pred = vertices
    ## save npy
    np.save(data_npy_file, vertices)

    data_dict = process_gt_seg_2scan3r(scan_id, ply_data_pred, objects_info_filt, cfg)
    return data_dict


def get_pred_obj_rel(sgfusion_pred, edge2idx, class2idx):
    idx2egde = {idx: edge for edge, idx in edge2idx.items()}
    idx2class = {idx: classname for classname, idx in class2idx.items()}
    preds = sgfusion_pred

    # Edges
    relationships = []
    pred_edges = preds["edges"]

    for edge in pred_edges.keys():
        sub = edge.split("_")[0]
        obj = edge.split("_")[1]

        edge_log_softmax = list(pred_edges[edge].values())
        edge_probs = common.log_softmax_to_probabilities(edge_log_softmax)
        edge_id = np.argmax(edge_probs)
        edge_name = idx2egde[edge_id]
        if edge_name not in ["none"] and edge_name is not None:
            relationships.append([str(sub), str(obj), str(edge_id), edge_name])

    # Objects
    objects = []
    object_data = preds["nodes"]

    for object_id in object_data:
        obj_log_softmax = list(object_data[object_id].values())
        obj_probs = common.log_softmax_to_probabilities(obj_log_softmax)

        obj_id = np.argmax(obj_probs)
        obj_name = idx2class[obj_id]

        if obj_name not in ["none"] and obj_name is not None:
            objects.append(
                {"label": obj_name, "id": str(object_id), "global_id": str(obj_id)}
            )

    return {"relationships": relationships, "objects": objects}


def filter_merge_same_part_pc(segments_pd, objects, relationships):
    instanceid2objname = {
        int(object_data["id"]): object_data["label"] for object_data in objects
    }

    pairs = []
    filtered_relationships = []
    for relationship in relationships:
        if relationship[-1] != define.NAME_SAME_PART:
            filtered_relationships.append(relationship)
        elif (
            relationship[-1] == define.NAME_SAME_PART
            and instanceid2objname[int(relationship[0])]
            == instanceid2objname[int(relationship[1])]
        ):
            pairs.append([int(relationship[0]), int(relationship[1])])
            filtered_relationships.append(relationship)

    same_parts = common.merge_duplets(pairs)
    relationship_data = deepcopy(filtered_relationships)

    del_objects_idxs = []

    for same_part in same_parts:
        root_segment_id = same_part[0]

        for part_segment_id in same_part[1:]:
            segments_pd[np.where(segments_pd == part_segment_id)[0]] = root_segment_id

            for idx, object_data_raw in enumerate(objects[:]):
                if int(object_data_raw["id"]) == part_segment_id:
                    del_objects_idxs.append(idx)

            for idx, (sub, ob, rel_id, rel_name) in enumerate(filtered_relationships):
                sub = int(sub)
                ob = int(ob)
                rel_id = int(rel_id)

                if sub == part_segment_id:
                    sub = root_segment_id
                if ob == part_segment_id:
                    ob = root_segment_id

                if sub == ob:
                    continue

                relationship_data[idx][0] = str(sub)
                relationship_data[idx][1] = str(ob)

    del_objects_idxs = list(set(del_objects_idxs))
    object_data = [
        object_data_idx
        for idx, object_data_idx in enumerate(objects)
        if idx not in del_objects_idxs
    ]
    segment_ids_filtered = np.unique(segments_pd)

    return segment_ids_filtered, segments_pd, object_data, relationship_data


def process_ply_data2scan3r(scan_id, ply_data, rels_dict, objs_dict, rel2idx, cfg):
    objects_ids = []
    global_objects_ids = []
    objects_cat = []
    objects_attributes = []
    barry_centers = []

    # obj points
    points = np.stack([ply_data["x"], ply_data["y"], ply_data["z"]]).transpose((1, 0))
    object_points = {}
    for pc_resolution in cfg.data.preprocess.pc_resolutions:
        object_points[pc_resolution] = []
    object_data = objs_dict["objects"]
    for idx, object in enumerate(object_data):

        object_id = int(object["id"])
        object_id_for_pcl = int(object["id"])
        global_object_id = int(object["global_id"])
        obj_pt_idx = np.where(ply_data["objectId"] == object_id)
        obj_pcl = points[obj_pt_idx]

        if obj_pcl.shape[0] < cfg.data.preprocess.min_obj_points:
            continue
        hull = ConvexHull(obj_pcl)
        cx = np.mean(hull.points[hull.vertices, 0])
        cy = np.mean(hull.points[hull.vertices, 1])
        cz = np.mean(hull.points[hull.vertices, 2])
        for pc_resolution in object_points.keys():
            obj_pcl = point_cloud.pcl_farthest_sample(obj_pcl, pc_resolution)
            object_points[pc_resolution].append(obj_pcl)
        barry_centers.append([cx, cy, cz])
        objects_ids.append(object_id)
        global_objects_ids.append(global_object_id)
        objects_cat.append(global_object_id)
    for pc_resolution in object_points.keys():
        object_points[pc_resolution] = np.array(object_points[pc_resolution])

    if len(objects_ids) < 2:
        return -1
    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_ids):
        object_id2idx[v] = index

    # relationship between objects
    relationships = rels_dict["relationships"]
    triples = []
    pairs = []
    edges_cat = []
    for idx, triple in enumerate(relationships):
        sub = int(triple[0])
        obj = int(triple[1])
        rel_id = int(triple[2])
        rel_name = triple[3]

        if rel_name in list(rel2idx.keys()):
            rel_id = int(rel2idx[rel_name])

            if sub in objects_ids and obj in objects_ids:
                assert rel_id <= len(rel2idx)
                triples.append([sub, obj, rel_id])
                edges_cat.append(rel2idx[rel_name])

                if triple[:2] not in pairs:
                    pairs.append([sub, obj])

    if len(pairs) == 0:
        return -1

    # Root Object - object with highest outgoing degree
    all_edge_objects_ids = np.array(pairs).flatten()
    root_obj_id = np.argmax(np.bincount(all_edge_objects_ids))
    root_obj_idx = object_id2idx[root_obj_id]

    # Calculate barry center and relative translation
    rel_trans = []
    for barry_center in barry_centers:
        rel_trans.append(np.subtract(barry_centers[root_obj_idx], barry_center))

    rel_trans = np.array(rel_trans)

    for i in objects_ids:
        for j in objects_ids:
            if i == j or [i, j] in pairs:
                continue
            triples.append([i, j, rel2idx["none"]])  # supplement the 'none' relation
            pairs.append(([i, j]))
            edges_cat.append(rel2idx["none"])

    s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
    s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)

    for index, v in enumerate(s):
        s[index] = object_id2idx[v]  # s_idx

    for index, v in enumerate(o):
        o[index] = object_id2idx[v]  # o_idx

    edges = np.stack((s, o), axis=1)

    data_dict = {}
    data_dict["scan_id"] = scan_id
    data_dict["objects_id"] = np.array(objects_ids)
    data_dict["global_objects_id"] = np.array(global_objects_ids)
    data_dict["objects_cat"] = np.array(objects_cat)
    data_dict["triples"] = triples
    data_dict["pairs"] = pairs
    data_dict["edges"] = edges
    data_dict["obj_points"] = object_points
    data_dict["objects_count"] = len(objects_ids)
    data_dict["edges_count"] = len(edges)
    data_dict["object_id2idx"] = object_id2idx
    data_dict["object_attributes"] = objects_attributes
    data_dict["edges_cat"] = edges_cat
    data_dict["rel_trans"] = rel_trans
    data_dict["root_obj_id"] = root_obj_id
    return data_dict


def process_gt_seg_2scan3r(scan_id, ply_data, objects_info, cfg):
    objects_ids = []
    global_objects_ids = []
    objects_cat = []
    barry_centers = []

    # obj points
    points = np.stack([ply_data["x"], ply_data["y"], ply_data["z"]]).transpose((1, 0))
    object_points = {}
    for pc_resolution in cfg.data.preprocess.pc_resolutions:
        object_points[pc_resolution] = []

    for key, object in objects_info.items():

        object_id = int(object["inst_id"])
        global_object_id = int(object["label"])

        obj_pt_idx = np.where(ply_data["objectId"] == object_id)
        obj_pcl = points[obj_pt_idx]

        if obj_pcl.shape[0] < cfg.data.preprocess.min_obj_points:
            continue
        hull = ConvexHull(obj_pcl)
        cx = np.mean(hull.points[hull.vertices, 0])
        cy = np.mean(hull.points[hull.vertices, 1])
        cz = np.mean(hull.points[hull.vertices, 2])
        for pc_resolution in object_points.keys():
            obj_pcl = point_cloud.pcl_farthest_sample(obj_pcl, pc_resolution)
            object_points[pc_resolution].append(obj_pcl)
        barry_centers.append([cx, cy, cz])
        objects_ids.append(object_id)
        global_objects_ids.append(global_object_id)
        objects_cat.append(global_object_id)
    for pc_resolution in object_points.keys():
        object_points[pc_resolution] = np.array(object_points[pc_resolution])

    if len(objects_ids) < 2:
        return -1
    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_ids):
        object_id2idx[v] = index

    data_dict = {}
    data_dict["scan_id"] = scan_id
    data_dict["objects_id"] = np.array(objects_ids)
    data_dict["global_objects_id"] = np.array(global_objects_ids)
    data_dict["objects_cat"] = np.array(objects_cat)
    data_dict["obj_points"] = object_points
    data_dict["objects_count"] = len(objects_ids)
    data_dict["object_id2idx"] = object_id2idx
    return data_dict


def calculate_bow_node_edge_feats(data_dict_filename, rel2idx):
    data_dict = common.load_pkl_data(data_dict_filename)

    idx_2_rel = {idx: relation_name for relation_name, idx in rel2idx.items()}
    wordToIx = {}
    for key in rel2idx.keys():
        wordToIx[key] = len(wordToIx)

    edge = data_dict["edges"]
    objects_ids = data_dict["objects_id"]
    triples = data_dict["triples"]
    edges = data_dict["edges"]

    entities_edge_names = [None] * len(objects_ids)
    for idx in range(len(edges)):
        edge = edges[idx]
        entity_idx = edge[0]
        rel_name = idx_2_rel[triples[idx][2]]
        if entities_edge_names[entity_idx] is None:
            entities_edge_names[entity_idx] = [rel_name]
        else:
            entities_edge_names[entity_idx].append(rel_name)
    entity_edge_feats = None
    for entity_edge_names in entities_edge_names:
        entity_edge_feat = np.expand_dims(
            make_bow_vector(entity_edge_names, wordToIx), 0
        )
        entity_edge_feats = (
            entity_edge_feat
            if entity_edge_feats is None
            else np.concatenate((entity_edge_feats, entity_edge_feat), axis=0)
        )

    data_dict["bow_vec_object_edge_feats"] = entity_edge_feats
    assert data_dict["bow_vec_object_edge_feats"].shape[0] == data_dict["objects_count"]
    common.write_pkl_data(data_dict, data_dict_filename)


def make_bow_vector(sentence, word_2_idx):
    # create a vector of zeros of vocab size = len(word_to_idx)
    vec = np.zeros(len(word_2_idx))
    for word in sentence:
        if word not in word_2_idx:
            print(word)
            raise ValueError("houston we have a problem")
        else:
            vec[word_2_idx[word]] += 1
    return vec


def save_ply_pcs(data_dir, scan_id, label_file_name, save_file):
    filename_in = osp.join(data_dir, scan_id, label_file_name)
    file = open(filename_in, "rb")
    ply_data = PlyData.read(file)
    file.close()
    x = ply_data["vertex"]["x"]
    y = ply_data["vertex"]["y"]
    z = ply_data["vertex"]["z"]
    red = ply_data["vertex"]["red"]
    green = ply_data["vertex"]["green"]
    blue = ply_data["vertex"]["blue"]
    object_id = ply_data["vertex"]["objectId"]
    global_id = ply_data["vertex"]["globalId"]
    nyu40_id = ply_data["vertex"]["NYU40"]
    eigen13_id = ply_data["vertex"]["Eigen13"]
    rio27_id = ply_data["vertex"]["RIO27"]

    vertices = np.empty(
        len(x),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("objectId", "h"),
            ("globalId", "h"),
            ("NYU40", "u1"),
            ("Eigen13", "u1"),
            ("RIO27", "u1"),
        ],
    )

    vertices["x"] = x.astype("f4")
    vertices["y"] = y.astype("f4")
    vertices["z"] = z.astype("f4")
    vertices["red"] = red.astype("u1")
    vertices["green"] = green.astype("u1")
    vertices["blue"] = blue.astype("u1")
    vertices["objectId"] = object_id.astype("h")
    vertices["globalId"] = global_id.astype("h")
    vertices["NYU40"] = nyu40_id.astype("u1")
    vertices["Eigen13"] = eigen13_id.astype("u1")
    vertices["RIO27"] = rio27_id.astype("u1")

    np.save(save_file, vertices)

    return vertices


def load_scan_pcs(
    data_dir, scan_id, folder="scene_graph_fusion", downsample_voxel_size=0.05
):
    import open3d as o3d

    ply_data_npy_file = osp.join(data_dir, scan_id, folder, "data.npy")
    ply_data = np.load(ply_data_npy_file)
    points = np.stack([ply_data["x"], ply_data["y"], ply_data["z"]]).transpose((1, 0))
    # downsample
    ## create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ## downsample
    pcd_voxeled = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    points_voxeled = np.asarray(pcd_voxeled.points)
    ## augment points_voxeled with one
    points_voxeled = np.concatenate(
        (points_voxeled, np.ones((points_voxeled.shape[0], 1))), axis=1
    )

    return points_voxeled.astype(np.float32)


def raycastImgFromMesh(scene, mesh_triangles_arr, img_w, img_h, intrinsic, pose_C_W):
    import open3d as o3d

    # raycasting
    num_triangles = mesh_triangles_arr.shape[0]
    ## rays
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsic.astype(np.float64),
        extrinsic_matrix=pose_C_W.astype(np.float64),
        width_px=img_w,
        height_px=img_h,
    )
    ans = scene.cast_rays(rays)
    hit_triangles_ids = ans["primitive_ids"].numpy()
    hit_triangles_ids_valid_masks = hit_triangles_ids < num_triangles
    hit_triangles_ids_valid = hit_triangles_ids[hit_triangles_ids_valid_masks]
    hit_triangles_valid = mesh_triangles_arr[hit_triangles_ids_valid]
    hit_points_idx = hit_triangles_valid[:, 0]

    ray_idxs = hit_triangles_ids_valid_masks
    return ray_idxs, hit_points_idx


def getPatchAnno(gt_anno_2D, patch_w, patch_h, th=0.5):
    image_h, image_w = gt_anno_2D.shape
    patch_h_size = int(image_h / patch_h)
    patch_w_size = int(image_w / patch_w)

    patch_annos = np.zeros((patch_h, patch_w), dtype=np.uint64)
    for patch_h_i in range(patch_h):
        h_start = round(patch_h_i * patch_h_size)
        h_end = round((patch_h_i + 1) * patch_h_size)
        for patch_w_j in range(patch_w):
            w_start = round(patch_w_j * patch_w_size)
            w_end = round((patch_w_j + 1) * patch_w_size)
            patch_size = (w_end - w_start) * (h_end - h_start)

            anno = gt_anno_2D[h_start:h_end, w_start:w_end]
            obj_ids, counts = np.unique(anno.reshape(-1), return_counts=True)
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            if max_count > th * patch_size:
                patch_annos[patch_h_i, patch_w_j] = obj_ids[max_idx]
    return patch_annos


class ElasticDistortion:  # from torch-points3d
    """Apply elastic distortion on sparse coordinate space. First projects the position onto a
    voxel grid and then apply the distortion to the voxel grid.

    Parameters
    ----------
    granularity: List[float]
        Granularity of the noise in meters
    magnitude:List[float]
        Noise multiplier in meters
    Returns
    -------
    data: Data
        Returns the same data object with distorted grid
    """

    def __init__(
        self,
        apply_distorsion: bool = True,
        granularity: list = [0.2, 0.8],
        magnitude=[0.4, 1.6],
    ):
        assert len(magnitude) == len(granularity)
        self._apply_distorsion = apply_distorsion
        self._granularity = granularity
        self._magnitude = magnitude

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        coords = coords.numpy()
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=0, fill_value=0
        )
        coords = coords + interp(coords) * magnitude
        return torch.tensor(coords).float()

    def __call__(self, pcs_pos):
        # coords = pcs_pos / self._spatial_resolution
        if self._apply_distorsion:
            if random.random() < 0.95:
                for i in range(len(self._granularity)):
                    pcs_pos = ElasticDistortion.elastic_distortion(
                        pcs_pos,
                        self._granularity[i],
                        self._magnitude[i],
                    )
        return pcs_pos

    def __repr__(self):
        return "{}(apply_distorsion={}, granularity={}, magnitude={})".format(
            self.__class__.__name__,
            self._apply_distorsion,
            self._granularity,
            self._magnitude,
        )


def sampleCandidateScenesForEachScan(
    scan_id, scan_ids, refscans2scans, scans2refscans, num_scenes
):
    scans_same_scene = refscans2scans[scans2refscans[scan_id]]
    # sample other scenes
    sample_candidate_scans = [scan for scan in scan_ids if scan not in scans_same_scene]
    if num_scenes < 0:
        return sample_candidate_scans
    elif num_scenes <= len(sample_candidate_scans):
        return random.sample(sample_candidate_scans, num_scenes)
    else:
        return sample_candidate_scans


def sampleCrossTime(scan_id, refscans2scans, scans2refscans):
    candidate_scans = []
    ref_scan = scans2refscans[scan_id]
    for scan in refscans2scans[ref_scan]:
        if scan != scan_id:
            candidate_scans.append(scan)
    if len(candidate_scans) == 0:
        return None
    else:
        sampled_scan = random.sample(candidate_scans, 1)[0]
        return sampled_scan


def run_bash_batch(commands, jobs_per_step=1):
    class BashThread(threading.Thread):
        def __init__(self, task_queue, id):
            threading.Thread.__init__(self)
            self.queue = task_queue
            self.th_id = id
            self.start()

        def run(self):
            while True:
                try:
                    command = self.queue.get(block=False)
                    subprocess.call(command, shell=True)
                    self.queue.task_done()
                except queue.Empty:
                    break

    class BashThreadPool:
        def __init__(self, task_queue, thread_num):
            self.queue = task_queue
            self.pool = []
            for i in range(thread_num):
                self.pool.append(BashThread(task_queue, i))

        def joinAll(self):
            self.queue.join()

    # task submission
    commands_queue = queue.Queue()
    for command in commands:
        commands_queue.put(command)
    map_eval_thread_pool = BashThreadPool(commands_queue, jobs_per_step)
    map_eval_thread_pool.joinAll()
