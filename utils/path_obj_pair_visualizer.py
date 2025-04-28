import logging
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch

sys.path.append("..")
sys.path.append("../..")
import PIL

from configs import Config
from utils import common, scan3r, torch_util

_LOGGER = logging.getLogger(__name__)


class PatchObjectPairVisualizer:
    # not using cuda
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.cfg.data.img_encoding.img_rotate:
            self.image_w = self.cfg.data.img_encoding.resize_h
            self.image_h = self.cfg.data.img_encoding.resize_w
            self.image_patch_w = self.cfg.data.img_encoding.patch_h
            self.image_patch_h = self.cfg.data.img_encoding.patch_w
        else:
            self.image_w = self.cfg.data.img.w
            self.image_h = self.cfg.data.img.h
            self.image_patch_w = self.cfg.data.img_encoding.patch_w
            self.image_patch_h = self.cfg.data.img_encoding.patch_h

        self.patch_w_size_int = int(self.image_w / self.image_patch_w)
        self.patch_h_size_int = int(self.image_h / self.image_patch_h)

        # some traditional dir from SGAligner
        self.use_predicted = cfg.autoencoder.encoder.use_predicted
        self.scan_type = cfg.autoencoder.encoder.scan_type

        # data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = "" if self.scan_type == "scan" else "out"
        scan_dirname = (
            osp.join(scan_dirname, "predicted") if self.use_predicted else scan_dirname
        )

        obj_json_filename = "objects.json"
        obj_json = common.load_json(
            osp.join(self.data_root_dir, "files", obj_json_filename)
        )["scans"]

        # out dir
        self.out_dir = osp.join(cfg.output_dir, "visualize")

        # get scan objs color imformation
        self.scans_objs_color = {}
        for objs_info_ps in obj_json:
            scan_id = objs_info_ps["scan"]
            scan_objs_color = {}
            for obj_info in objs_info_ps["objects"]:
                color_str = obj_info["ply_color"][1:]
                # convert hex color to rgb color
                color_rgb = torch.Tensor(
                    [int(color_str[i : i + 2], 16) for i in (0, 2, 4)]
                )
                scan_objs_color[int(obj_info["id"])] = color_rgb
            self.scans_objs_color[scan_id] = scan_objs_color

        # limit the number of scans to visualize
        self.scan_id_limit = 10
        self.scan_ids = set()

    def visualize(self, data_dict, embs, epoch):

        # generate and save visualized image for each data item in batch for each iteration
        batch_size = data_dict["batch_size"]
        patch_obj_sim_batch = embs["patch_obj_sim"]
        count = 0
        for batch_i in range(batch_size):
            scan_id = data_dict["scan_ids"][batch_i]
            if scan_id not in self.scan_ids:
                if len(self.scan_ids) <= self.scan_id_limit:
                    self.scan_ids.add(scan_id)
                else:
                    continue

            frame_idx = data_dict["frame_idxs"][batch_i]

            obj_3D_id2idx = data_dict["scene_graphs"]["obj_ids"][
                data_dict["assoc_data_dict"][batch_i]["scans_sg_obj_idxs"].cpu()
            ]
            obj_3D_idx2id = {idx: id.item() for idx, id in enumerate(obj_3D_id2idx)}
            e1i_matrix = data_dict["assoc_data_dict"][batch_i]["e1i_matrix"]  # (N_P, O)

            patch_obj_sim_exp = torch.exp(
                patch_obj_sim_batch[batch_i]
            )  # (N_P, N_O), no temperature
            matched_obj_idxs = torch.argmax(patch_obj_sim_exp, dim=1).reshape(
                -1, 1
            )  # (N_P)
            matched_obj_confidence = patch_obj_sim_exp.gather(
                1, matched_obj_idxs
            ).reshape(-1) / patch_obj_sim_exp.sum(
                dim=1
            )  # (N_P)

            # to numpy
            matched_obj_confidence = torch_util.release_cuda(matched_obj_confidence)
            e1i_matrix = torch_util.release_cuda(e1i_matrix)  # (N_P, O)
            img = np.asarray(
                PIL.Image.open(
                    f"{self.cfg.data.root_dir}/scenes/{scan_id.item()}/sequence/frame-{frame_idx.item()}.color.jpg"
                )
            )
            if self.cfg.data.img_encoding.img_rotate:
                img = img.transpose(1, 0, 2)
                img = np.flip(img, 1)
            img = cv2.resize(img, (self.image_w, self.image_h))

            patch_labeled = e1i_matrix.sum(axis=1) > 0  # (N_P)

            # dye image patches with color of matched objects
            alpha_map = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float64)
            img_color = np.zeros_like(img, dtype=np.float64)
            img_correct = np.zeros_like(
                img, dtype=np.float64
            )  # show whether the patch is correctly matched
            correct_color = np.array([0, 255, 0])
            wrong_color = np.array([255, 0, 0])
            not_labeled_color = np.array([0, 0, 0])
            for patch_h_i in range(self.image_patch_h):
                patch_h_shift = patch_h_i * self.image_patch_w
                for patch_w_j in range(self.image_patch_w):
                    start_h = patch_h_i * self.patch_h_size_int
                    end_h = start_h + self.patch_h_size_int
                    start_w = patch_w_j * self.patch_w_size_int
                    end_w = start_w + self.patch_w_size_int
                    patch_w_shift = patch_w_j
                    patch_idx = patch_h_shift + patch_w_shift
                    matched_obj_idx = matched_obj_idxs[patch_idx].item()
                    matched_obj_id = obj_3D_idx2id[matched_obj_idx]
                    if matched_obj_id not in self.scans_objs_color[scan_id]:
                        print(f"Object {matched_obj_id} not in scan {scan_id}")
                        img_color[start_h:end_h, start_w:end_w, :] = not_labeled_color
                        img_correct[start_h:end_h, start_w:end_w, :] = not_labeled_color
                        alpha_map[start_h:end_h, start_w:end_w] = 1
                        continue
                    matched_obj_color = self.scans_objs_color[scan_id][matched_obj_id]
                    true_obj_idx = np.argmax(e1i_matrix[patch_idx]).item()
                    true_obj_id = obj_3D_idx2id[true_obj_idx]
                    true_obj_color = self.scans_objs_color[scan_id][true_obj_id]

                    if not patch_labeled[patch_idx]:
                        img_color[start_h:end_h, start_w:end_w, :] = not_labeled_color
                        img_correct[start_h:end_h, start_w:end_w, :] = not_labeled_color
                        alpha_map[start_h:end_h, start_w:end_w] = 1
                    else:
                        img_color[start_h:end_h, start_w:end_w, :] = matched_obj_color
                        img_correct[start_h:end_h, start_w:end_w, :] = true_obj_color
                        alpha_map[
                            start_h:end_h, start_w:end_w
                        ] = matched_obj_confidence[patch_idx]

            # alpha blending
            alpha_base = 0.5
            alpha_map = np.clip(alpha_map + alpha_base, 0, 1)
            beta_map = np.clip(1 - alpha_map, 0, 1)
            img_color_blend = img_color * alpha_map + img * beta_map
            img_correct_blend = img_correct * alpha_map + img * beta_map

            # from tensor to numpy
            img_color_blend = torch_util.release_cuda(img_color_blend)
            img_correct_blend = torch_util.release_cuda(img_correct_blend)

            # save images
            img_color_blend = img_color_blend.astype(np.uint8)
            img_correct_blend = img_correct_blend.astype(np.uint8)
            # convert rgb to bgr
            img_color_blend = cv2.cvtColor(img_color_blend, cv2.COLOR_RGB2BGR)
            img_correct_blend = cv2.cvtColor(img_correct_blend, cv2.COLOR_RGB2BGR)
            img_color_out_dir = osp.join(self.out_dir, scan_id, frame_idx, "color")
            img_color_out_path = osp.join(
                img_color_out_dir, "epoch_{}.jpg".format(epoch)
            )
            img_correct_out_dir = osp.join(self.out_dir, scan_id, frame_idx, "correct")
            img_correct_out_path = osp.join(
                img_correct_out_dir, "epoch_{}.jpg".format(epoch)
            )
            common.ensure_dir(img_color_out_dir)
            common.ensure_dir(img_correct_out_dir)
            cv2.imwrite(img_color_out_path, img_color_blend)
            cv2.imwrite(img_correct_out_path, img_correct_blend)
            # write also original image
            img_out_path = osp.join(self.out_dir, scan_id, frame_idx, "original.jpg")
            common.ensure_dir(osp.dirname(img_out_path))
            cv2.imwrite(img_out_path, img)
            _LOGGER.info(
                f"Written to {img_color_out_path} - {img_correct_out_path} - {img_out_path}"
            )
            graph_per_obj_count = data_dict["scene_graphs"]["graph_per_obj_count"]
            count += graph_per_obj_count[batch_i].item()
