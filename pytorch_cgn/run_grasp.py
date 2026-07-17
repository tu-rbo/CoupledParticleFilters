import glob
import os
import argparse
import cv2
import torch
import numpy as np
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils

from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.data import load_available_input_data




class GraspNet:
    def __init__(self, ckpt_dir, forward_passes, arg_configs, force_cpu=False):
        self.local_regions = False
        self.skip_border_objects = True
        self.filter_grasps = False
        self.segmap = None
        self.z_range = [0.2, 3]
        self.forward_passes = forward_passes
        self.all_pts = False
        self.scale_heatmap = True
        self.selected_threshold = 0.30
        # self.vis = visdom.Visdom()

        if ckpt_dir is None:
            ckpt_dir = os.environ.get(
                "CONTACT_GRASPNET_CHECKPOINT",
                os.path.join(os.path.dirname(__file__), "contact_graspnet_pytorch", "checkpoints", "contact_graspnet"),
            )
        if self.forward_passes is None:
            self.forward_passes = 1
        if arg_configs is None:
            arg_configs = []


        global_config = config_utils.load_config(ckpt_dir, batch_size=self.forward_passes,
                                                 arg_configs=arg_configs)
        self.grasp_estimator = GraspEstimator(global_config, force_cpu=force_cpu)

        # Load weights
        model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
        checkpoint_io = CheckpointIO(
            checkpoint_dir=model_checkpoint_dir,
            map_location=self.grasp_estimator.device,
            model=self.grasp_estimator.model,
        )
        try:
            load_dict = checkpoint_io.load('model.pt')
        except FileExistsError:
            print('No model checkpoint found')
            load_dict = {}



    def set_parameters(self, params):
        if "local_regions" in params.keys():
            self.local_regions = params["local_regions"]
        if "skip_border_objects" in params.keys():
            self.skip_border_objects = params["skip_border_objects"]
        if "filter_grasps" in params.keys():
            self.filter_grasps = params["filter_grasps"]
        if "segmap" in params.keys():
            self.segmap = params["segmap"]
        if "z_range" in params.keys():
            self.z_range = params["z_range"]
        if "forward_passes" in params.keys():
            self.forward_passes = params["forward_passes"]
        if "all_pts" in params.keys():
            self.all_pts = params["all_pts"]
        if "selected_threshold" in params.keys():
            self.selected_threshold = params["selected_threshold"]
        if "scale_heatmap" in params.keys():
            self.scale_heatmap = params["scale_heatmap"]

    def full_output(self, rgb, depth, cam_K, segmap=None, draw_pcl=False, regularize_pcd = True):

        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=self.skip_border_objects,
                                                                                    z_range=self.z_range)

        # pc_full_stack = np.array([pc_full[i::15] for i in range(0, 15)])


        # NOTE: all points in graspnet is more stable but also not good
        # NOTE: better is the enemy of good! thank you furkan for this wisdom

        # pred_grasps_cam, scores, contact_pts, _, selections_idcs = self.grasp_estimator.predict_scene_grasps(self.sess,
        #                                                                                                      pc_full,
        #                                                                                                      pc_segments=pc_segments,
        #                                                                                                      local_regions=self.local_regions,
        #                                                                                                      filter_grasps=self.filter_grasps,
        #                                                                                                      forward_passes=self.forward_passes,
        #                                                                                                      regularize=regularize_pcd)
        (
            pred_grasps_cam,
            scores,
            contact_pts,
            _,
            selection_idcs,
        ) = self.grasp_estimator.predict_scene_grasps(
            pc_full,
            pc_segments=pc_segments,
            local_regions=self.local_regions,
            filter_grasps=self.filter_grasps,
            forward_passes=self.forward_passes,
            return_all=True,
        )
        selections_idcs = selection_idcs[-1]

        if draw_pcl:
            visualize_grasps(pc_full, {'-1':pred_grasps_cam[-1][selections_idcs]}, {'-1':scores[-1][selections_idcs]},
                        plot_opencv_cam=True, pc_colors=pc_colors)

        return pred_grasps_cam[-1], scores[-1], contact_pts[-1], selections_idcs

    def infer_from_custom_pointcloud(self, pc_custom, camk, depth):
        pc_segments = torch.ones_like(pc_custom)[:, 0]
        pred_grasps_cam, scores, contact_pts, _, selections_idcs = self.grasp_estimator.predict_scene_grasps(self.sess,
                                                                                                             pc_custom,
                                                                                                             pc_segments=pc_segments,
                                                                                                             local_regions=self.local_regions,
                                                                                                             filter_grasps=self.filter_grasps,
                                                                                                             forward_passes=self.forward_passes)
        pred_grasps_cam = pred_grasps_cam[-1]
        scores = scores[-1]
        contact_pts = contact_pts[-1]

        unproc_output = dict({'predicted_grasps': pred_grasps_cam,
                              'scores': scores,
                              'cntct_pts': contact_pts,
                              'selections_idcs': selections_idcs})

        return self.choose_output(contact_pts, scores, selections_idcs, camk, depth)

    def infer(self, rgb, depth, cam_K, draw_pcl=False, regularize_pcd = True):
        p_grasp, scrs, cntct_pts, selections_idcs = self.full_output(rgb, depth, cam_K, draw_pcl=draw_pcl, regularize_pcd = regularize_pcd)
        unproc_output = dict({'predicted_grasps': p_grasp,
                              'scores': scrs,
                              'cntct_pts': cntct_pts,
                              'selections_idcs': selections_idcs})
        # self.vis.scatter(cntct_pts[selections_idcs], (scrs[selections_idcs]*100).astype(int)+1, win='test',
        #                        opts=dict({'markersize': 2}))
        # self.vis.histogram(scrs[scrs>0.05], win='scrs_distribution', opts=dict({'numbins': 100}))
        # pcd = o.geometry.PointCloud()
        # pcd.points = o.utility.Vector3dVector(cntct_pts)
        # self.vis.scatter(cntct_pts, np.asarray(pcd.cluster_dbscan(0.15,5))+2, win='clustering', opts=dict({'markersize': 2}))
        if self.all_pts:
            return self.heatmap(cntct_pts, scrs, cam_K, depth.shape), unproc_output
        elif self.selected_threshold is not None:
            return self.heatmap(cntct_pts[scrs > self.selected_threshold], scrs[scrs > self.selected_threshold], cam_K,
                                depth.shape), unproc_output
        elif False:
            return self.weighted_heatmap(cntct_pts, scores, depth.shape), unproc_output
        else:
            return self.heatmap(cntct_pts[selections_idcs], scrs[selections_idcs], cam_K, depth.shape), unproc_output

    def grasp_pt_grasp(self, predicted_grasp, cntct_pts, cam_K):
        pts = self.project_pts(cntct_pts, cam_K)
        # grasp map
        return (pts, predicted_grasp)

    def choose_output(self, cntct_pts, scrs, selections_idcs, cam_K, depth):
        if self.all_pts:
            return self.heatmap(cntct_pts, scrs, cam_K, depth.shape)
        elif self.selected_threshold is not None:
            return self.heatmap(cntct_pts[scrs > self.selected_threshold], scrs[scrs > self.selected_threshold], cam_K,
                                depth.shape)
        elif False:
            return self.weighted_heatmap(cntct_pts, scores, depth.shape)
        else:
            return self.heatmap(cntct_pts[selections_idcs], scrs[selections_idcs], cam_K, depth.shape)

    def weighted_heatmap(self, selected_pts, scores, image_size):
        kernel = gaussian_kde(selected_pts.T, weights=scores)
        X, Y = np.mgrid[0:image_size[1], 0:image_size[0]]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        return (Z / Z.max()).T

    def project_pts(self, selected_pts, cam_K):
        pts = selected_pts
        points_3d = torch.tensor(pts, dtype=torch.float32)
        intrinsic_matrix = torch.tensor(cam_K, dtype=torch.float32)
        # Project 3D points onto 2D image plane
        points_2d = torch.matmul(intrinsic_matrix, points_3d.t()).t()
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
        return points_2d

    def heatmap(self, selected_pts, scores, cam_K, image_size):
        points_2d = self.project_pts(selected_pts, cam_K)
        # generate_heatmap_for_each_component(points_2d, 480, 640, torch.Tensor(scores))
        return self.compute_heatmap(points_2d, scores, image_size)[0]

    def compute_heatmap(self, points, scores, image_size, k_ratio=6.0):
        points = np.asarray(points)
        heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        n_points = points.shape[0]
        heat_scalar = 1
        heatmap[tuple(points.T.astype(int))] = scores

        """
        for i in range(n_points):
            x = points[i, 0]
            y = points[i, 1]
            col = int(x)
            row = int(y)
            try:
                heatmap[col, row] += 1.0 * scores[i]
            except:
                col = min(max(col, 0), image_size[0] - 1)
                row = min(max(row, 0), image_size[1] - 1)
                heatmap[col, row] += 1.0 * scores[i]
        """
        k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
        if k_size % 2 == 0:
            k_size += 1
        heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
        if heatmap.max() > 0 and self.scale_heatmap:
            heat_scalar = heatmap.max()
            heatmap /= heatmap.max()
        else:
            heat_scalar = None
        heatmap = heatmap.transpose()
        # mask = torch.zeros_like(heatmap)
        # padding = 30
        # mask[padding:-padding, padding:-padding] = 1
        # heatmap *= mask
        return heatmap, heat_scalar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--forward_passes', type=int, default=1)
    parser.add_argument('--arg_configs', type=str, default=None)
    parser.add_argument('--cpu-only', action='store_true', help='Force running Contact-GraspNet on CPU.')
    parser.add_argument('--depth', type=str, default=None, help='Path to a depth .npy file for the smoke-test runner.')
    parser.add_argument('--rgb', type=str, default=None, help='Path to an RGB .npy file for the smoke-test runner.')
    args = parser.parse_args()

    graspnet = GraspNet(args.ckpt_dir, args.forward_passes, args.arg_configs, force_cpu=args.cpu_only)

    K = np.array([532.8690747933439, 0.0, 313.8061214077935, 0.0, 533.6136423107343, 243.6213315260905, 0.0, 0.0, 1.0]).reshape(3,3)
    if args.depth is None or args.rgb is None:
        raise ValueError("Pass --depth and --rgb .npy files to run this smoke test.")
    depth = np.load(args.depth)
    rgb = np.load(args.rgb)
    output = graspnet.infer(rgb, depth, K, regularize_pcd = False)
    print("GraspNet initialized")
    import ipdb;ipdb.set_trace()
