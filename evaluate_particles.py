
import io
import os
import ast
import xml
import torch
import trimesh
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance

from coupled_particle_filters.evaluation_support import (
    NullRunContext,
    combined_mesh_bounds,
    compute_heatmap_torch,
    image_artifact,
    pxlpos2pcd,
    simplify_mesh,
    torch_project_to_image_plane,
)

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set seeds for reproducibility
torch.manual_seed(777)
np.random.seed(777)

# the metrics we want to evaluate
global METRICS
METRICS =  ['Global Sample Precision', 'Local Sample Precision'] # ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'True Negative Rate', 'Balanced Accuracy',]

class Evaluator3D:
    def __init__(self, object_bag_name='ikea01_o', run_name='default',
                 path2data=None,
                 run_context=None,
                 precompute=False
                 ):
        # setup paths
        self.base_path = os.environ.get('CPF_BASE_PATH', '.')
        self.path_to_data = path2data or os.path.join('data', 'rbo_dataset')
        self.path_to_interactions = os.path.join(self.path_to_data, 'interactions')
        self.object_name = object_bag_name[:-4]
        self.object_bag_name = object_bag_name
        self.log_path = self.make_log_dir(object_bag_name[:-2], run_name)
        self.path_to_object = os.path.join(self.path_to_data, 'objects', self.object_name)

        self.run_context = run_context or NullRunContext()

        # load bag annotations and meshes
        self.interactions = pd.read_csv(os.path.join(self.path_to_data,'interactions/interactions_index.csv'))
        if not precompute:
            self.meshes = self.get_urdf()
            self.transform_mesh_to_camera = self.load_transforms_to_camera()
            self.make_kaolin_meshes()

        # logging conditions
        self.counter = 0
        self.log = False
        self.render=False
        self.img = None
        # to track particles for interaction volumes
        self.num_particles_per_mesh = {"3D":{}, "2D":{}}
        # used metrics
        self.metric_names = METRICS
        # evaluate hap in 2d or not
        self.evaluate_in_2d = False

    def rpy_to_matrix(self,coords):
        """
        Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

        Parameters
        ----------
        coords : (3,) float
            The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).

        Returns
        -------
        R : (3,3) float
            The corresponding homogenous 3x3 rotation matrix.
        """

        coords = np.asanyarray(coords)
        c3, c2, c1 = np.cos(coords)
        s3, s2, s1 = np.sin(coords)

        return np.array([
            [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
            [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
            [-s2, c2 * s3, c2 * c3]
        ])

    def xyz_rpy_to_matrix(self, xyz_rpy):
        """
        Convert xyz_rpy coordinates to a 4x4 homogenous matrix.

        Parameters
        ----------
        xyz_rpy : (6,) float
            The xyz_rpy vector.

        Returns
        -------
        matrix : (4,4) float
            The homogenous transform matrix.
        """

        matrix = np.eye(4)
        matrix[:3,3] = xyz_rpy[:3]
        matrix[:3,:3] = self.rpy_to_matrix(xyz_rpy[3:])
        return matrix

    def from_np_array(self, array_string):
        """
        Create a Numpy array from a string representation.
        """
        array_string = ','.join(array_string.replace('[ ', '[').split())
        return np.array(ast.literal_eval(array_string))

    def get_info_from_xml(self, xmltree):
        """
        Load the orignal object mesh and its interaction volumes from an URDF file as an XMLTree.
        We simplify the mesh to reduce the number of vertices and faces and increase performance.
        Additionally, we extract the origin information to transform the mesh to the frame of the object.

        Parameters:
        -----------
        xmltree : xml.etree.ElementTree
            The XML tree containing visual elements, from which mesh file paths and origin
            information are extracted.

        Returns:
        --------
        A List of Dicts for each Meshes containing the following keys:
            - 'mesh': The trimesh object for the simplified mesh.
            - 'orig_mesh': The trimesh object for the original mesh.
            - 'transform_to_frame': A transformation matrix (4x4) from the origin to the frame.
            - 'source_frame': The name of the source frame from which the transformation applies.
            - 'name': The file name of the mesh (without the extension).
        """

        mesh_trafo_pairs = []
        parent_map = {c:p for p in xmltree.iter() for c in p}

        for visual in xmltree.iter('visual'):
            source_frame = parent_map[visual].attrib['name']
            origin = visual[0].attrib
            xyz = np.fromstring(origin["xyz"], dtype=float, sep=' ')
            rpy = np.fromstring(origin["rpy"], dtype=float, sep=' ')

            mesh_path_ros = visual[1][0].attrib['filename']
            mesh_path = '/'.join(mesh_path_ros.split('/')[-4:])
            # some meshes are required for loading but not for evaluation. Therefore, we skip them.
            if mesh_path.split('/')[-1].split('.')[0] == 'microwave':
                tri_mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 0.001], [0, 0.001, 0]], faces=[[0, 1, 2]])
            elif not any(bad_base in mesh_path for bad_base in ['ikea_base', 'cabinet_base', 'ikeasmall_base']):
                tri_mesh = trimesh.load(os.path.join(self.path_to_data, mesh_path), force='mesh')
            else:
                tri_mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 0.001], [0, 0.001, 0]], faces=[[0, 1, 2]])

            orig_mesh = trimesh.load(os.path.join(self.path_to_data, 'objects_original'+mesh_path[7:]), force='mesh')

            tri_mesh = simplify_mesh(tri_mesh, face_count=5000)
            orig_mesh = simplify_mesh(orig_mesh, face_count=5000)


            transform_mesh_to_frame = self.xyz_rpy_to_matrix(np.concatenate((xyz,rpy)))

            mesh_trafo_pairs.append({'mesh':tri_mesh,
                                     'orig_mesh':orig_mesh,
                                    'transform_to_frame':transform_mesh_to_frame,
                                    'source_frame': source_frame,
                                    'name': mesh_path_ros.split('/')[-1][:-4]})
        return mesh_trafo_pairs

    def get_config(self):
        """
        The mesh configuration vaires between recording dates and we have to get the right one
        """
        config_date = self.interactions.loc[self.interactions['Name']==self.object_bag_name[:-2]][['recording date(marker set id)']].values
        config_name = 'configuration_' + config_date[0][0]
        return config_name

    def get_urdf(self):
        """
        Loads the URDF file for the object and extracts the meshes and their transformations.
        """
        file = open(os.path.join(self.path_to_object, self.get_config(), self.object_name+".urdf"), "r").read()
        ao_description = xml.etree.ElementTree.fromstring(file)
        meshes = self.get_info_from_xml(ao_description)
        return meshes

    def make_kaolin_meshes(self):
        """
        Converts trimesh objects to kaolin mesh objects.
        This is necessary for fast point-to-mesh distance computations as it can be performed on the GPU.
        """
        for mesh in self.meshes:
            vertices = torch.tensor(mesh['mesh'].vertices, dtype=float).cuda().unsqueeze(0)
            faces = torch.tensor(mesh['mesh'].faces, dtype=torch.int64).cuda()
            mesh["kaolin_fv"] = index_vertices_by_faces(vertices, faces).float()
            # Untransformed whole-object vertices for fast per-timestamp bounds.
            mesh["orig_vertices"] = np.asarray(mesh['orig_mesh'].vertices, dtype=float)

    def load_transforms_to_camera(self):
        """
        Load and store the mesh transformation to the camera frame.
        """
        transform_mesh_to_camera = pd.read_csv(os.path.join(self.path_to_interactions,
                                                            self.object_name,
                                                            self.object_bag_name + '_tf_mesh_to_rgb_opt_cam.csv'))

        for key in transform_mesh_to_camera.columns[2:]:
            transform_mesh_to_camera[key] = transform_mesh_to_camera[key].apply(lambda x: self.from_np_array(x))
        return transform_mesh_to_camera

    def transform_meshes_for_timestamp(self, timestamp=-1):
        """
        Applies the transformation from object frame to camera frame to the meshes for a given timestamp.
        Some timestamps are missing and take the closest one.
        """

        transform_meshes = deepcopy(self.meshes) # has to be deepcopy because trimesh objects are mutable
        df = self.transform_mesh_to_camera['timestamp']
        nearest_timestamp_idx = abs(df-timestamp).argmin()

        for mesh_dict in transform_meshes:

            T = self.transform_mesh_to_camera.iloc[nearest_timestamp_idx][mesh_dict['name']]
            T_cuda = torch.Tensor(T).cuda()

            mesh_dict['mesh'] = mesh_dict['mesh'].apply_transform(T)
            mesh_dict['orig_mesh'] = mesh_dict['orig_mesh'].apply_transform(T)
            mesh_dict['kaolin_fv'] = mesh_dict['kaolin_fv'] @ T_cuda[:3,:3].T  + T_cuda[:3,3]
        return transform_meshes

    def transform_metric_meshes_for_timestamp(self, timestamp=-1):
        """
        Transform only the metric-relevant mesh data to the camera frame.

        The trimesh objects stay untouched (transform_meshes_for_timestamp deepcopies
        them because apply_transform mutates in place), so the metric path avoids the
        per-frame deepcopy entirely: it transforms the cached kaolin face vertices and
        the cached whole-object vertices for the bounding box instead. The rendering
        path keeps using transform_meshes_for_timestamp.
        """
        df = self.transform_mesh_to_camera['timestamp']
        nearest_timestamp_idx = abs(df-timestamp).argmin()

        metric_meshes = []
        for mesh_dict in self.meshes:
            T = self.transform_mesh_to_camera.iloc[nearest_timestamp_idx][mesh_dict['name']]
            T_cuda = torch.Tensor(T).cuda()
            transformed_vertices = mesh_dict['orig_vertices'] @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
            metric_meshes.append({
                'name': mesh_dict['name'],
                'kaolin_fv': mesh_dict['kaolin_fv'] @ T_cuda[:3, :3].T + T_cuda[:3, 3],
                'orig_bounds': (transformed_vertices.min(axis=0), transformed_vertices.max(axis=0)),
            })
        return metric_meshes


    def label_points_based_on_mesh_distance(self, transformed_meshes, points, tresh=0.02):
        """
        Labels points based on their distance to the mesh using point-to-mesh distance from Kaolin.

        This function computes the Euclidean distance between each point and the nearest surface
        of the transformed meshes. If the distance is below the specified threshold (`tresh`),
        the point is considered "near surface" and labeled accordingly.

        Parameters:
        -----------
        transformed_meshes : Object mesh transformed to timestamp t

        points : 3D coordinates of the points that should be labeled, i.e. our particles.

        tresh : threshold for the distance to the mesh surface to be considered "near surface"

        Returns:
        --------
        Near Surface labels for each points stored in the mesh dictionary.
        """
        for m in transformed_meshes:
            distance, index, dist_type = point_to_mesh_distance(points.unsqueeze(0).cuda(),m["kaolin_fv"].float())
            # p2m_distanxe returns squared euclidean distance -> sqrt()
            distance = torch.sqrt(distance).squeeze()
            m['avg_distance'] = distance.mean()
            m['near_surface'] = distance < tresh

        return transformed_meshes

    def label_mesh_points(self, transformed_meshes, pcd, timestamp, tresh):
        '''
        Just a wrapper for the label_points_based_on_mesh_distance function.
        '''
        transformed_meshes = self.label_points_based_on_mesh_distance(transformed_meshes, pcd, tresh=tresh)
        return transformed_meshes

    def render_ground_truth_maks(self, K, timestamp):
        """
        Renders a mesh mask for the given object at timestep t and extracts the 2d bounding box.
        """
        transformed_meshes = self.transform_meshes_for_timestamp(timestamp)
        object_masks = {}

        png = self.render_object('mesh', transformed_meshes, K)

        png[png!=255]=1
        png[png==255]=0
        ground_truth = png[:,:,0]
        #small "fix" selfinflicted bug for unnecessary rendering for now
        for m in transformed_meshes:
            object_masks[m['name']] = png[:,:,0]

        whole_object = self.render_object('orig_mesh', transformed_meshes, K)

        whole_object[whole_object!=255]=1
        whole_object[whole_object==255]=0
        whole_object = whole_object[:,:,0]

        inds = whole_object.nonzero()
        bbox = [inds[0].min(), inds[0].max(), inds[1].min(), inds[1].max()]
        return ground_truth, object_masks, bbox

    def render_object(self, mesh_type, transformed_meshes, K, resolution=(640, 480)):
        """
        Render meshes using software rasterization for stable headless output.
        """
        scene = trimesh.Scene()

        scene.camera_transform = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        scene.camera.K = K

        for m in transformed_meshes:
            scene.add_geometry(m[mesh_type])

        png = scene.save_image(resolution=[640, 480], visible=True)
        f = io.BytesIO(png)
        png = np.array(Image.open(f))[:,:,:3]
        return png

    def classify_particles_for_known_affordance_bbox(self, particles, gt_mask, obj_bbox):
        """
        Computes the Precision for a particle set for the whole scene and inside the bounding box of the object.
        """

        gt_mask = torch.tensor(gt_mask)
        pts_in_bbox = ((particles[:,0] > obj_bbox[0]) & (particles[:,0] < obj_bbox[1]) & (particles[:,1] > obj_bbox[2]) & (particles[:,1] < obj_bbox[3])).float().sum()
        pts_in_target = gt_mask[tuple(particles.T.long())].sum()
        # Local precision is undefined when no particle lies in the bounding box.
        local_precision = pts_in_target / pts_in_bbox if pts_in_bbox > 0 else None
        global_precision = pts_in_target / particles.shape[0]

        return tuple((global_precision, local_precision))


    def classify_particles_for_known_affordances3d(self, particles, transformed_meshes):
        """
        Computes the Precision for a particle set for the whole scene and inside the bounding box of the object.
        We compute the number of particles that are assigned to each interaction volume with a slack value of 0.02m.
        Local Precision: Total Interatable Particles / Total Particles in Bounding Box
        Global Precision: Total Interatable Particles / Total Particles in Scene
        We additionally compute the number of particles that are assigned to each mesh.

        """
        interaction_volume_names = [mesh_['name'] for mesh_ in transformed_meshes] + [' missed'] # names for each interaction volume + missed particles

        particle_labels_stack = torch.stack([mesh_['near_surface'] for mesh_ in transformed_meshes]) # stack labels

        # compute number of particles inside the mesh bounding box
        if 'orig_bounds' in transformed_meshes[0]:
            min_corner = np.min([mesh_['orig_bounds'][0] for mesh_ in transformed_meshes], axis=0)
            max_corner = np.max([mesh_['orig_bounds'][1] for mesh_ in transformed_meshes], axis=0)
        else:
            min_corner, max_corner = combined_mesh_bounds(
                mesh_['orig_mesh'] for mesh_ in transformed_meshes
            )
        inside_mask = torch.all((particles >= torch.Tensor(min_corner)) & (particles <= torch.Tensor(max_corner)), dim=1)
        pts_in_bbox = inside_mask.sum()

        # compute metrics
        interactable_points = (particle_labels_stack.sum(0)!=0).sum()
        # Local precision is undefined when no particle lies in the bounding box.
        local_precision = interactable_points / pts_in_bbox if pts_in_bbox > 0 else None
        global_precision = interactable_points / particles.shape[0]

        values = [mesh_['near_surface'].sum().cpu().item() for mesh_ in transformed_meshes]            # interactable particles for each interaction volume
        values.append((particle_labels_stack.sum(0)==0).sum().cpu().item()) # particles that are not interactable

        # logging for Recall Estimate
        for n,v in zip(interaction_volume_names, values):
            self.run_context.log({f'Particle hits on {n}'+  self.object_bag_name[:-2]: v, "custom_step": self.counter})

        particle_distribution = {n:v for n,v in zip(interaction_volume_names, values)}
        avg_distance = np.array([mesh_['avg_distance'].cpu().item() for mesh_ in transformed_meshes])
        avg_distance = {n:v for n,v in zip(interaction_volume_names, avg_distance)}

        return tuple((global_precision, local_precision, particle_distribution, avg_distance)) # , local_precision

    def render_scene(self, pcd, transformed_meshes, K, colors=None):
        '''
        Render the scene with the particles.
        '''
        scene = trimesh.Scene()
        if colors is None:
            colors = np.zeros((pcd.shape[0], 3))
            colors[:,0] = 255

        scene.add_geometry(trimesh.PointCloud(pcd, colors = colors))
        scene.camera_transform = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        scene.camera.K = K

        for m in transformed_meshes:
            #scene.add_geometry(m['orig_mesh'])
            scene.add_geometry(m['mesh'])
            png = scene.save_image(resolution=[640, 480], visible=True)
            f = io.BytesIO(png)
            png = np.array(Image.open(f))[:,:,:3]
        return png

    def get_current_mesh_volume(self, timestamp, pcd):
        metric_meshes = self.transform_metric_meshes_for_timestamp(timestamp)
        min_corner = np.min([mesh['orig_bounds'][0] for mesh in metric_meshes], axis=0)
        max_corner = np.max([mesh['orig_bounds'][1] for mesh in metric_meshes], axis=0)
        inside_mask = torch.all((pcd >= torch.Tensor(min_corner)) & (pcd <= torch.Tensor(max_corner)), dim=1)
        return pcd[inside_mask]

    def evaluate3d(self, pcd, timestamp, K, tresh=0.02, render=True):
        '''
        Evaluate the particles in 3D.
        '''
        metric_meshes = self.transform_metric_meshes_for_timestamp(timestamp) # transform metric mesh data to current timestamp
        labeled_meshes = self.label_mesh_points(metric_meshes, pcd, timestamp, tresh) # label particles based on distance to mesh
        metrics = self.classify_particles_for_known_affordances3d(pcd, labeled_meshes) # compute metrics
        #self.extract_additional_information(transformed_meshes=labeled_meshes, particles=pcd)

        if self.render and self.log:
            # Rendering needs actual trimesh objects; only this rare path pays for the deepcopy.
            transformed_meshes = self.transform_meshes_for_timestamp(timestamp)

            import cv2
            points = torch_project_to_image_plane(pcd, K)
            points[:,0] = points[:,0].clamp(0,480-1)
            points[:,1] = points[:,1].clamp(0,640-1)

            mesh_image = self.render_object('mesh', transformed_meshes, K)
            mesh_image_orig = self.render_object('orig_mesh', transformed_meshes, K)
            mesh_image[mesh_image.sum(2)==255*3]=0
            mesh_image_orig[mesh_image_orig.sum(2)==255*3]=0
            heat = compute_heatmap_torch(points, torch.ones(points.shape[0]), torch.Tensor([480, 640]), k_ratio=16, normalize=True).squeeze()
            heat_viz =  cv2.cvtColor(cv2.applyColorMap((heat*255).numpy().astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB).squeeze()
            colors_ = heat_viz[points[:,0].long(), points[:,1].long()]
            scene_img = self.render_scene(pcd, transformed_meshes, K, colors=colors_)
            mesh_image[mesh_image.sum(2)==0]=self.img[mesh_image.sum(2)==0]
            mesh_image_orig[mesh_image_orig.sum(2)==0]=self.img[mesh_image_orig.sum(2)==0]

            Image.fromarray(mesh_image).save('./microwave22/ground_truth/ground_truth' + str(self.counter) + '.png')
            Image.fromarray(mesh_image_orig).save('./microwave22/mesh_image/orig_object_mesh' + str(self.counter) + '.png')

            self.run_context.log({"Scene " + self.object_bag_name[:-2]: [image_artifact(scene_img)], "custom_step": self.counter})

        return metrics

    def evaluate2d(self, particles, timestamp, K, tresh=0.02):
        '''
        Evaluate Particles in 2D!
        '''
        ground_truth, mesh_masks, obj_bbox = self.render_ground_truth_maks(K, timestamp)
        metrics = self.classify_particles_for_known_affordance_bbox(particles, ground_truth, obj_bbox)
        #self.extract_additional_information(mesh_masks=mesh_masks, particles=particles)

        return metrics

    def evaluate(self, pcd, timestamp, K, tresh=0.02, depth=None, pipe=None, render=True):
        dim = pcd.shape[1]
        if dim == 3:
            metrics = self.evaluate3d(pcd, timestamp, K, tresh, render=render)
        elif dim == 2:
            point_depth = depth[tuple(pcd.T.long())]
            points3d = pxlpos2pcd(pcd[:, 1].unsqueeze(0), pcd[:, 0].unsqueeze(0), K, point_depth.unsqueeze(0))

            metrics = self.evaluate3d(points3d, timestamp, K, tresh, render=render)
            if self.evaluate_in_2d:
                # The 2D path yields no per-mesh distribution or distance information.
                metrics = self.evaluate2d(pcd, timestamp, K, tresh) + (None, None)
        else:
            raise ValueError("Invalid shape of input particles")


        if pipe is None or 'graspnet' in pipe.cfg['model'].lower():
            for i,m in enumerate(self.metric_names):
                self.run_context.log({ f'Graspability {m} ' + self.object_bag_name[:-2] : metrics[i], "custom_step": self.counter})
        elif 'hap' in pipe.cfg['model'].lower():
            for i,m in enumerate(self.metric_names):
                self.run_context.log({ f'Interaction {m} ' + self.object_bag_name[:-2] : metrics[i], "custom_step": self.counter})
        elif 'where2act' in pipe.cfg['model'].lower():
            for i,m in enumerate(self.metric_names):
                self.run_context.log({ f'Graspability {m} ' + self.object_bag_name[:-2] : metrics[i], "custom_step": self.counter})
        elif 'hrp' in pipe.cfg['model'].lower():
            for i,m in enumerate(self.metric_names):
                self.run_context.log({ f'HRP Affordance {m} ' + self.object_bag_name[:-2] : metrics[i], "custom_step": self.counter})
        return {'Global Precision': metrics[0], 'Local Precision': metrics[1], 'Particle Distribution': metrics[2], 'Avg Distance': metrics[3]}

    def compute_particle_metrics2d(self, particles, mesh_masks):
        if 'missed' not in self.num_particles_per_mesh['2D']:
            self.num_particles_per_mesh['2D']['missed'] = []
            self.num_particles_per_mesh['2D']['multiple_assignments'] = []

        particles_assignment = []
        for mesh_name, mask in mesh_masks.items():
            particles_label = mask[particles[:,0].int(), particles[:,1].int()]
            if mesh_name not in self.num_particles_per_mesh['2D']:
                self.num_particles_per_mesh['2D'][mesh_name] = []
            self.num_particles_per_mesh['2D'][mesh_name].append(particles_label.sum())
            particles_assignment.append(particles_label)

        particles_assigned = np.array(particles_assignment)
        particles_assigned = particles_assigned.sum(0)
        multiple_assignments = (particles_assigned > 1).sum()

        self.num_particles_per_mesh['2D']['missed'].append(particles.shape[0] - particles_assigned.sum() + multiple_assignments)
        self.num_particles_per_mesh['2D']['multiple_assignments'].append(multiple_assignments)

        return True

    def compute_particle_metrics3d(self, particles, transformed_meshes):
        if 'missed' not in self.num_particles_per_mesh['3D']:
            self.num_particles_per_mesh['3D']['missed'] = []
            self.num_particles_per_mesh['3D']['multiple_assignments'] = []

        sum_of_particles = 0
        assignment = np.array([m['near_surface'] for m in transformed_meshes])
        aggregated_assignemnt = assignment.sum(0) # count how often a particle is assigned
        num_multiple_assignments = aggregated_assignemnt[aggregated_assignemnt > 1].sum() #count multiple assignement
        num_multiple_assignments -= (aggregated_assignemnt > 1).sum() # one assignment is correct and needs to be subtracted

        for m in transformed_meshes:
            if m['name'] not in self.num_particles_per_mesh['3D']:
                self.num_particles_per_mesh['3D'][m['name']] = []
            self.num_particles_per_mesh['3D'][m['name']].append(m['near_surface'].sum())
            sum_of_particles += m['near_surface'].sum()

        self.num_particles_per_mesh['3D']['missed'].append(particles.shape[0] - sum_of_particles + num_multiple_assignments)
        self.num_particles_per_mesh['3D']['multiple_assignments'].append(num_multiple_assignments)
        return True

    def extract_additional_information(self, transformed_meshes=None, mesh_masks=None, particles=None):
        if transformed_meshes is not None:
            self.compute_particle_metrics3d(particles, transformed_meshes)
        if mesh_masks is not None:
            self.compute_particle_metrics2d(particles, mesh_masks)
        return True

    def make_plots_for_additional_information(self):
        # Stackplots for 3D
        fs = 24
        num_plots = len(self.num_particles_per_mesh)
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots*6))

        for i,modeltype in enumerate(self.num_particles_per_mesh):
            data = self.num_particles_per_mesh[modeltype]
            keys = ['no assigned interaction (false positive) ', 'assigned to multiple interactions']+['assigned to '+m+ ' interaction' for m in list(data.keys())[2:]]

            values = np.array(list(data.values()))
            axs[i].stackplot(range(values.shape[1]), values, labels=keys)

            axs[i].set_title(modeltype, fontsize=fs)
            axs[i].set_xlabel('Frame', fontsize=fs)
            axs[i].set_ylabel('Interaction Composition', fontsize=fs)
            axs[i].legend(loc='center left',bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=fs)
            axs[i].tick_params(axis='x', labelsize=fs)
            axs[i].tick_params(axis='y', labelsize=fs)

        plt.tight_layout()
        # self.run_context.log({"Point Distributions": fig})
        plt.savefig(os.path.join(self.log_path,'figures/num_particles_per_mesh_3D.png'))


    def make_plots(self, metric_list, model_names = ['GraspNet3D', 'model2']):
        min_len = min([len(m) for m in metric_list])
        same_length_list = [m[:min_len]for m in metric_list]
        metric_list = np.array(same_length_list)

        try:
            metric_list = metric_list.transpose(0, 2, 1)
        except ValueError as error:
            raise ValueError("metric_list must be a rectangular, three-dimensional array") from error
        # Create subplots for each metric
        fs = 24
        fig, axs = plt.subplots(3, 2, figsize=(32,20))
        # Iterate over metrics and create plots
        for j, metrics in enumerate(metric_list):
            for i, metric in enumerate(metrics):
                # Extract values for the metric
                values = metric
                # Create a subplot for the metric
                ax = axs[i//2,i%2]
                # Set the title and labels
                # Plot the values
                ax.plot(values, linewidth="5", label=model_names[j])
                ax.grid(True)
                # Set the y-axis limits
                ax.set_ylim([0, 1])

                ax.set_title(self.metric_names[i], fontsize=fs)
                ax.set_ylabel("Percentage", fontsize=fs)
                ax.set_xlabel("Frame", fontsize=fs)
                ax.tick_params(axis='both', which='major', labelsize='large')
                # Create the legend with larger font size
                ax.legend(loc='center left',bbox_to_anchor=(1, 0.5), frameon=False, fontsize=fs)
                ax.tick_params(axis='x', labelsize=fs)
                ax.tick_params(axis='y', labelsize=fs)

        # Adjust the spacing between subplots
        plt.tight_layout()
        fig.savefig(self.log_path+'/'+'figures/metrics.png', bbox_inches='tight')
        return fig

    def get_current_bounding_box3d(self, t):
        transformed_meshes = self.transform_meshes_for_timestamp(t)
        mesh = sum([m['orig_mesh'] for m in transformed_meshes])
        return mesh.bounding_box_oriented

    def make_log_dir(self, name='object_nr', run_name='deault'):
        object_name = name[:-2]
        log_path = 'logs'

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        log_path = os.path.join(log_path, run_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        log_path = os.path.join(log_path, object_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        log_path = os.path.join(log_path, name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        vis_path = os.path.join(log_path,'figures')
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)

        grasp_path = os.path.join(log_path,'grasps')
        if not os.path.exists(grasp_path):
            os.mkdir(grasp_path)

        return log_path


class EvalMaster():
    def __init__(self, configs, interactions_csv=None) -> None:
        interactions_csv = interactions_csv or os.environ.get(
            'CPF_INTERACTIONS_CSV',
            os.path.join('data', 'interactions', 'interactions_index.csv'),
        )
        self.df = pd.read_csv(interactions_csv)
        self.configs = configs
        self.metric_names = METRICS
        self.instance_scores = {}
        self.category_scores = {}
        self.cross_category_scores = {k:{'grasp':0, 'interaction':0} for k in self.metric_names}


    def setup_evaluation_set(self, rbo_main, category):
        '''
        Collect all bagfiles for the evaluation in the specified path.
        '''
        path = os.path.join(rbo_main, 'interactions2')
        all_bagfiles = sorted([str(path.resolve()) for path in sorted(Path(rbo_main).rglob('*.db3'))])


        if category is not None:
            all_bagfiles = self.filter_bags_by_category(path=path, all_bags=all_bagfiles, categories=category)
            #all_bagfiles = all_bagfiles[::2]

        return all_bagfiles

    def filter_bags_by_category(self, path, all_bags, categories, bag_interactions_csv=None, filter_adverse_cond=False):
        '''
        Returns the bagfiles for a specfic object if the last two characters are digits.
        Otherwise return all bagfiles for all objects.
        Optional: Filter out adverse conditions like dark lighting and cluttered scenes.
        '''
        if categories[0][-2:].isdigit():
            self.df = self.df[self.df['Name'] == categories[0]]
            self.category_scores[categories[0][:-2]] = {k:{'grasp':0, 'interaction':0} for k in self.metric_names}
            self.instance_scores[categories[0]] = {k:{'grasp':0, 'interaction':0} for k in self.metric_names}
            return np.array([os.path.join(path, categories[0][:-2], categories[0]+ '_o_ros2', categories[0]+ '_o_ros2.db3')])

        df = self.df.copy()
        df = df[df['Object'].isin(categories)]
        new_df = pd.DataFrame(columns=df.columns)

        if filter_adverse_cond:
            for obj in df['Object'].unique():
                df_objs = df[df['Object'] == obj]
                df_objs = df_objs[df_objs['Lighting'] != 'dark']
                df_objs = df_objs[df_objs['cluttered'] != 1]
                new_df = new_df.append(df_objs, ignore_index=True)
            df = new_df
        self.instance_scores = {name:{k:{'grasp':0, 'interaction':0} for k in self.metric_names} for name in df['Name'].unique()}
        self.category_scores = {obj:{k:{'grasp':0, 'interaction':0} for k in self.metric_names} for obj in df['Object'].unique()}
        self.cross_category = {'grasp':0, 'interaction':0}

        bag_path = np.array(path + '/' +  df['Object'] +'/'+ df['Name']+'_o_ros2' + '/'+ df['Name'] +'_o_ros2.db3').tolist()
        self.df = df
        return bag_path

if __name__ == "__main__":
    '''
    Test the evaluation.
    '''
    particles3d = np.load("particles3d.npy")

    evaluate = Evaluator3D()

    print(evaluate.evaluate(pcd=particles3d, timestamp=-1, tresh=0.02))

    pct = trimesh.PointCloud(particles3d)
    for i in range(0, len(evaluate.transform_mesh_to_camera['timestamp']), 500):
        scene = trimesh.Scene()
        scene.add_geometry(pct)

        for m in evaluate.transform_meshes_for_timestamp(i):
            scene.add_geometry(m['mesh'])
        #scene.show()
