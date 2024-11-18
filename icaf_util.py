import numpy as np

def save_results(test_result, pred, camcs_per_point, gt, id):
    # Save the results and gt into hdf5 for further optimization
    batch_size = pred["seg_per_point"].shape[0]
    for b in range(batch_size):
        group = test_result.create_group(f"{id[b]}")
        group.create_dataset(
            "camcs_per_point",
            data=camcs_per_point[b].detach().cpu().numpy(),
            compression="gzip",
        )

        # save prediction results
        raw_segmentations = pred['seg_per_point'][b].detach().cpu().numpy()
        raw_npcs_points = pred['npcs_per_point'][b].detach().cpu().numpy()
        segmentations, npcs_points = get_prediction_vertices(raw_segmentations, raw_npcs_points)
        group.create_dataset('pred_seg_per_point', data=segmentations, compression="gzip")
        group.create_dataset('pred_npcs_per_point', data=npcs_points, compression="gzip")
        raw_naocs_points = pred['naocs_per_point'][b].detach().cpu().numpy()
        _, naocs_points = get_prediction_vertices(raw_segmentations, raw_naocs_points)
        raw_joint_associations = pred['joint_cls_per_point'][b].detach().cpu().numpy()
        joint_associations = np.argmax(raw_joint_associations, axis=1)
        joint_axes = pred['axis_per_point'][b].detach().cpu().numpy()
        point_heatmaps = pred['heatmap_per_point'][b].detach().cpu().numpy().flatten()
        unit_vectors = pred['unitvec_per_point'][b].detach().cpu().numpy()

        group.create_dataset('pred_naocs_per_point', data=naocs_points, compression="gzip")
        group.create_dataset('pred_joint_cls_per_point', data=joint_associations, compression="gzip")
        group.create_dataset('pred_axis_per_point', data=joint_axes, compression="gzip")
        group.create_dataset('pred_heatmap_per_point', data=point_heatmaps, compression="gzip")
        group.create_dataset('pred_unitvec_per_point', data=unit_vectors, compression="gzip")

        # Save the gt
        for k, v in gt.items():
            if k=='urdf_id':
                continue
            group.create_dataset(
                f"gt_{k}", data=gt[k][b].detach().cpu().numpy(), compression="gzip"
            )

def save_ReArt_results(test_result, pred, camcs_per_point, gt, scene, id, idx):
    # Save the results and gt into hdf5 for further optimization
    batch_size = pred["seg_per_point"].shape[0]
    for b in range(batch_size):
        group = test_result.create_group(f"{scene[b]}_{id[b]}_{idx[b]}")
        group.create_dataset(
            "camcs_per_point",
            data=camcs_per_point[b].detach().cpu().numpy(),
            compression="gzip",
        )

        # save prediction results
        raw_segmentations = pred['seg_per_point'][b].detach().cpu().numpy()
        raw_npcs_points = pred['npcs_per_point'][b].detach().cpu().numpy()
        segmentations, npcs_points = get_prediction_vertices(raw_segmentations, raw_npcs_points)
        group.create_dataset('pred_seg_per_point', data=segmentations, compression="gzip")
        group.create_dataset('pred_npcs_per_point', data=npcs_points, compression="gzip")
        raw_naocs_points = pred['naocs_per_point'][b].detach().cpu().numpy()
        _, naocs_points = get_prediction_vertices(raw_segmentations, raw_naocs_points)
        raw_joint_associations = pred['joint_cls_per_point'][b].detach().cpu().numpy()
        joint_associations = np.argmax(raw_joint_associations, axis=1)
        joint_axes = pred['axis_per_point'][b].detach().cpu().numpy()
        point_heatmaps = pred['heatmap_per_point'][b].detach().cpu().numpy().flatten()
        unit_vectors = pred['unitvec_per_point'][b].detach().cpu().numpy()

        group.create_dataset('pred_naocs_per_point', data=naocs_points, compression="gzip")
        group.create_dataset('pred_joint_cls_per_point', data=joint_associations, compression="gzip")
        group.create_dataset('pred_axis_per_point', data=joint_axes, compression="gzip")
        group.create_dataset('pred_heatmap_per_point', data=point_heatmaps, compression="gzip")
        group.create_dataset('pred_unitvec_per_point', data=unit_vectors, compression="gzip")

        # Save the gt
        for k, v in gt.items():
            group.create_dataset(
                f"gt_{k}", data=gt[k][b].detach().cpu().numpy(), compression="gzip"
            )

def save_grasp_results(test_result, pred, camcs_per_point, gt, id):
    # Save the results and gt into hdf5 for further optimization
    batch_size = pred["seg_per_point"].shape[0]
    for b in range(batch_size):
        group = test_result.create_group(f"{id[b]}")
        group.create_dataset(
            "camcs_per_point",
            data=camcs_per_point[b].detach().cpu().numpy(),
            compression="gzip",
        )

        # save prediction results
        raw_segmentations = pred['seg_per_point'][b].detach().cpu().numpy()
        raw_npcs_points = pred['npcs_per_point'][b].detach().cpu().numpy()
        segmentations, npcs_points = get_prediction_vertices(raw_segmentations, raw_npcs_points)
        group.create_dataset('pred_seg_per_point', data=segmentations, compression="gzip")
        group.create_dataset('pred_npcs_per_point', data=npcs_points, compression="gzip")
        raw_naocs_points = pred['naocs_per_point'][b].detach().cpu().numpy()
        _, naocs_points = get_prediction_vertices(raw_segmentations, raw_naocs_points)
        raw_joint_associations = pred['joint_cls_per_point'][b].detach().cpu().numpy()
        joint_associations = np.argmax(raw_joint_associations, axis=1)
        joint_axes = pred['axis_per_point'][b].detach().cpu().numpy()
        point_heatmaps = pred['heatmap_per_point'][b].detach().cpu().numpy().flatten()
        unit_vectors = pred['unitvec_per_point'][b].detach().cpu().numpy()
        D_quats = pred['D_quats_per_point'][b].detach().cpu().numpy()
        anchor_seg_per_point = pred['anchor_seg_per_point'][b].detach().cpu().numpy()
        anchor_candidates = pred['anchor_Candidates_per_point'][b].detach().cpu().numpy()
        grasp_per_point = pred['grasp_seg_per_point'][b].detach().cpu().numpy()


        group.create_dataset('pred_naocs_per_point', data=naocs_points, compression="gzip")
        group.create_dataset('pred_joint_cls_per_point', data=joint_associations, compression="gzip")
        group.create_dataset('pred_axis_per_point', data=joint_axes, compression="gzip")
        group.create_dataset('pred_heatmap_per_point', data=point_heatmaps, compression="gzip")
        group.create_dataset('pred_unitvec_per_point', data=unit_vectors, compression="gzip")
        group.create_dataset('pred_D_quats_per_point', data=D_quats, compression="gzip")
        group.create_dataset('pred_anchor_seg_per_point', data=anchor_seg_per_point, compression="gzip")
        group.create_dataset('pred_anchor_candidates_per_point', data=anchor_candidates, compression="gzip")
        group.create_dataset('pred_grasp_per_point', data=grasp_per_point, compression="gzip")

        # Save the gt
        for k, v in gt.items():
            if k=='urdf_id':
                continue
            group.create_dataset(
                f"gt_{k}", data=gt[k][b].detach().cpu().numpy(), compression="gzip"
            )


def get_prediction_vertices(pred_segmentation, pred_coordinates):
    segmentations = np.argmax(pred_segmentation, axis=1)
    coordinates = pred_coordinates[
        np.arange(pred_coordinates.shape[0]).reshape(-1, 1),
        np.arange(3) + 3 * np.tile(segmentations.reshape(-1, 1), [1, 3])]
    return segmentations, coordinates