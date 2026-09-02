"""Metrics and prediction helpers for evaluating a trained POI model.

All errors are reported in millimetres. ``*_proj`` variants are computed after
projecting both the prediction and the ground truth onto the vertebra surface.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from TPTBox import NII
from TPTBox.core.poi import POI

from verpex.geometry.spaces import batch_to_device
from verpex.geometry.surface import (
    surface_project_coords,
)
from verpex.modules.data_modules import POIDataModule
from verpex.modules.poi_module import PoiPredictionModule
from verpex.paths import get_path

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_dataloader(data_module, split: str):
    """Return the dataloader for ``split`` ("val" or "test")."""
    if split == "val":
        return data_module.val_dataloader()
    elif split == "test":
        return data_module.test_dataloader()
    elif split == "train":
        return data_module.train_noaug_dataloader()
    raise ValueError(f"Invalid split: {split}")


def _setup_prediction(dm_path, ckpt_path, split, **dm_kwargs):
    """Load the data module and checkpoint, and return them with a dataloader."""
    """Load data module, model, and dataloader in one step."""
    data_module = load_data_module_from_config(dm_path, **dm_kwargs)
    data_module.setup()
    poi_module = PoiPredictionModule.load_from_checkpoint(ckpt_path)
    poi_module.eval()
    return data_module, poi_module, _get_dataloader(data_module, split)


def _affine_check(batch):
    """Warn if a sample's subregion and vertebra masks disagree on their affine."""
    subreg = NII.load(batch["subreg_path"][0], seg=True)
    vertseg = NII.load(batch["msk_path"][0], seg=True)
    if not subreg.assert_affine(vertseg, raise_error=False, verbose=True):
        print(f"❌ Affine/metadata mismatch between: \nsubreg: {subreg.affine}, \nvertseg: {vertseg.affine}")


def _resolve_save_path(poi_path, poi_file_ending, save_in_dir, save_path, sub, vert, data_module):
    """Work out where one subject's prediction file should be written."""
    if save_in_dir:
        ctd_save_path = poi_path.replace(data_module.poi_file_ending, poi_file_ending)
        if ctd_save_path == poi_path:
            print(
                f"Warning: The save path {ctd_save_path} is the same as the original POI path. "
                "The new file will be saved with the ending '_pred.json'"
            )
            ctd_save_path = poi_path.replace(".json", "_pred.json")
    else:
        os.makedirs(save_path, exist_ok=True)
        ctd_save_path = os.path.join(save_path, str(sub) + "_" + str(vert) + "_" + poi_file_ending)
    return ctd_save_path


def _sibling_path(ctd_save_path: str, suffix: str) -> str:
    """Return a companion output path beside ``ctd_save_path``.

    Derived from the stem rather than by string-replacing ``"_pred.json"``: that
    replacement is a silent no-op for any other ``poi_file_ending``, which made the
    ground truth overwrite the prediction that had just been written.
    """
    stem = ctd_save_path.removesuffix(".json").removesuffix("_pred")
    return stem + suffix


def _write_prediction_files(pois: POI, poi_path: str, ctd_save_path: str, gt_verts: list) -> bool:
    """Save prediction POI, global marker, GT POI, and segmentation files."""
    pois.save(ctd_save_path, verbose=False)
    if not os.path.exists(ctd_save_path):
        print(f"⚠️ Prediction file not found: {ctd_save_path}")
        return False

    POI.load(ctd_save_path).to_global().save_mrk(_sibling_path(ctd_save_path, "_pred_global.json"))

    gt_poi = POI.load(poi_path).extract_region(*gt_verts)
    gt_save_path = _sibling_path(ctd_save_path, "_gt.json")
    gt_poi.save(gt_save_path)
    gt_poi.to_global().save_mrk(_sibling_path(ctd_save_path, "_gt_global.json"))

    seg_vert_path = poi_path.replace("poi.json", "vertseg.nii.gz")
    seg_save_path = _sibling_path(ctd_save_path, "_seg.nii.gz")
    if os.path.exists(seg_vert_path):
        shutil.copy(seg_vert_path, seg_save_path)
    else:
        print(f"⚠️ Segmentation file not found: {seg_vert_path}")

    return True


def _compute_grouped_metrics(df: pd.DataFrame, group_col: str, error_col: str) -> pd.DataFrame:
    """Aggregate ``error_col`` per distinct value of ``group_col``."""
    metrics_df = df.groupby(group_col)[error_col].apply(lambda x: calculate_metrics(x)).apply(pd.Series)
    metrics_df.columns = ["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"]
    return metrics_df


def _filter_high_error(df: pd.DataFrame, error_col: str, threshold: float) -> pd.DataFrame:
    """Return the rows whose ``error_col`` exceeds ``threshold``."""
    filtered = df[df[error_col] > threshold]
    return filtered[["subject", "vertebra", "poi_idx", error_col]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data_module_from_config(config_path, alternative_poi_ending=None, **kwargs):  # noqa: ANN201
    """Rebuild the data module a training run saved alongside its checkpoint."""
    with open(config_path) as f:
        config = json.load(f)
    config["batch_size"] = 1
    if kwargs:
        config.update(kwargs)
    if alternative_poi_ending is not None:
        config["poi_file_ending"] = alternative_poi_ending
    return POIDataModule(**config)


def load_model_from_checkpoint(checkpoint_path):  # noqa: ANN201
    """Load a trained :class:`PoiPredictionModule` from a checkpoint."""
    return PoiPredictionModule.load_from_checkpoint(checkpoint_path)


def combine_centroids(data_list):  # noqa: ANN201
    """Merge per-vertebra prediction entries into one POI for the subject.

    Args:
        data_list: Per-vertebra entries, all from the same subject.

    Returns:
        The subject id and the combined POI.

    Raises:
        ValueError: If ``data_list`` is empty, or the entries disagree on subject,
            shape, zoom or orientation.
    """
    if not data_list:
        # Reachable whenever safe_collate skipped every batch for a subject.
        raise ValueError("combine_centroids needs at least one entry; got an empty list.")

    first_entry = data_list[0]
    expected_subject = first_entry["subject"]
    expected_shape = first_entry["original_shape"]
    expected_zoom = first_entry["original_zoom"]
    expected_orientation = first_entry["original_orientation"]
    expected_rotation = first_entry["original_rotation"]
    expected_origin = first_entry["original_origin"]

    combined_centroids = {}
    for entry in data_list:
        assert entry["subject"] == expected_subject, "Subjects do not match."
        assert entry["original_shape"] == expected_shape, "Original shapes do not match."
        assert entry["original_zoom"] == expected_zoom, "Original zooms do not match."
        assert entry["original_orientation"] == expected_orientation, "Original orientations do not match."
        assert np.allclose(entry["original_rotation"], expected_rotation, rtol=1e-10), "Original rotations do not match."
        for v_idx, p_idx, c in entry["centroids"].items():
            combined_centroids[v_idx, p_idx] = c

    return expected_subject, POI(
        centroids=combined_centroids,
        orientation=expected_orientation,
        zoom=expected_zoom,
        shape=expected_shape,
        rotation=expected_rotation,
        origin=expected_origin,
    )


def np_to_ctd(  # noqa: ANN201
    t,
    vertebra,
    origin,
    rotation,
    idx_list=None,
    shape=(128, 128, 96),
    zoom=(1, 1, 1),
    offset=(0, 0, 0),
    orientation=None,
):
    """Build a POI from an array of predicted coordinates.

    Args:
        t: Coordinates, ``(n_landmarks, 3)``.
        vertebra: Vertebra label these landmarks belong to.
        origin: World origin of the target frame.
        rotation: Rotation matrix of the target frame.
        idx_list: Landmark ids, in the same order as ``t``. Defaults to ``0..n-1``.
        shape: Shape of the target frame.
        zoom: Voxel spacing of the target frame.
        offset: Subtracted from every coordinate, to undo a padding offset.
        orientation: Axis codes of the target frame. Required.

    Returns:
        The assembled POI.

    Raises:
        ValueError: If ``orientation`` is not given.
    """
    if orientation is None:
        raise ValueError("You must provide the orientation of the input POI.")
    ctd = {}
    for i, coords in enumerate(t):
        coords = np.array(coords).astype(float) - np.array(offset).astype(float)
        coords = (coords[0], coords[1], coords[2])
        if idx_list is None:
            ctd[vertebra, i] = coords
        elif i < len(idx_list):
            ctd[vertebra, idx_list[i]] = coords
    return POI(centroids=ctd, orientation=orientation, zoom=zoom, shape=shape, origin=origin, rotation=rotation)


def create_prediction_poi_files(  # noqa: ANN201
    data_module_save_path,
    checkpoint_path,
    poi_file_ending,
    split="val",
    save_in_dir=False,
    save_path=None,
    return_paths=False,
    project=False,
    save_gt_proj=False,
):
    """Run a model over a split and write one predicted POI file per vertebra."""
    print(f"save_gt_proj: {save_gt_proj}")

    if return_paths:
        poi_paths_dict = {}
    if not save_in_dir and save_path is None:
        raise ValueError("Either save_in_dir or save_path must be set")
    if not poi_file_ending.endswith(".json"):
        raise ValueError("The poi_file_ending must be a json file")

    data_module, poi_module, val_dl = _setup_prediction(data_module_save_path, checkpoint_path, split)

    for batch in val_dl:
        batch = batch_to_device(batch, poi_module.device)
        batch = poi_module(batch)

        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]
        target_batch = batch["target"]
        target_indices_batch = batch["target_indices"]
        loss_mask_batch = batch["loss_mask"]
        refined_preds_batch = batch["refined_preds"]

        if save_gt_proj:
            target_batch, _ = surface_project_coords(target_batch, batch["surface"])
        if project:
            refined_preds_projected_batch, _ = surface_project_coords(refined_preds_batch, batch["surface"])

        offset_batch = batch["offset"]
        poi_path_batch = batch["poi_path"]

        _affine_check(batch)

        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        target_batch = target_batch.detach().cpu().numpy()
        if project:
            refined_preds_projected_batch = refined_preds_projected_batch.detach().cpu().numpy()
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        offset_batch = offset_batch.detach().cpu().numpy()
        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()

        pred_batch = refined_preds_projected_batch if project else refined_preds_batch

        for sub, vert, preds, targets, indices, poi_path, offset, mask in zip(
            subject_batch,
            vertebra_batch,
            pred_batch,
            target_batch,
            target_indices_batch,
            poi_path_batch,
            offset_batch,
            loss_mask_batch,
        ):
            preds = preds[mask]
            indices = indices[mask]
            targets = targets[mask]

            ref = POI.load(poi_path)
            print(f"subject {sub}, vertebra {vert}")

            ctd = np_to_ctd(
                preds,
                vert,
                ref.origin,
                ref.rotation,
                idx_list=indices,
                shape=ref.shape,
                zoom=ref.zoom,
                offset=offset,
                orientation=ref.orientation,
            )

            if save_gt_proj:
                gt_proj_ctd = np_to_ctd(
                    targets,
                    vert,
                    ref.origin,
                    ref.rotation,
                    idx_list=indices,
                    shape=ref.shape,
                    zoom=ref.zoom,
                    offset=offset,
                    orientation=ref.orientation,
                )
                save_path_gt_proj = os.path.join(save_path, "gt_projections")
                os.makedirs(save_path_gt_proj, exist_ok=True)
                gt_proj_save_path = os.path.join(save_path_gt_proj, f"{sub}_{vert}_gt_proj.json")
                gt_proj_ctd.save(gt_proj_save_path, verbose=False)
                POI.load(gt_proj_save_path).extract_region(vert).to_global().save_mrk(
                    gt_proj_save_path.replace("gt_proj.json", "gt_proj_global.json")
                )

            ctd_save_path = _resolve_save_path(poi_path, poi_file_ending, save_in_dir, save_path, sub, vert, data_module)
            print("ctd zoom: ", ctd.zoom)

            if _write_prediction_files(ctd, poi_path, ctd_save_path, [vert]) and return_paths:
                poi_paths_dict[sub, vert] = {
                    "gt": poi_path,
                    "pred": ctd_save_path,
                    "seg_vert": poi_path.replace("poi.json", "vertseg.nii.gz"),
                }

    if return_paths:
        return poi_paths_dict


def create_neighbor_prediction_poi_files(  # noqa: ANN201
    data_module_save_path,
    checkpoint_path,
    poi_file_ending,
    split="val",
    save_in_dir=False,
    save_path=None,
    return_paths=False,
    project=False,
):
    """Like :func:`create_prediction_poi_files`, for neighbour-aware models."""
    if return_paths:
        poi_paths_dict = {}
    if not save_in_dir and save_path is None:
        raise ValueError("Either save_in_dir or save_path must be set")
    if not poi_file_ending.endswith(".json"):
        raise ValueError("The poi_file_ending must be a json file")

    data_module, poi_module, val_dl = _setup_prediction(data_module_save_path, checkpoint_path, split)

    for batch in val_dl:
        batch = batch_to_device(batch, poi_module.device)
        batch = poi_module(batch)

        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]
        refined_preds_batch = batch["refined_preds"]
        if project:
            refined_preds_projected_batch, _ = surface_project_coords(refined_preds_batch, batch["surface"])
        target_indices_batch = batch["target_indices"]
        offset_batch = batch["offset"]
        poi_path_batch = batch["poi_path"]
        loss_mask_batch = batch["loss_mask"]
        n_pois_per_vert_batch = batch["n_pois_per_vertebra"]

        _affine_check(batch)

        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        if project:
            refined_preds_projected_batch = refined_preds_projected_batch.detach().cpu().numpy()
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        offset_batch = offset_batch.detach().cpu().numpy()
        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()
        n_pois_per_vert_batch = n_pois_per_vert_batch.detach().cpu().numpy()

        pred_batch = refined_preds_projected_batch if project else refined_preds_batch

        for sub, vert, preds, indices, poi_path, offset, mask, n_pois_per_vert in zip(
            subject_batch,
            vertebra_batch,
            pred_batch,
            target_indices_batch,
            poi_path_batch,
            offset_batch,
            loss_mask_batch,
            n_pois_per_vert_batch,
        ):
            ref = POI.load(poi_path)
            origin, rotation, shape, zoom, orientation = (ref.origin, ref.rotation, ref.shape, ref.zoom, ref.orientation)
            print(f"n_pois_per_vert: {n_pois_per_vert}")

            # The per-subject metadata is bound as default arguments so these
            # closures capture this iteration's values, not the loop variables.
            def _make_ctd(
                slice_preds,
                slice_indices,
                slice_mask,
                v,
                origin=origin,
                rotation=rotation,
                shape=shape,
                zoom=zoom,
                offset=offset,
                orientation=orientation,
            ):
                valid_preds = slice_preds[slice_mask]
                valid_indices = slice_indices[slice_mask]
                return np_to_ctd(
                    valid_preds, v, origin, rotation, idx_list=valid_indices, shape=shape, zoom=zoom, offset=offset, orientation=orientation
                )

            def _centroid_entry(
                ctd,
                sub=sub,
                shape=shape,
                zoom=zoom,
                orientation=orientation,
                rotation=rotation,
                origin=origin,
            ):
                return {
                    "subject": sub,
                    "original_shape": shape,
                    "original_zoom": zoom,
                    "original_orientation": orientation,
                    "original_rotation": rotation,
                    "original_origin": origin,
                    "centroids": ctd.centroids,
                }

            n = n_pois_per_vert
            partial_centroids = [_centroid_entry(_make_ctd(preds[:n], indices[:n], mask[:n], vert))]

            if vert > 1:
                partial_centroids.append(_centroid_entry(_make_ctd(preds[n : 2 * n], indices[n : 2 * n], mask[n : 2 * n], vert - 1)))

            if vert < 24:
                partial_centroids.append(_centroid_entry(_make_ctd(preds[2 * n :], indices[2 * n :], mask[2 * n :], vert + 1)))

            sub, pois = combine_centroids(partial_centroids)

            gt_verts = [vert] + ([vert - 1] if vert > 1 else []) + ([vert + 1] if vert < 24 else [])
            ctd_save_path = _resolve_save_path(poi_path, poi_file_ending, save_in_dir, save_path, sub, vert, data_module)

            if _write_prediction_files(pois, poi_path, ctd_save_path, gt_verts) and return_paths:
                poi_paths_dict[sub, vert] = {
                    "gt": poi_path,
                    "pred": ctd_save_path,
                    "seg_vert": poi_path.replace("poi.json", "vertseg.nii.gz"),
                }

    if return_paths:
        return poi_paths_dict


def run_predictions(  # noqa: ANN201
    data_module_save_path,
    checkpoint_path,
    split="val",
    alternative_poi_ending=None,
    vert_list=None,
    neighbor=False,
):
    """Run a model over a split and collect per-landmark errors."""
    dm_kwargs = {}
    if alternative_poi_ending is not None:
        dm_kwargs["alternative_poi_ending"] = alternative_poi_ending
    if vert_list is not None:
        dm_kwargs["include_vert_list"] = vert_list

    data_module, poi_module, val_dl = _setup_prediction(data_module_save_path, checkpoint_path, split, **dm_kwargs)
    zoom = getattr(data_module, "zoom", (1, 1, 1))
    print(f"ZOOM: {zoom}")

    project_gt = poi_module.hparams.refinement_config["params"]["project_gt"]
    print(f"project_gt: {project_gt}")

    pred_dict = {
        k: []
        for k in [
            "subject",
            "vertebra",
            "poi_idx",
            "target",
            "coarse",
            "refined",
            "coarse_proj",
            "refined_proj",
            "coarse_proj_dist",
            "refined_proj_dist",
            "loss_mask",
            "zoom",
        ]
    }

    for batch in val_dl:
        batch = batch_to_device(batch, poi_module.device)
        batch = poi_module(batch)

        subject_batch = batch["subject"]
        vertebra_batch = batch["vertebra"]
        target_batch = batch["target"]
        target_indices_batch = batch["target_indices"]
        loss_mask_batch = batch["loss_mask"]
        coarse_preds_batch = batch["coarse_preds"]
        refined_preds_batch = batch["refined_preds"]

        if neighbor:
            n_pois_per_vert_batch = batch["n_pois_per_vertebra"]

        if project_gt:
            target_batch, _ = surface_project_coords(target_batch, batch["surface"], debug=False)

        coarse_preds_projected_batch, coarse_pred_proj_distances_batch = surface_project_coords(coarse_preds_batch, batch["surface"])
        refined_preds_projected_batch, refined_preds_proj_distances_batch = surface_project_coords(refined_preds_batch, batch["surface"])

        vertebra_batch = vertebra_batch.detach().cpu().numpy()
        target_batch = target_batch.detach().cpu().numpy()
        target_indices_batch = target_indices_batch.detach().cpu().numpy()
        loss_mask_batch = loss_mask_batch.detach().cpu().numpy()
        coarse_preds_batch = coarse_preds_batch.detach().cpu().numpy()
        refined_preds_batch = refined_preds_batch.detach().cpu().numpy()
        coarse_preds_projected_batch = coarse_preds_projected_batch.detach().cpu().numpy()
        coarse_pred_proj_distances_batch = coarse_pred_proj_distances_batch.detach().cpu().numpy()
        refined_preds_projected_batch = refined_preds_projected_batch.detach().cpu().numpy()
        refined_preds_proj_distances_batch = refined_preds_proj_distances_batch.detach().cpu().numpy()
        if neighbor:
            n_pois_per_vert_batch = n_pois_per_vert_batch.detach().cpu().numpy()

        keys = [
            "target",
            "target_indices",
            "coarse_preds",
            "refined_preds",
            "coarse_preds_projected",
            "coarse_pred_proj_distances",
            "refined_preds_projected",
            "refined_preds_proj_distances",
            "loss_mask",
            "subject",
            "vertebra",
        ]
        batch_list = [
            target_batch,
            target_indices_batch,
            coarse_preds_batch,
            refined_preds_batch,
            coarse_preds_projected_batch,
            coarse_pred_proj_distances_batch,
            refined_preds_projected_batch,
            refined_preds_proj_distances_batch,
            loss_mask_batch,
            subject_batch,
            vertebra_batch,
        ]
        if neighbor:
            keys.append("n_pois_per_vert")
            batch_list.append(n_pois_per_vert_batch)

        for values in zip(*batch_list):
            data_dict = dict(zip(keys, values))

            if neighbor:
                n = data_dict["n_pois_per_vert"]
                indices = data_dict["target_indices"][:n]
                targets = data_dict["target"][:n]
                coarse = data_dict["coarse_preds"][:n]
                refined = data_dict["refined_preds"][:n]
                coarse_proj = data_dict["coarse_preds_projected"][:n]
                refined_proj = data_dict["refined_preds_projected"][:n]
                coarse_proj_dist = data_dict["coarse_pred_proj_distances"][:n]
                refined_proj_dist = data_dict["refined_preds_proj_distances"][:n]
                loss_mask = data_dict["loss_mask"][:n]
            else:
                indices = data_dict["target_indices"]
                targets = data_dict["target"]
                coarse = data_dict["coarse_preds"]
                refined = data_dict["refined_preds"]
                coarse_proj = data_dict["coarse_preds_projected"]
                refined_proj = data_dict["refined_preds_projected"]
                coarse_proj_dist = data_dict["coarse_pred_proj_distances"]
                refined_proj_dist = data_dict["refined_preds_proj_distances"]
                loss_mask = data_dict["loss_mask"]

            for poi_idx, t, c, r, c_proj, r_proj, c_proj_dist, r_proj_dist, mask_entry in zip(
                indices,
                targets,
                coarse,
                refined,
                coarse_proj,
                refined_proj,
                coarse_proj_dist,
                refined_proj_dist,
                loss_mask,
            ):
                pred_dict["subject"].append(data_dict["subject"])
                pred_dict["vertebra"].append(data_dict["vertebra"])
                pred_dict["poi_idx"].append(poi_idx)
                pred_dict["target"].append(t)
                pred_dict["coarse"].append(c)
                pred_dict["refined"].append(r)
                pred_dict["coarse_proj"].append(c_proj)
                pred_dict["refined_proj"].append(r_proj)
                pred_dict["coarse_proj_dist"].append(c_proj_dist)
                pred_dict["refined_proj_dist"].append(r_proj_dist)
                pred_dict["loss_mask"].append(mask_entry)
                pred_dict["zoom"].append(zoom)

    return pred_dict


def create_prediction_df(data_module_save_path, checkpoint_path, split="val", alternative_poi_ending=None, neighbor=False, vert_list=None):  # noqa: ANN201
    """Return a DataFrame of per-landmark predictions and errors for a split."""
    from time import perf_counter

    start = perf_counter()
    pred_dict = run_predictions(data_module_save_path, checkpoint_path, split, alternative_poi_ending, vert_list, neighbor)
    print(f"Time taken to run predictions: {perf_counter() - start:.2f} seconds")

    pred_dict["coarse_error"] = [
        np.linalg.norm((np.array(t) - np.array(c)) * np.array(zoom))
        for t, c, zoom in zip(pred_dict["target"], pred_dict["coarse"], pred_dict["zoom"])
    ]
    pred_dict["refined_error"] = [
        np.linalg.norm((np.array(t) - np.array(r)) * np.array(zoom))
        for t, r, zoom in zip(pred_dict["target"], pred_dict["refined"], pred_dict["zoom"])
    ]
    pred_dict["coarse_proj_error"] = [
        np.linalg.norm((np.array(t) - np.array(c)) * np.array(zoom))
        for t, c, zoom in zip(pred_dict["target"], pred_dict["coarse_proj"], pred_dict["zoom"])
    ]
    pred_dict["refined_proj_error"] = [
        np.linalg.norm((np.array(t) - np.array(r)) * np.array(zoom))
        for t, r, zoom in zip(pred_dict["target"], pred_dict["refined_proj"], pred_dict["zoom"])
    ]

    return pd.DataFrame(pred_dict)


def calculate_metrics(errors, threshold=2.0):  # noqa: ANN201
    """Return (mean, median, MSE, accuracy, max) for an array of errors in mm.

    ``accuracy`` is the fraction of landmarks closer than ``threshold`` mm.
    """
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    mse = np.mean(errors**2)
    accuracy = np.mean(errors < threshold)
    max_error = np.max(errors)
    return mean_error, median_error, mse, accuracy, max_error


def calculate_metrics4paper(errors, threshold=2.0, round_digits=2):  # noqa: ANN201
    """Return the same metrics as :func:`calculate_metrics`, rounded for reporting."""
    mean_error = np.mean(errors)
    std = np.std(errors)
    median_error = np.median(errors)
    mse = np.mean(errors**2)
    mse_std = np.std(errors**2)
    accuracy = np.mean(errors < threshold)
    accuracy2 = np.mean(errors < 4.0)
    max_error = np.max(errors)
    return (
        round(mean_error, round_digits),
        round(median_error, round_digits),
        round(mse, round_digits),
        round(accuracy, round_digits),
        round(accuracy2, round_digits),
        round(std, round_digits),
        round(mse_std, round_digits),
        round(max_error, round_digits),
    )


def compute_overall_metrics(df):  # noqa: ANN201
    """Return one metric row per error type, over the whole prediction frame."""
    metrics_df = pd.DataFrame(columns=["Mean Error", "Median Error", "MSE", "Accuracy", "Max Error"])
    for error_type in ["coarse_error", "refined_error", "coarse_proj_error", "refined_proj_error"]:
        metrics_df.loc[error_type] = calculate_metrics(df[error_type])
    return metrics_df


def compute_poi_wise_metrics(df):  # noqa: ANN201
    """Return refinement error aggregated per landmark type."""
    return _compute_grouped_metrics(df, "poi_idx", "refined_error")


def compute_poi_wise_metrics_proj(df):  # noqa: ANN201
    """Return surface-projected refinement error aggregated per landmark type."""
    return _compute_grouped_metrics(df, "poi_idx", "refined_proj_error")


def compute_vert_wise_metrics(df):  # noqa: ANN201
    """Return refinement error aggregated per vertebra."""
    return _compute_grouped_metrics(df, "vertebra", "refined_error")


def compute_vert_wise_metrics_proj(df):  # noqa: ANN201
    """Return surface-projected refinement error aggregated per vertebra."""
    return _compute_grouped_metrics(df, "vertebra", "refined_proj_error")


def compute_sub_wise_metrics(df):  # noqa: ANN201
    """Return refinement error aggregated per subject."""
    return _compute_grouped_metrics(df, "subject", "refined_error")


def compute_sub_wise_metrics_proj(df):  # noqa: ANN201
    """Return surface-projected refinement error aggregated per subject."""
    return _compute_grouped_metrics(df, "subject", "refined_proj_error")


def filter_high_refined_proj_error_pois(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return the landmarks whose projected refinement error exceeds ``threshold`` mm."""
    return _filter_high_error(df, "refined_proj_error", threshold)


def filter_high_refined_error_pois(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return the landmarks whose refinement error exceeds ``threshold`` mm."""
    return _filter_high_error(df, "refined_error", threshold)


def load_and_filter_csv(df):  # noqa: ANN201
    """Drop the vertebra-body subregion landmarks (41-50) from a prediction frame."""
    excluded_poi_idx = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    df_filtered = df[~df["poi_idx"].isin(excluded_poi_idx)]
    print(f"Original row count: {len(df)}")
    print(f"Filtered row count: {len(df_filtered)}")
    print(f"Rows removed: {len(df) - len(df_filtered)}")
    return df_filtered


def main() -> None:
    """Evaluate a trained model and write metric CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_module_save_path", type=str, default="", help="Path to the saved DataModule configuration")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved checkpoint")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on (val/test)")
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save the evaluation results. Defaults to the configured output_root."
    )
    parser.add_argument("--neighbor", action="store_true", help="Whether neighbor predictions were made")
    parser.add_argument("--project", action="store_true", help="Whether the final predictions should be projected to the surface.")
    parser.add_argument("--save_gt_proj", action="store_true", help="Whether to save projected GT POI files.")
    parser.add_argument("--save_files", action="store_true", help="Whether to save the prediction files at all.")

    args = parser.parse_args()

    save_path = args.save_path or str(get_path("output_root"))
    os.makedirs(save_path, exist_ok=True)

    out_metrics = os.path.join(save_path, "overall_metrics.csv")
    if os.path.exists(out_metrics):
        print(f"Overall metrics file already exists at {out_metrics}. Nothing to do; delete it to re-evaluate.")
        sys.exit(0)

    if args.data_module_save_path == "":
        t = Path(args.checkpoint_path)
        for _ in range(3):
            if t.joinpath("data_module_params.json").exists():
                args.data_module_save_path = str(t.joinpath("data_module_params.json"))
                break
            t = t.parent

    vert_list = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 20, 21, 22, 23, 24]

    prediction_df = create_prediction_df(
        data_module_save_path=args.data_module_save_path,
        checkpoint_path=args.checkpoint_path,
        split=args.split,
        neighbor=args.neighbor,
        vert_list=vert_list,
    )
    prediction_df = prediction_df[prediction_df["loss_mask"]]
    prediction_df.to_csv(os.path.join(save_path, "results.csv"))
    print(f"Prediction DataFrame saved {os.path.join(save_path, 'results.csv')}")

    metrics_df = compute_overall_metrics(prediction_df)
    metrics_df.to_csv(out_metrics)
    print("Overall metrics saved")

    if args.save_files:
        if args.project:
            compute_poi_wise_metrics_proj(prediction_df).to_csv(os.path.join(save_path, "poi_metrics_projected.csv"))
            print("POI-wise projected metrics saved")
            compute_vert_wise_metrics_proj(prediction_df).to_csv(os.path.join(save_path, "vertebra_metrics_projected.csv"))
            print("Vertebra-wise projected metrics saved")
            compute_sub_wise_metrics_proj(prediction_df).to_csv(os.path.join(save_path, "subject_metrics_projected.csv"))
            print("Subject-wise projected metrics saved")
            filter_high_refined_proj_error_pois(prediction_df, 10).to_csv(
                os.path.join(save_path, "outliers_refined_proj_error_higher_10.csv")
            )
            print("Outliers (refined_proj_error > 10) saved")

        compute_poi_wise_metrics(prediction_df).to_csv(os.path.join(save_path, "poi_metrics.csv"))
        print("POI-wise metrics saved")
        compute_vert_wise_metrics(prediction_df).to_csv(os.path.join(save_path, "vertebra_metrics.csv"))
        print("Vertebra-wise metrics saved")
        compute_sub_wise_metrics(prediction_df).to_csv(os.path.join(save_path, "subject_metrics.csv"))
        print("Subject-wise metrics saved")
        filter_high_refined_error_pois(prediction_df, 10).to_csv(os.path.join(save_path, "outliers_refined_error_higher_10.csv"))
        print("Outliers (refined_error > 10) saved")

        prediction_files_path = os.path.join(save_path, "prediction_files-no_proj")
        prediction_files_path_proj = os.path.join(save_path, "prediction_files")
        os.makedirs(prediction_files_path, exist_ok=True)
        os.makedirs(prediction_files_path_proj, exist_ok=True)

        if args.neighbor:
            create_neighbor_prediction_poi_files(
                data_module_save_path=args.data_module_save_path,
                checkpoint_path=args.checkpoint_path,
                poi_file_ending="_pred.json",
                split=args.split,
                save_path=prediction_files_path,
                return_paths=True,
                project=args.project,
            )
        else:
            create_prediction_poi_files(
                data_module_save_path=args.data_module_save_path,
                checkpoint_path=args.checkpoint_path,
                poi_file_ending="_pred.json",
                split=args.split,
                save_path=prediction_files_path,
                return_paths=True,
                project=False,
                save_gt_proj=args.save_gt_proj,
            )
            create_prediction_poi_files(
                data_module_save_path=args.data_module_save_path,
                checkpoint_path=args.checkpoint_path,
                poi_file_ending="_pred.json",
                split=args.split,
                save_path=prediction_files_path_proj,
                return_paths=True,
                project=True,
                save_gt_proj=args.save_gt_proj,
            )
        print(f"Saved predictions and ground truths to: {prediction_files_path}")


if __name__ == "__main__":
    main()
