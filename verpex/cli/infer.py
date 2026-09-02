"""End-to-end POI inference from segmentation masks.

Given vertebra and subregion segmentation masks plus a trained model, this pipeline:

1. loads the masks;
2. cuts out each vertebra and builds a master_df in a temporary directory;
3. reorients and rescales the cutouts to 1 mm isotropic;
4. pads them to a fixed size and arranges them into a batch;
5. runs the model and extracts the predicted landmarks;
6. reverts the landmarks to the original space - un-pad, then **add the cutout
   offset, then rescale**, then reorient. The offset counts voxels in the
   preprocessed grid, so it must be applied before the rescale; see
   :func:`verpex.geometry.spaces.revert_poi_to_original_space`, which explains why
   getting this order wrong is invisible at 1 mm isotropic spacing;
7. saves them as a BIDS POI file and removes the temporary directory.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from TPTBox import BIDS_FILE, NII, BIDS_Global_info, np_utils, v_idx_order
from TPTBox.core.poi import POI
from TypeSaveArgParse import Class_to_ArgParse

import verpex.evaluation.metrics as ev
from verpex.data.dataloading import get_subreg, get_vertseg_bfile, pad_array_to_shape
from verpex.evaluation.metrics import combine_centroids
from verpex.geometry.spaces import batch_to_device, extract_space_meta, revert_poi_to_original_space
from verpex.geometry.surface import surface_project_coords
from verpex.modules.data_modules import NEIGHBOR_DATASETS
from verpex.paths import get_path

poi_flip_pairs = {
    # These are the middle points, i.e. the ones that are not flipped
    81: 81,
    101: 101,
    103: 103,
    102: 102,
    104: 104,
    105: 105,
    106: 106,
    107: 107,
    108: 108,
    125: 125,
    127: 127,
    # Flipped left to right
    83: 82,
    84: 85,
    86: 87,
    88: 89,
    109: 117,
    111: 119,
    110: 118,
    112: 120,
    113: 121,
    114: 122,
    115: 123,
    116: 124,
    # Flipped right to left
    82: 83,
    85: 84,
    87: 86,
    89: 88,
    117: 109,
    118: 110,
    119: 111,
    120: 112,
    121: 113,
    122: 114,
    123: 115,
    124: 116,
}

DEFAULT_POI_INDICES = [
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    127,
]


class SpineInferenceDataset(Dataset):
    """Inference dataset for vertebra-wise POI prediction.

    When include_neighbors=True, the mask also covers adjacent vertebrae and
    poi_indices/poi_list_idx are tripled (current, top, bottom), matching
    the neighbour training setup.
    """

    def __init__(
        self,
        master_df,
        input_shape,
        input_data_type,
        include_vert_list,
        zoom=(1, 1, 1),
        poi_indices=None,
        include_neighbors=False,
    ):
        if poi_indices is None:
            poi_indices = DEFAULT_POI_INDICES
        self.master_df = master_df
        self.input_shape = input_shape
        self.input_data_type = input_data_type
        self.zoom = zoom
        self.include_neighbors = include_neighbors
        self.poi_indices = torch.tensor(poi_indices)
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.include_vert_list = include_vert_list
        self.vert_idx_to_list_idx = {vert: idx for idx, vert in enumerate(include_vert_list)}

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, index):
        row = self.master_df.iloc[index]
        vertebra = row["vert"]
        vert_path = row["vert_path"]
        input_data_path = row["input_data_path"]
        surface_path = row["surface_path"]
        x_min = row["x_min"]
        y_min = row["y_min"]
        z_min = row["z_min"]

        original_orientation = row["original_orientation"]
        original_zoom = row["original_zoom"]
        original_shape = row["original_shape"]
        original_rotation = row["original_rotation"]
        original_origin = row["original_origin"]

        preprocessed_orientation = row["preprocessed_orientation"]
        preprocessed_zoom = row["preprocessed_zoom"]
        preprocessed_rotation = row["preprocessed_rotation"]
        preprocessed_origin = row["preprocessed_origin"]
        preprocessed_shape = row["preprocessed_shape"]

        subject = row["subject"]

        if self.input_data_type in ("ct", "surface_msk+ct"):
            input_data = NII.load(input_data_path, seg=False)
            input_data.normalize_ct(min_out=0, max_out=1, inplace=True)
        else:
            input_data = NII.load(input_data_path, seg=True)
        vertseg = NII.load(vert_path, seg=True)
        surface = NII.load(surface_path, seg=True)
        surface.extract_label_(surface.unique())

        assert input_data.shape == vertseg.shape, f"shape mismatch input={input_data.shape} vert={vertseg.shape}"
        assert input_data.orientation == vertseg.orientation, (
            f"orientation mismatch input={input_data.orientation} vert={vertseg.orientation}"
        )
        assert input_data.orientation == ("L", "A", "S")
        assert all(abs(a - b) < 1e-3 for a, b in zip(input_data.zoom, vertseg.zoom)), (
            f"zoom mismatch input={input_data.zoom} vert={vertseg.zoom}"
        )

        print("zoom in __getitem__: ", input_data.zoom)

        input_data_arr = input_data.get_array()
        vertseg_arr = vertseg.get_array()
        surface_arr = surface.get_array()

        if self.include_neighbors:
            v_idx = self.include_vert_list.index(vertebra)
            neighbor_top = self.include_vert_list[v_idx - 1] if v_idx > 0 else 0
            neighbor_bottom = self.include_vert_list[v_idx + 1] if v_idx < len(self.include_vert_list) - 1 else 0
            actual_vert = [v for v in [vertebra, neighbor_top, neighbor_bottom] if v != 0]
            mask = np.isin(vertseg_arr, actual_vert)
        else:
            mask = vertseg_arr == vertebra

        input_data_arr = input_data_arr * mask
        surface_arr = surface_arr * mask

        if any(s > t for s, t in zip(input_data_arr.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {input_data_arr.shape} > {self.input_shape})")
            return None
        elif any(s > t for s, t in zip(vertseg_arr.shape, self.input_shape)):
            print(f"Skipping subject {subject} vertebra {vertebra} (shape {vertseg_arr.shape} > {self.input_shape})")
            return None

        input_data_arr, offset = pad_array_to_shape(input_data_arr, self.input_shape)
        vertseg_arr, _ = pad_array_to_shape(vertseg_arr, self.input_shape)
        surface_arr, _ = pad_array_to_shape(surface_arr, self.input_shape)

        input_data_t = torch.from_numpy(input_data_arr.astype(np.float32)).unsqueeze(0)
        surface_t = torch.from_numpy(surface_arr.astype(np.float32)).unsqueeze(0)

        if self.input_data_type == "surface_msk+ct":
            input_data_t = torch.cat([surface_t, input_data_t], dim=0)

        poi_indices = self.poi_indices
        poi_list_idx = torch.tensor([self.poi_idx_to_list_idx[poi.item()] for poi in poi_indices])

        if self.include_neighbors:
            poi_indices = torch.cat([poi_indices, poi_indices, poi_indices])
            poi_list_idx = torch.cat([poi_list_idx, poi_list_idx, poi_list_idx])

        return {
            "input": input_data_t,
            "surface": surface_t,
            "vertebra": vertebra,
            "padding_offset": torch.tensor(offset).float(),
            "poi_indices": poi_indices,
            "poi_list_idx": poi_list_idx,
            "vert_list_idx": torch.tensor([self.vert_idx_to_list_idx[vertebra]]),
            "cutout_offset": torch.tensor([x_min, y_min, z_min]),
            "original_orientation": str(original_orientation),
            "original_zoom": original_zoom,
            "original_shape": original_shape,
            "original_rotation": original_rotation,
            "original_origin": original_origin,
            "preprocessed_orientation": str(preprocessed_orientation),
            "preprocessed_zoom": preprocessed_zoom,
            "preprocessed_rotation": preprocessed_rotation,
            "preprocessed_origin": preprocessed_origin,
            "preprocessed_shape": preprocessed_shape,
            "subject": subject,
            "vert_path": vert_path,
            "input_data_path": input_data_path,
        }


def safe_collate(batch) -> dict | None:
    """Collate a batch, dropping samples the dataset returned as None.

    Args:
        batch: Samples from the dataset, some of which may be None.

    Returns:
        The collated batch, or None if every sample was dropped.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def _load_ct(container, split_info) -> NII:
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")
    if split_info is not None:
        ct_query.filter("split", split_info)
    return ct_query.candidates[0].open_nii()


def preprocess_segmentation_masks(  # noqa: ANN201
    subject,
    vert_msk: NII,
    input_data: NII | None,
    input_data_type: str,
    vert_list,
    zoom=(1, 1, 1),
    tmp_name: str = "tmp",
):
    """Preprocess segmentation masks and create a master dataframe."""
    print(f"preprocessing subject: {subject}")
    print("preprocess_seg - zoom: ", zoom)

    original_orientation = vert_msk.orientation
    original_zoom = vert_msk.zoom
    original_shape = vert_msk.shape
    original_rotation = vert_msk.rotation
    original_origin = vert_msk.origin

    print(
        "Original msk meta",
        original_orientation,
        original_zoom,
        original_shape,
        original_rotation,
        original_origin,
    )

    temp_dir = str(get_path("tmp_root") / tmp_name) + "/"
    os.makedirs(os.path.join(temp_dir, subject), exist_ok=True)

    msk_vert_list = vert_msk.unique()
    vertebrae = sorted([v for v in vert_list if v in msk_vert_list], key=lambda x: v_idx_order.index(x))

    vert_msk = vert_msk.reorient(("L", "A", "S"), verbose=True).map_labels_({50: 49}, verbose=False)
    original_zoom = vert_msk.zoom
    original_shape = vert_msk.shape
    if input_data is not None:
        input_data.reorient_(("L", "A", "S"))

    cutout_info = []
    first = True
    for vert in vertebrae:
        vert_msk.rescale_(zoom, verbose=first)
        if input_data is not None:
            input_data.resample_from_to_(vert_msk, verbose=False)

        com = np_utils.np_center_of_mass(vert_msk.extract_label(vert).get_seg_array())[1]

        input_data_path = os.path.join(temp_dir, subject, f"vert_{vert}-input.nii.gz")
        vert_path = os.path.join(temp_dir, subject, f"vert_{vert}-vertseg.nii.gz")
        surface_path = os.path.join(temp_dir, subject, f"vert_{vert}-surface.nii.gz")

        _, crop, padding = np_utils.np_calc_crop_around_centerpoint(
            (round(com[0]), round(com[1]), round(com[2])),
            vert_msk.get_seg_array(),
            cutout_size=(128, 128, 144),
        )

        if input_data is not None:
            input_data_cropped = input_data.apply_crop(crop).apply_pad(padding, verbose=False)

        vert_cropped = vert_msk.apply_crop(crop).apply_pad(padding, verbose=False)
        print("Cropped from shape", vert_msk.shape, "to", vert_cropped.shape) if first else None
        x_min, x_max, y_min, y_max, z_min, z_max = crop[0].start, crop[0].stop, crop[1].start, crop[1].stop, crop[2].start, crop[2].stop
        x_min -= padding[0][0]
        y_min -= padding[1][0]
        z_min -= padding[2][0]

        vert_cropped.save(vert_path, verbose=False)
        surface = vert_cropped.compute_surface_mask(connectivity=3)
        surface.save(surface_path, verbose=False)
        if input_data is not None:
            input_data_cropped.save(input_data_path, verbose=False)

        if input_data_type == "vertseg":
            input_data_path = vert_path
        elif input_data_type == "surface_msk":
            input_data_path = surface_path

        preprocessed_origin = vert_cropped.origin
        preprocessed_rotation = vert_cropped.rotation
        preprocessed_orientation = vert_cropped.orientation
        preprocessed_zoom = vert_cropped.zoom
        preprocessed_shape = vert_cropped.shape

        (
            print(
                "Preprocessed msk meta",
                preprocessed_orientation,
                preprocessed_zoom,
                preprocessed_shape,
                preprocessed_rotation,
                preprocessed_origin,
            )
            if first
            else None
        )
        first = False

        cutout_info.append(
            {
                "subject": subject,
                "vert": vert,
                "x_min": int(x_min),
                "x_max": int(x_max),
                "y_min": int(y_min),
                "y_max": int(y_max),
                "z_min": int(z_min),
                "z_max": int(z_max),
                "vert_path": vert_path,
                "input_data_path": input_data_path,
                "surface_path": surface_path,
                "preprocessed_orientation": preprocessed_orientation,
                "preprocessed_zoom": preprocessed_zoom,
                "preprocessed_rotation": preprocessed_rotation,
                "preprocessed_origin": preprocessed_origin,
                "preprocessed_shape": preprocessed_shape,
                "original_orientation": original_orientation,
                "original_zoom": original_zoom,
                "original_shape": original_shape,
                "original_rotation": original_rotation,
                "original_origin": original_origin,
            }
        )

    master_df = pd.DataFrame(cutout_info)
    master_df_path = os.path.join(temp_dir, subject, "cutout_df.csv")
    master_df.to_csv(master_df_path, index=False)

    return master_df, temp_dir


def predict_subject_pois(  # noqa: ANN201
    subject,
    vert_msk_ref: BIDS_FILE,
    dm_path,
    model_path,
    vert_out,
    poi_out,
    poi_global_out,
    container=None,
    project_to_surface=False,
    inference_flipped=False,
    vert_msk_override: NII | None = None,
    input_data_override: NII | None = None,
):
    """Run the model over one subject and write its predicted POI file.

    Distinct from :func:`verpex.evaluation.metrics.create_prediction_poi_files`,
    which runs a trained model over a whole dataset split described by a saved data
    module. This one starts from a single subject's segmentation masks.

    The ``*_override`` arguments let a caller supply masks that are not on disk, for
    datasets whose masks have to be built at read time. When they are None,
    everything is read through ``vert_msk_ref`` and the BIDS container.
    """
    with open(dm_path, encoding="utf-8") as handle:
        dm_params = json.load(handle)
    print(dm_params)
    input_shape = dm_params["input_shape"]
    input_data_type = dm_params["input_data_type"]
    vert_list = sorted(dm_params["include_vert_list"] + [6, 7], key=lambda x: v_idx_order.index(x))
    poi_indices = dm_params["include_poi_list"]
    zoom = dm_params.get("zoom", (1, 1, 1))
    print("zoom: ", zoom)

    vert_msk = vert_msk_ref.open_nii() if vert_msk_override is None else vert_msk_override
    vert_list = [v for v in vert_list if v in vert_msk.unique()]
    split_info = vert_msk_ref.info.get("split")

    def _from_container(kind: str) -> NII:
        if input_data_override is not None:
            return input_data_override
        if container is None:
            raise ValueError(f"container must be provided when input_data_type='{kind}'")
        return get_subreg(container, split=split_info) if kind == "subreg" else _load_ct(container, split_info)

    if input_data_type == "vertseg":
        input_data = vert_msk
    elif input_data_type in ("subreg", "ct", "surface_msk+ct"):
        input_data = _from_container(input_data_type)
    elif input_data_type == "surface_msk":
        input_data = vert_msk.compute_surface_mask(connectivity=3, dilated_surface=False)
    else:
        raise ValueError(f"Unknown input data type: {input_data_type}")

    if vert_msk is None or input_data is None:
        print(f"Skip Subject: {subject} - not all data available")
        return

    if vert_msk.shape != input_data.shape:
        print(f"Skip Subject: {subject} - vertseg {vert_msk.shape} and input_data {input_data.shape} shapes don't match")
        return

    master_df, temp_dir = preprocess_segmentation_masks(
        subject,
        vert_msk,
        input_data,
        input_data_type,
        vert_list,
        zoom,
        tmp_name="tmp_" + vert_out.name,
    )

    print(f"inferencing subject: {subject}")
    use_neighbor = dm_params["dataset"] in NEIGHBOR_DATASETS
    ds = SpineInferenceDataset(
        master_df,
        input_shape=input_shape,
        input_data_type=input_data_type,
        include_vert_list=vert_list,
        zoom=zoom,
        poi_indices=poi_indices,
        include_neighbors=use_neighbor,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=safe_collate)

    model = ev.load_model_from_checkpoint(model_path)

    partial_centroids = []
    first = True
    for batch in dl:
        if batch is None:
            continue

        batch = batch_to_device(batch, model.device)
        if inference_flipped:
            flip_axis = input_data.get_axis("L") + 2
            batch["input"] = torch.flip(batch["input"], dims=[flip_axis])

        batch = model(batch)

        refined_preds_batch = batch["refined_preds"]
        if inference_flipped:
            backflip_axis = flip_axis - 2
            refined_preds_batch[:, :, backflip_axis] = batch["input"].shape[flip_axis] - refined_preds_batch[:, :, backflip_axis]

        if project_to_surface:
            refined_preds_projected_batch, _ = surface_project_coords(refined_preds_batch, batch["surface"], requires_filling=True)
            pred_coords = refined_preds_projected_batch.squeeze().detach().cpu().numpy()
        else:
            pred_coords = refined_preds_batch.squeeze().detach().cpu().numpy()

        padding_offset = batch["padding_offset"].squeeze().detach().cpu().numpy()
        vertebra = batch["vertebra"].squeeze().detach().cpu().numpy()
        poi_indices_batch = batch["poi_indices"].squeeze().detach().cpu().numpy()
        cutout_offset = batch["cutout_offset"].squeeze().detach().cpu().numpy()
        subject = batch["subject"][0]
        vert_path = batch["vert_path"][0]
        input_data_path = batch["input_data_path"][0]

        pre_meta = extract_space_meta(batch, "preprocessed")
        orig_meta = extract_space_meta(batch, "original")

        n_poi = len(DEFAULT_POI_INDICES)

        def _make_ctd(coords, v, idx, meta=pre_meta, offset=padding_offset):
            # meta/offset are bound as defaults so the closure captures this
            # iteration's values rather than the loop variable.
            return ev.np_to_ctd(
                coords,
                vertebra=v,
                origin=meta.origin,
                rotation=meta.rotation,
                idx_list=idx,
                shape=meta.shape,
                zoom=meta.zoom,
                offset=offset,
                orientation=meta.orientation,
            )

        print("np_to_ctd input-zoom: ", pre_meta.zoom)
        unpadded_refined_preds_ctd: POI = _make_ctd(pred_coords[:n_poi], vertebra.item(), poi_indices_batch[:n_poi])
        if use_neighbor:
            unpadded_cutout_poi = unpadded_refined_preds_ctd.join_left(
                _make_ctd(pred_coords[n_poi : 2 * n_poi], vertebra.item() - 1, poi_indices_batch[n_poi : 2 * n_poi])
            ).join_left(_make_ctd(pred_coords[2 * n_poi :], vertebra.item() + 1, poi_indices_batch[2 * n_poi :]))
        else:
            unpadded_cutout_poi = unpadded_refined_preds_ctd
        print("unpadded_refined_preds_ctd: ", unpadded_refined_preds_ctd) if first else None

        if inference_flipped:
            unpadded_cutout_poi.map_labels_(label_map_subregion=poi_flip_pairs)
            if use_neighbor:
                unpadded_refined_preds_ctd.map_labels_(label_map_subregion=poi_flip_pairs)

        subject_dir: Path = vert_out.parent.joinpath("cutouts-preproccessed")
        subject_dir.mkdir(parents=True, exist_ok=True)

        ctd_save_path = os.path.join(subject_dir, str(subject) + "_" + str(vertebra) + "_pred.json")
        ctd_global_save_path = ctd_save_path.replace("_pred.json", "_pred_global.json")

        unpadded_cutout_poi.save(ctd_save_path, verbose=True)
        unpadded_refined_preds_ctd_poi = POI.load(ctd_save_path)
        unpadded_refined_preds_ctd_poi.to_global().save_mrk(ctd_global_save_path)

        vertseg_save_path = ctd_save_path.replace("_pred.json", "_vertseg.nii.gz")
        input_data_save_path = ctd_save_path.replace("_pred.json", f"_{input_data_type}.nii.gz")

        if os.path.exists(vert_path):
            shutil.copy(vert_path, vertseg_save_path)
        else:
            print(f"⚠️ Segmentation file not found: {vert_path}")

        if os.path.exists(input_data_path):
            shutil.copy(input_data_path, input_data_save_path)
        else:
            print(f"⚠️ Segmentation file not found: {input_data_path}")

        revert_poi_to_original_space(unpadded_refined_preds_ctd, cutout_offset, orig_meta)
        first = False

        partial_centroids.append(
            {
                "subject": subject,
                "original_shape": unpadded_refined_preds_ctd.shape,
                "original_zoom": unpadded_refined_preds_ctd.zoom,
                "original_orientation": unpadded_refined_preds_ctd.orientation,
                "original_rotation": orig_meta.rotation,
                "original_origin": orig_meta.origin,
                "centroids": unpadded_refined_preds_ctd.extract_region(vertebra).centroids,
            }
        )

    _sub, pois = combine_centroids(partial_centroids)

    pois.save(poi_out)
    pois.to_global().save_mrk(poi_global_out, split_by_region=True)
    vert_msk.save(vert_out)

    os.system(f"rm -r {temp_dir}")


def resolve_run_paths(checkpoint: str) -> tuple[str, str, str]:
    """Locate everything a checkpoint needs to be run, from its path alone.

    A training run is laid out as
    ``<model_dir>/version_<n>/checkpoints/<name>.ckpt`` with the data module
    configuration saved beside the checkpoint directory, so both the config and a
    name for the output derivative can be derived from the checkpoint path.

    Args:
        checkpoint: Path to a ``.ckpt`` file. A relative path is resolved against
            the configured ``model_root``; the ``.ckpt`` suffix may be omitted.

    Returns:
        The checkpoint path, the path to its ``data_module_params.json``, and a
        short label identifying the run, for naming the output derivative.

    Raises:
        FileNotFoundError: If the checkpoint, or its data module configuration,
            cannot be found.
    """
    path = Path(checkpoint)
    if not path.is_absolute():
        path = get_path("model_root") / path
    if path.suffix != ".ckpt":
        path = path.with_suffix(".ckpt")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Walk up from the checkpoint, the way verpex-eval does, so a run saved with a
    # different directory depth still resolves.
    for candidate in path.parents:
        params = candidate / "data_module_params.json"
        if params.exists():
            run_dir = candidate
            break
    else:
        raise FileNotFoundError(f"No data_module_params.json found above {path}")

    label = f"{run_dir.parent.name}-{run_dir.name}"
    return str(path), str(run_dir / "data_module_params.json"), label


@dataclass
class InferenceConfig(Class_to_ArgParse):
    """Command-line configuration for :func:`main`."""

    datasets: list[str] = field(
        default_factory=lambda: [
            "dataset-verse19training_1mmiso",
            "dataset-verse20training_1mmiso",
            "dataset-verse19validation_1mmiso",
            "dataset-verse20validation_1mmiso",
            "dataset-verse19test_1mmiso",
            "dataset-verse20test_1mmiso",
        ]
    )
    der_msk: str = "derivatives_combined"
    der_out_base: str = "derivatives_poi_"
    #: Checkpoints to run, as paths to ``.ckpt`` files. Relative paths are resolved
    #: against the configured ``model_root``. Each run's ``data_module_params.json``
    #: is located by walking up from its checkpoint.
    checkpoints: list[str] = field(default_factory=list)
    inference_flipped: bool = False
    project_to_surface: bool = False


def main() -> None:
    """Run POI inference over one or more BIDS datasets."""
    opt = InferenceConfig().get_opt()
    ds_names = opt.datasets
    DER_MSK = opt.der_msk

    checkpoints = opt.checkpoints

    if not isinstance(checkpoints, list):
        checkpoints = [checkpoints]

    if not checkpoints:
        raise SystemExit("No checkpoints given. Pass --checkpoints PATH [PATH ...].")

    for checkpoint in checkpoints:
        model_path, dm_path, run_label = resolve_run_paths(checkpoint)

        project_to_surface = opt.project_to_surface

        for ds_name in ds_names:
            ds_path = str(get_path("data_root") / ds_name) + "/"
            bgi = BIDS_Global_info(
                datasets=[ds_path],
                parents=[DER_MSK, "rawdata"],
            )

            der_out = f"{opt.der_out_base}{run_label}"
            if opt.inference_flipped:
                der_out += "_flipped"
            if project_to_surface:
                der_out += "_sproj"

            for sub, container in bgi.enumerate_subjects():
                print(f"Subject: {sub}")

                vert_msk_refs: BIDS_FILE = get_vertseg_bfile(container)
                for vert_msk_ref in vert_msk_refs:
                    if not vert_msk_ref.exists():
                        continue

                    subreg_msk_p = vert_msk_ref.get_changed_path(info={"seg": "subreg"}, parent=vert_msk_ref.parent)
                    if not subreg_msk_p.exists():
                        print(f"Skip Subject: {sub} - subreg file not found for vertseg {vert_msk_ref.path}")
                        continue

                    if vert_msk_ref is None:
                        print(f"Skip Subject: {sub} - not all data available")
                        continue

                    vert_out = vert_msk_ref.get_changed_path(info={"seg": "vert"}, parent=der_out)
                    poi_out = vert_msk_ref.get_changed_path(
                        bids_format="poi", info={"mod": "ct", "seg": "vert"}, file_type="json", parent=der_out
                    )
                    poi_global_out = vert_msk_ref.get_changed_path(
                        bids_format="poi", info={"mod": "ct", "seg": "vert", "space": "global"}, file_type="mrk.json", parent=der_out
                    )

                    if poi_global_out.exists() and poi_out.exists():
                        print(f"Skip Subject: {sub} - output POI files already exist")
                        continue

                    try:
                        vert_msk = vert_msk_ref.open_nii()
                        subreg_msk = NII.load(subreg_msk_p, seg=True)
                    except Exception as e:
                        print(f"Error opening vertseg: {e!s}")
                        vert_msk = None

                    if vert_msk is None:
                        print(f"Skip Subject: {sub} - not all data available")
                        continue

                    vert_msk.assert_affine(other=subreg_msk)

                    predict_subject_pois(
                        sub,
                        vert_msk_ref,
                        dm_path,
                        model_path,
                        vert_out,
                        poi_out,
                        poi_global_out,
                        container=container,
                        project_to_surface=project_to_surface,
                        inference_flipped=opt.inference_flipped,
                    )


if __name__ == "__main__":
    main()
