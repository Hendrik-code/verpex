"""Datasets serving per-vertebra cutouts and their landmark annotations."""

import ast
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# from BIDS import NII, POI
from TPTBox import NII
from TPTBox.core.poi import POI

from verpex.data.dataloading import compute_surface, get_gt_pois, pad_array_to_shape
from verpex.data.transforms import (  # was src.transforms.transforms
    Compose,
    LandMarksRandHorizontalFlip,
    LandMarksRandHorizontalFlipNeighbor,
)
from verpex.paths import get_path


def resolve_cutout_dir(file_dir: str) -> str:
    """Resolve a ``master_df`` cutout directory to an absolute path.

    ``prepare_data`` writes ``file_dir`` relative to the cutout root, so the same
    ``master_df.csv`` works on any machine. Absolute entries (as written by older
    versions of ``prepare_data``) are passed through unchanged.

    Args:
        file_dir: The ``file_dir`` value from a ``master_df`` row.

    Returns:
        An absolute path to the cutout directory.
    """
    path = Path(file_dir)
    if path.is_absolute():
        return str(path)
    return str(get_path("cutout_root") / path)


class PoiDataset(Dataset):
    """Base dataset: one item per vertebra cutout listed in ``master_df``.

    Loads the cutout's image and masks, extracts the ground-truth landmarks, and
    builds the loss mask that excludes landmarks flagged as bad annotations.
    """

    def __init__(
        self,
        master_df,
        poi_indices,
        include_vert_list,
        poi_flip_pairs=None,
        input_data_type="subreg",
        input_shape=(128, 128, 96),
        zoom=(1, 1, 1),
        transforms=None,
        flip_prob=0.5,
        include_com=False,
        poi_file_ending="poi.json",
        iterations=1,
        show_neighbors=False,
        neighbor_drop_prob=0.05,
    ):
        # If master_df has a column use_sample, filter on it
        if "use_sample" in master_df.columns:
            master_df = master_df[master_df["use_sample"]]
        self.master_df = master_df
        self.input_data_type = input_data_type
        self.input_shape = input_shape
        self.zoom = zoom
        self.poi_indices = poi_indices
        self.transform = Compose(transforms) if transforms else None
        if flip_prob > 0:
            self.transform = Compose([self.transform, LandMarksRandHorizontalFlip(flip_prob, poi_flip_pairs)])
        self.include_com = include_com
        self.poi_flip_pairs = poi_flip_pairs
        self.flip_prob = flip_prob
        self.poi_file_ending = poi_file_ending
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.vert_idx_to_list_idx = {vert: idx for idx, vert in enumerate(include_vert_list)}
        self.iterations = iterations
        self.show_neighbors = show_neighbors
        self.neighbor_drop_prob = neighbor_drop_prob

    def __len__(self):
        return len(self.master_df)

    def preprocess_nifti(self, nii_path, is_img=False, verbose=False):  # noqa: ANN201
        """Load a cutout NIfTI and bring it to the dataset's orientation and spacing.

        Args:
            nii_path: Path to the cutout file.
            is_img: True for intensity images, False for segmentation masks, which
                selects nearest-neighbour rather than linear resampling.
            verbose: Print the path being loaded.

        Returns:
            The loaded and resampled array.
        """
        if verbose:
            print(f"Loading from {nii_path}")
        nii = NII.load(nii_path, seg=not is_img)
        nii.rescale_and_reorient_(axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False)
        if is_img:
            nii = nii.normalize_ct(min_out=0, max_out=1, inplace=True)
        array = nii.get_array()

        array, offset = pad_array_to_shape(array, self.input_shape)

        tensor = torch.from_numpy(array.astype(float)).unsqueeze(0)  # Add channel dim

        return tensor, offset

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        subject = row["subject"]
        vertebra = row["vertebra"]
        file_dir = resolve_cutout_dir(row["file_dir"])

        # If the master_dir has a column bad_poi_list, use this to create a loss mask
        if "bad_poi_list" in self.master_df.columns:
            bad_poi_list = ast.literal_eval(row["bad_poi_list"])
            bad_poi_list = [int(poi) for poi in bad_poi_list]
            bad_poi_list = torch.tensor(bad_poi_list)
        else:
            bad_poi_list = torch.tensor([], dtype=torch.int)

        # Get the paths
        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "vertseg.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        surface_msk_path = os.path.join(file_dir, "surface_msk.nii.gz")
        poi_path = os.path.join(file_dir, self.poi_file_ending)

        # Get the ground truth POIs
        poi = POI.load(poi_path)
        poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(self.zoom, verbose=False)
        poi, missing_pois = get_gt_pois(poi, vertebra, self.poi_indices)

        poi_indices = torch.tensor(self.poi_indices)

        # Load Data
        # ct, offset = self.preprocess_nifti(ct_path, is_img=True)
        subreg, offset = self.preprocess_nifti(subreg_path, is_img=False)
        vertseg, _ = self.preprocess_nifti(msk_path, is_img=False)

        if self.show_neighbors:
            vertseg_label = vertseg.unique()
            # Define neighbor vertebrae
            neighbor_top = vertebra - 1 if vertebra > 1 and vertebra - 1 in vertseg_label else 0  # 0 = dummy (no top/bottom neighbor)
            neighbor_bottom = vertebra + 1 if vertebra < 24 and vertebra + 1 in vertseg_label else 0

            # AUGMENTATION: Random labeldrop of neighbor vertebrae for data augmentation
            if torch.rand(1).item() < self.neighbor_drop_prob:
                if torch.rand(1).item() < 0.5:
                    # top
                    neighbor_top = 0
                else:
                    # bottom
                    neighbor_bottom = 0

            all_vert = [("current", vertebra), ("top", neighbor_top), ("bottom", neighbor_bottom)]

            # Filter out dummy for seg mask
            actual_vert = [vert for _, vert in all_vert if vert != 0]

            mask = np.isin(vertseg, actual_vert)
        else:
            mask = vertseg == vertebra
        # surface_msk, _ = self.preprocess_nifti(surface_msk_path, is_img=False)

        if self.input_data_type == "vertseg":
            data_dict["input"] = vertseg * mask
        elif self.input_data_type == "subreg":
            data_dict["input"] = subreg * mask
        elif self.input_data_type == "ct":
            ct, _ = self.preprocess_nifti(ct_path, is_img=True)
            data_dict["input"] = ct * mask
        elif self.input_data_type == "ct_raw":
            ct, _ = self.preprocess_nifti(ct_path, is_img=True)
            data_dict["input"] = ct
        elif self.input_data_type == "surface_msk":
            surface_msk, _ = self.preprocess_nifti(
                surface_msk_path,
                is_img=False,
                verbose=False,
            )
            data_dict["input"] = surface_msk * mask
        elif self.input_data_type == "surface_msk+ct":
            # 2-channel input: surface mask + normalised CT, both masked to the
            # current vertebra. Gives the coarse encoder anatomical context
            # (cortical bone / disc space) on top of the pure surface geometry.
            surface_msk, _ = self.preprocess_nifti(
                surface_msk_path,
                is_img=False,
                verbose=False,
            )
            ct, _ = self.preprocess_nifti(ct_path, is_img=True)
            data_dict["input"] = torch.cat([surface_msk * mask, ct * mask], dim=0)  # (2, H, W, D)
        else:
            raise ValueError(f"Unknown input data type: {self.input_data_type}")

        # Load the BIDS objects
        # ct = NII.load(ct_path, seg = False)
        # subreg = NII.load(subreg_path, seg=True)
        # vertseg = NII.load(msk_path, seg=True)
        # surface_msk = NII.load(surface_msk_path, seg=True)
        # poi = POI.load(poi_path)
        #
        # ct.rescale_and_reorient_(
        #    axcodes_to=('L', 'A', 'S'), voxel_spacing = self.zoom, verbose = False
        # )
        # subreg.rescale_and_reorient_(
        #    axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False
        # )
        # vertseg.rescale_and_reorient_(
        #    axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False
        # )
        # surface_msk.rescale_and_reorient_(
        #    axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False
        # )
        #

        # ct.normalize_ct(min_out=0, max_out=1, inplace=True)

        # Get arrays
        # ct = ct.get_array()
        # subreg = subreg.get_array()
        # vertseg = vertseg.get_array()
        # surface_msk = surface_msk.get_array()

        # mask = vertseg == vertebra

        # ct_msk = ct * mask
        # subreg = subreg * mask
        # vertseg = vertseg * mask
        # surface_msk = surface_msk * mask
        #
        # subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        # vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)
        # surface_msk, _ = pad_array_to_shape(surface_msk, self.input_shape)
        # ct, _ = pad_array_to_shape(ct, self.input_shape)
        # ct_msk, _ = pad_array_to_shape(ct_msk, self.input_shape)[0]
        #
        poi = poi + torch.tensor(offset)
        #
        ## Convert subreg, vertseg and surface_msk to tensors
        # ct = torch.from_numpy(ct.astype(float))
        # ct_msk = torch.from_numpy(ct_msk.astype(float))
        # subreg = torch.from_numpy(subreg.astype(float))
        # vertseg = torch.from_numpy(vertseg.astype(float))
        # surface_msk = torch.from_numpy(surface_msk.astype(float))
        #
        ## Add channel dimension
        # ct = ct.unsqueeze(0)
        # ct_msk = ct_msk.unsqueeze(0)
        # subreg = subreg.unsqueeze(0)
        # vertseg = vertseg.unsqueeze(0)
        # surface_msk = surface_msk.unsqueeze(0)

        # data_dict["input"] = vertseg#subreg
        data_dict["target"] = poi
        data_dict["target_indices"] = poi_indices

        data_dict = self.transform(data_dict) if self.transform else data_dict

        # Identify pois outside of the input shape
        max_x = self.input_shape[0] - 1
        max_y = self.input_shape[1] - 1
        max_z = self.input_shape[2] - 1

        outside_poi_indices = (
            (data_dict["target"][:, 0] < 0)
            | (data_dict["target"][:, 0] > max_x)
            | (data_dict["target"][:, 1] < 0)
            | (data_dict["target"][:, 1] > max_y)
            | (data_dict["target"][:, 2] < 0)
            | (data_dict["target"][:, 2] > max_z)
        )

        # Create a loss mask for pois shifted oustide of the image due to augmentation,
        # missing pois from the ground truth and bad pois
        loss_mask = torch.ones_like(data_dict["target"][:, 0])
        loss_mask[outside_poi_indices] = 0
        bad_poi_list_idx = [self.poi_idx_to_list_idx[bad_poi.item()] for bad_poi in bad_poi_list if bad_poi.item() in self.poi_indices]
        loss_mask[bad_poi_list_idx] = 0
        missing_poi_list_idx = [self.poi_idx_to_list_idx[missing_poi.item()] for missing_poi in missing_pois]
        loss_mask[missing_poi_list_idx] = 0

        data_dict["loss_mask"] = loss_mask.bool()

        transformed_mask = mask
        if self.show_neighbors:
            transformed_mask *= vertseg == vertebra
        surface = compute_surface(transformed_mask, iterations=self.iterations)

        data_dict["surface"] = surface
        data_dict["subject"] = str(subject)
        data_dict["vertebra"] = vertebra
        data_dict["zoom"] = torch.tensor(self.zoom).float()
        data_dict["offset"] = torch.tensor(offset).float()
        data_dict["ct_path"] = ct_path
        data_dict["msk_path"] = msk_path
        data_dict["subreg_path"] = subreg_path
        data_dict["poi_path"] = poi_path
        data_dict["poi_list_idx"] = torch.tensor([self.poi_idx_to_list_idx[poi.item()] for poi in poi_indices])
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])

        return data_dict


class SpineDataset(PoiDataset):
    """Single-vertebra cutout dataset."""

    def __init__(
        self,
        master_df,
        input_data_type="subreg",
        input_shape=(128, 128, 96),
        zoom=(1, 1, 1),
        transforms=None,
        flip_prob=0.5,
        include_com=False,
        include_poi_list=None,
        include_vert_list=None,
        poi_file_ending="poi.json",
        iterations=1,
        show_neighbors=False,
        neighbor_drop_prob=0.05,
    ):
        super().__init__(
            master_df,
            poi_indices=(
                include_poi_list
                or (
                    [
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
                        41,
                        42,
                        43,
                        44,
                        45,
                        46,
                        47,
                        48,
                        49,
                        50,
                    ]
                    if include_com
                    else [
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
                )
            ),
            include_vert_list=(
                include_vert_list
                or [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                ]
            ),
            poi_flip_pairs={
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
                # Center of mass, does not need to be flipped
                # TODO: Is this correct? Does it also hold when include_com=False and the POIs are not defined at all?
                41: 41,
                42: 42,
                43: 43,
                44: 44,
                45: 45,
                46: 46,
                47: 47,
                48: 48,
                49: 49,
                50: 50,
                0: 0,
            },
            input_data_type=input_data_type,
            input_shape=input_shape,
            zoom=zoom,
            transforms=transforms,
            flip_prob=flip_prob,
            include_com=include_com,
            poi_file_ending=poi_file_ending,
            iterations=iterations,
            show_neighbors=show_neighbors,
            neighbor_drop_prob=neighbor_drop_prob,
        )


class PoiNeighborDataset(Dataset):
    """Base dataset that serves each vertebra together with its neighbours."""

    def __init__(
        self,
        master_df,
        poi_indices,
        include_vert_list,
        poi_flip_pairs=None,
        input_data_type="subreg",
        input_shape=(120, 121, 149),
        zoom=(1, 1, 1),
        transforms=None,
        flip_prob=0.0,
        include_com=False,
        poi_file_ending="poi.json",
        iterations=1,
        neighbor_drop_prob=0.05,
    ):
        # If master_df has a column use_sample, filter on it
        if "use_sample" in master_df.columns:
            master_df = master_df[master_df["use_sample"]]
        self.master_df = master_df
        self.input_data_type = input_data_type
        self.input_shape = input_shape
        self.zoom = tuple(zoom)
        self.poi_indices = poi_indices
        self.transform = Compose(transforms) if transforms else None
        if flip_prob > 0:
            self.transform = Compose([self.transform, LandMarksRandHorizontalFlipNeighbor(flip_prob, poi_flip_pairs)])
        self.include_com = include_com
        self.poi_flip_pairs = poi_flip_pairs
        self.flip_prob = flip_prob
        self.poi_file_ending = poi_file_ending
        self.poi_idx_to_list_idx = {poi: idx for idx, poi in enumerate(poi_indices)}
        self.vert_idx_to_list_idx = {vert: idx for idx, vert in enumerate(include_vert_list)}
        self.iterations = iterations
        self.neighbor_drop_prob = neighbor_drop_prob

    def __len__(self):
        return len(self.master_df)

    def _process_single_vert(self, poi, subject, vertebra):
        """Processes a single vertebra and returns its POIs and loss mask."""
        pois, missing_pois = get_gt_pois(poi, vertebra, self.poi_indices)

        loss_mask = torch.ones(len(self.poi_indices), dtype=torch.float)

        missing_poi_list_idx = [self.poi_idx_to_list_idx[missing_poi.item()] for missing_poi in missing_pois]
        loss_mask[missing_poi_list_idx] = 0

        if "bad_poi_list" not in self.master_df.columns or vertebra == 0:
            bad_poi_list = torch.tensor([], dtype=torch.int)

        else:
            vert_row = self.master_df[(self.master_df["subject"] == subject) & (self.master_df["vertebra"] == vertebra)]

            if len(vert_row) == 0:
                bad_poi_list = torch.tensor([], dtype=torch.int)

            else:
                bad_poi_list = ast.literal_eval(vert_row.iloc[0]["bad_poi_list"])
                bad_poi_list = [int(poi) for poi in bad_poi_list]
                bad_poi_list = torch.tensor(bad_poi_list)

        bad_poi_list_idx = [self.poi_idx_to_list_idx[bad_poi.item()] for bad_poi in bad_poi_list if bad_poi.item() in self.poi_indices]
        loss_mask[bad_poi_list_idx] = 0

        return pois, loss_mask

    def preprocess_nifti(self, nii_path, is_img=False, verbose=False):  # noqa: ANN201
        """Load a cutout NIfTI and bring it to the dataset's orientation and spacing.

        Args:
            nii_path: Path to the cutout file.
            is_img: True for intensity images, False for segmentation masks, which
                selects nearest-neighbour rather than linear resampling.
            verbose: Print the path being loaded.

        Returns:
            The loaded and resampled array.
        """
        if verbose:
            print(f"Loading from {nii_path}")
        nii = NII.load(nii_path, seg=not is_img)
        nii.rescale_and_reorient_(axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False)
        if is_img:
            nii = nii.normalize_ct(min_out=0, max_out=1, inplace=True)
        array = nii.get_array()

        array, offset = pad_array_to_shape(array, self.input_shape)

        tensor = torch.from_numpy(array.astype(float)).unsqueeze(0)  # Add channel dim

        return tensor, offset

    def __getitem__(self, index):
        data_dict = {}

        # Read the row from the master dataframe
        row = self.master_df.iloc[index]
        subject = row["subject"]
        vertebra = row["vertebra"]
        file_dir = resolve_cutout_dir(row["file_dir"])

        # Get the paths
        ct_path = os.path.join(file_dir, "ct.nii.gz")
        msk_path = os.path.join(file_dir, "vertseg.nii.gz")
        subreg_path = os.path.join(file_dir, "subreg.nii.gz")
        surface_msk_path = os.path.join(file_dir, "surface_msk.nii.gz")
        poi_path = os.path.join(file_dir, self.poi_file_ending)

        # Load the BIDS objects
        ct = NII.load(ct_path, seg=False)
        subreg = NII.load(subreg_path, seg=True)
        vertseg = NII.load(msk_path, seg=True)
        # surface_msk = NII.load(surface_msk_path, seg=True)
        poi = POI.load(poi_path)
        poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(self.zoom, verbose=False)
        # poi, missing_pois = get_gt_pois(poi, vertebra, self.poi_indices)

        # zoom = (1, 1, 1)

        subreg, offset = self.preprocess_nifti(subreg_path, is_img=False)
        vertseg, _ = self.preprocess_nifti(msk_path, is_img=False)

        # ct.rescale_and_reorient_(
        #    axcodes_to=('L', 'A', 'S'), voxel_spacing = self.zoom, verbose = False
        # )
        # subreg.rescale_and_reorient_(
        #    axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False
        # )
        # vertseg.rescale_and_reorient_(
        #    axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False
        # )
        # surface_msk.rescale_and_reorient_(
        #    axcodes_to=("L", "A", "S"), voxel_spacing=self.zoom, verbose=False
        # )
        # poi.reorient_(axcodes_to=("L", "A", "S"), verbose=False).rescale_(
        #    self.zoom, verbose=False
        # )

        # ct.normalize_ct(min_out=0, max_out=1, inplace=True)
        vertseg_label = vertseg.unique()

        # Define neighbor vertebrae
        neighbor_top = vertebra - 1 if vertebra > 1 and vertebra - 1 in vertseg_label else 0  # 0 = dummy (no top/bottom neighbor)
        neighbor_bottom = vertebra + 1 if vertebra < 24 and vertebra + 1 in vertseg_label else 0

        # AUGMENTATION: Random labeldrop of neighbor vertebrae for data augmentation
        if torch.rand(1).item() < self.neighbor_drop_prob:
            if torch.rand(1).item() < 0.5:
                # top
                neighbor_top = 0
            else:
                # bottom
                neighbor_bottom = 0

        all_vert = [("current", vertebra), ("top", neighbor_top), ("bottom", neighbor_bottom)]

        # Filter out dummy for seg mask
        actual_vert = [vert for _, vert in all_vert if vert != 0]

        # Get arrays
        # ct = ct.get_array()
        # subreg = subreg.get_array()
        # vertseg = vertseg.get_array()
        # surface_msk = surface_msk.get_array()

        mask = np.isin(vertseg, actual_vert)

        # mask vert with neighbors
        # ct = ct * mask
        # subreg = subreg * mask
        # vertseg = vertseg * mask
        # surface_msk = surface_msk * mask

        if self.input_data_type == "vertseg":
            data_dict["input"] = vertseg * mask
        elif self.input_data_type == "subreg":
            data_dict["input"] = subreg * mask
        elif self.input_data_type == "ct":
            ct, _ = self.preprocess_nifti(ct_path, is_img=True)
            data_dict["input"] = ct * mask
        elif self.input_data_type == "ct_raw":
            ct, _ = self.preprocess_nifti(ct_path, is_img=True)
            data_dict["input"] = ct
        elif self.input_data_type == "surface_msk":
            surface_msk, _ = self.preprocess_nifti(
                surface_msk_path,
                is_img=False,
                verbose=False,
            )
            data_dict["input"] = surface_msk * mask
        else:
            raise ValueError(f"Unknown input data type: {self.input_data_type}")

        # Padding and offset
        # subreg, offset = pad_array_to_shape(subreg, self.input_shape)
        # vertseg, _ = pad_array_to_shape(vertseg, self.input_shape)
        # surface_msk, _ = pad_array_to_shape(surface_msk, self.input_shape)
        # ct, _ = pad_array_to_shape(ct, self.input_shape)

        # process each vertebra separately
        all_pois = []
        all_loss_masks = []

        for _label, vert in all_vert:
            if vert == 0:  # dummy
                vert_pois = torch.full((len(self.poi_indices), 3), -1)
                vert_loss_mask = torch.zeros(len(self.poi_indices), dtype=torch.float)

            else:  # real
                vert_pois, vert_loss_mask = self._process_single_vert(poi, subject, vert)

            all_pois.append(vert_pois)
            all_loss_masks.append(vert_loss_mask)

        # combine pois and loss masks
        combined_pois = torch.cat(all_pois, dim=0)
        combined_loss_mask = torch.cat(all_loss_masks, dim=0)

        # add global offset
        combined_pois = combined_pois + torch.tensor(offset)

        # Convert subreg, vertseg and surface_msk to tensors
        # ct = torch.from_numpy(ct.astype(float))
        # subreg = torch.from_numpy(subreg.astype(float))
        # vertseg = torch.from_numpy(vertseg.astype(float))
        # surface_msk = torch.from_numpy(surface_msk.astype(float))

        # Add channel dimension
        # ct = ct.unsqueeze(0)
        # subreg = subreg.unsqueeze(0)
        # vertseg = vertseg.unsqueeze(0)
        # surface_msk = surface_msk.unsqueeze(0)

        # if self.input_data_type == "vertseg":
        #    data_dict["input"] = vertseg
        # elif self.input_data_type == "subreg":
        #    data_dict["input"] = subreg
        # elif self.input_data_type == "ct":
        #    data_dict["input"] = ct
        # elif self.input_data_type == "surface_msk":
        #    data_dict["input"] = surface_msk

        data_dict["target"] = combined_pois

        poi_indices = torch.tensor(self.poi_indices)
        repeat_poi_indices = torch.cat([poi_indices for _ in range(3)])
        data_dict["target_indices"] = repeat_poi_indices

        # apply transform
        data_dict = self.transform(data_dict) if self.transform else data_dict

        # Identify pois outside of the input shape after transform
        max_x = self.input_shape[0] - 1
        max_y = self.input_shape[1] - 1
        max_z = self.input_shape[2] - 1

        outside_poi_indices = (
            (data_dict["target"][:, 0] < 0)
            | (data_dict["target"][:, 0] > max_x)
            | (data_dict["target"][:, 1] < 0)
            | (data_dict["target"][:, 1] > max_y)
            | (data_dict["target"][:, 2] < 0)
            | (data_dict["target"][:, 2] > max_z)
        )

        combined_loss_mask[outside_poi_indices] = 0

        data_dict["loss_mask"] = combined_loss_mask.bool()

        transformed_mask = torch.from_numpy(mask)
        surface = compute_surface(transformed_mask, iterations=self.iterations)

        data_dict["surface"] = surface
        data_dict["subject"] = str(subject)
        data_dict["vertebra"] = vertebra
        data_dict["zoom"] = torch.tensor(self.zoom).float()
        data_dict["offset"] = torch.tensor(offset).float()
        data_dict["ct_path"] = ct_path
        data_dict["msk_path"] = msk_path
        data_dict["subreg_path"] = subreg_path
        data_dict["poi_path"] = poi_path
        data_dict["poi_list_idx"] = torch.tensor([self.poi_idx_to_list_idx[poi.item()] for poi in repeat_poi_indices])
        data_dict["vert_list_idx"] = torch.tensor([self.vert_idx_to_list_idx[vertebra]])

        data_dict["current_vertebra"] = vertebra
        data_dict["n_pois_per_vertebra"] = len(self.poi_indices)
        # Number of concatenated per-vertebra landmark blocks, in the order set by
        # `all_vert` above: block 0 is the current vertebra, the rest are neighbours.
        # PoiNeighborPredictionModule needs this to weight the blocks differently;
        # without it the weighting silently fell back to a flat loss.
        data_dict["n_vertebrae"] = len(all_vert)

        return data_dict


class SpineNeighborDataset(PoiNeighborDataset):
    """Cutout dataset serving each vertebra with its neighbours."""

    def __init__(
        self,
        master_df,
        input_data_type="subreg",
        input_shape=(120, 121, 149),
        zoom=(1, 1, 1),
        transforms=None,
        flip_prob=0.0,
        include_com=False,
        include_poi_list=None,
        include_vert_list=None,
        poi_file_ending="poi.json",
        iterations=1,
        neighbor_drop_prob=0.05,
    ):
        super().__init__(
            master_df,
            poi_indices=(
                include_poi_list
                or (
                    [
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
                        41,
                        42,
                        43,
                        44,
                        45,
                        46,
                        47,
                        48,
                        49,
                        50,
                    ]
                    if include_com
                    else [
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
                )
            ),
            include_vert_list=(
                include_vert_list
                or [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                ]
            ),
            poi_flip_pairs={
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
                # Center of mass, does not need to be flipped
                # TODO: Is this correct? Does it also hold when include_com=False and the POIs are not defined at all?
                41: 41,
                42: 42,
                43: 43,
                44: 44,
                45: 45,
                46: 46,
                47: 47,
                48: 48,
                49: 49,
                50: 50,
                0: 0,
            },
            input_data_type=input_data_type,
            input_shape=input_shape,
            zoom=zoom,
            transforms=transforms,
            flip_prob=flip_prob,
            include_com=include_com,
            poi_file_ending=poi_file_ending,
            iterations=iterations,
            neighbor_drop_prob=neighbor_drop_prob,
        )
