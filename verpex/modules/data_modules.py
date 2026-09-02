"""Lightning data modules: reading master_df.csv and serving vertebra cutouts."""

import copy
import os
import random
from collections.abc import Callable
from functools import partial
from os import PathLike
from typing import TypeVar

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from monai.data.utils import worker_init_fn as _monai_worker_init_fn
from pqdm.processes import pqdm
from torch.utils.data import DataLoader

# from BIDS import BIDS_Global_info
from TPTBox import BIDS_Global_info

from verpex.data.dataloading import (
    get_ct,
    get_spine_poi,
    get_subreg,
    get_vertseg,
    process_container,
)
from verpex.data.dataloading import get_files as _get_files
from verpex.data.dataset import PoiDataset, PoiNeighborDataset, SpineDataset, SpineNeighborDataset
from verpex.data.transforms import create_transform
from verpex.registry import build


def _seed_worker(worker_id):
    """Seed one dataloader worker deterministically.

    Seeds the numpy and random globals from torch's per-worker seed, and every
    MONAI ``Randomizable`` attached to the worker's dataset. RandAffine holds its
    own PRNG state, so without this the augmentation stream diverges between runs
    even when ``pl.seed_everything`` was called.
    """
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    if torch.utils.data.get_worker_info() is not None:
        _monai_worker_init_fn(worker_id)


PoiType = TypeVar("PoiType", bound=PoiDataset)

#: Either dataset family a data module may serve. The two share no base beyond
#: torch's `Dataset`, which declares neither `__len__` nor the augmentation
#: attributes the dataloaders below rely on.
CutoutDataset = PoiDataset | PoiNeighborDataset

#: Value stored in a data module's saved hyperparameters to identify its dataset layout.
SPINE_DATASET = "Spine"
SPINE_NEIGHBOR_DATASET = "SpineNeighbor"
SINGLE_VERTEBRA_DATASETS = (SPINE_DATASET,)
NEIGHBOR_DATASETS = (SPINE_NEIGHBOR_DATASET,)


class POIDataModule(pl.LightningDataModule):
    """Base data module: reads a master_df.csv and serves per-vertebra cutouts.

    Subclasses supply the dataset class and the BIDS file-lookup functions for a
    particular dataset layout.
    """

    #: Populated by `setup()`, from either dataset family - see `CutoutDataset`.
    train_dataset: CutoutDataset
    val_dataset: CutoutDataset
    test_dataset: CutoutDataset

    def __init__(
        self,
        dataset: str,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_data_type: str = "subreg",
        input_shape: tuple = (128, 128, 96),
        zoom: tuple = (1, 1, 1),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_vert_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
        show_neighbors: bool = False,
        neighbor_drop_prob: float = 0.05,
    ):
        super().__init__()
        self.dataset = dataset
        self.master_df_path = master_df
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects
        self.input_data_type = input_data_type
        self.input_shape = input_shape
        self.zoom = zoom
        self.flip_prob = flip_prob
        self.include_com = include_com
        self.include_poi_list = include_poi_list
        self.include_vert_list = include_vert_list
        self.poi_file_ending = poi_file_ending
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_config = transform_config
        self.surface_erosion_iterations = surface_erosion_iterations
        self.show_neighbors = show_neighbors
        self.neighbor_drop_prob = neighbor_drop_prob
        self.save_hyperparameters()

    def build_cutouts(
        self,
        bids_surgery_info: BIDS_Global_info,
        save_path: str,
        get_files: Callable | None = None,
        rescale_zoom: tuple | None = None,
        poi_source: str | None = None,  # noqa: ARG002 - subclasses use it to build `get_files`
    ) -> None:
        """Build the per-vertebra cutouts this data module reads.

        Named ``build_cutouts`` rather than ``prepare_data``: the latter overrides
        PyTorch Lightning's own no-argument ``prepare_data()`` hook, which Lightning
        calls automatically whenever a data module is passed to ``Trainer.fit`` as
        ``datamodule=``. With this signature that call would raise TypeError.

        Note this crops to each vertebra's bounding box plus a margin, which is a
        different strategy from the fixed-size cutouts that ``verpex-prepare-data``
        writes. Prefer the CLI unless you specifically want this behaviour.
        """
        if get_files is None:
            raise ValueError("build_cutouts needs a `get_files` callable; the subclasses supply their own.")

        partial_process_container = partial(
            process_container,
            save_path=save_path,
            rescale_zoom=rescale_zoom,
            get_files=get_files,
        )
        master = pqdm(
            bids_surgery_info.enumerate_subjects(),
            partial_process_container,
            n_jobs=8,
            argument_type="args",
            exception_behaviour="immediate",
        )
        master = [item for sublist in master for item in sublist]
        master_df = pd.DataFrame(master)
        master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)

    def setup(self, stage=None) -> None:  # noqa: ARG002 - Lightning passes it; unused here
        """Load master_df.csv and construct the train, val and test datasets."""
        self.master_df = pd.read_csv(self.master_df_path)
        # empty column bad_poi_list
        # n_rows = len(self.master_df)
        # self.master_df["bad_poi_list"] = None  # Initialize the 'bad_poi_list' column with None
        # self.master_df["bad_poi_list"] = ["[]" for _ in range(n_rows)]  # Initialize the 'bad_poi_list' column with empty lists

        # Only keep rows where vertebra is in "include_vert_list"
        if self.include_vert_list is not None:
            # convert vertebra column to int
            self.master_df["vertebra"] = self.master_df["vertebra"].astype(int)
            self.master_df = self.master_df[self.master_df["vertebra"].isin(self.include_vert_list)]

        all_assigned_subjects = set(self.train_subjects) | set(self.val_subjects) | set(self.test_subjects)
        train_mask = self.master_df["subject"].isin(self.train_subjects) | ~self.master_df["subject"].isin(all_assigned_subjects)
        self.train_df = self.master_df[train_mask]
        self.val_df = self.master_df[self.master_df["subject"].isin(self.val_subjects)]
        self.test_df = self.master_df[self.master_df["subject"].isin(self.test_subjects)]

        # Bound unconditionally: the datasets below always take a `transforms=`
        # argument, and `transform_config=None` is the declared default.
        transform = [create_transform(self.transform_config)] if self.transform_config is not None else None

        if self.dataset in SINGLE_VERTEBRA_DATASETS:
            self.train_dataset = SpineDataset(
                self.train_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=transform,
                flip_prob=self.flip_prob,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                show_neighbors=self.show_neighbors,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.val_dataset = SpineDataset(
                self.val_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                show_neighbors=self.show_neighbors,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.test_dataset = SpineDataset(
                self.test_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                show_neighbors=self.show_neighbors,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )

        elif self.dataset in NEIGHBOR_DATASETS:
            self.train_dataset = SpineNeighborDataset(
                self.train_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=transform,
                flip_prob=self.flip_prob,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.val_dataset = SpineNeighborDataset(
                self.val_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
            self.test_dataset = SpineNeighborDataset(
                self.test_df,
                input_data_type=self.input_data_type,
                input_shape=self.input_shape,
                zoom=self.zoom,
                include_com=self.include_com,
                include_poi_list=self.include_poi_list,
                include_vert_list=self.include_vert_list,
                transforms=None,
                flip_prob=0.0,
                poi_file_ending=self.poi_file_ending,
                iterations=self.surface_erosion_iterations,
                neighbor_drop_prob=self.neighbor_drop_prob,
            )
        print("Setup complete. Dataset sizes:")
        print(f"  Train: {len(self.train_dataset)}")
        print(f"  Validation: {len(self.val_dataset)}")
        print(f"  Test: {len(self.test_dataset)}")

    def _dataloader_generator(self):
        return torch.Generator().manual_seed(42)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader, with augmentation enabled."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
            # collate_fn=custom_collate_fn
        )

    def train_noaug_dataloader(self) -> DataLoader:
        """Return the training dataloader with augmentation disabled."""
        # Shallow-copy first: assigning to `self.train_dataset` would disable
        # augmentation for the real training dataloader too. The composed pipeline
        # lives on `.transform` (singular) with flip_prob already baked into it at
        # construction, so clearing that attribute is what actually turns it off.
        train_2_dataset = copy.copy(self.train_dataset)
        train_2_dataset.flip_prob = 0.0
        train_2_dataset.transform = None
        return DataLoader(
            dataset=train_2_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
            # collate_fn=custom_collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=_seed_worker,
            generator=self._dataloader_generator(),
        )


class SpineDataModule(POIDataModule):
    """Data module for the single-vertebra cutout dataset."""

    def __init__(
        self,
        master_df: PathLike,
        train_subjects: list,
        val_subjects: list,
        test_subjects: list,
        input_data_type: str = "subreg",
        input_shape: tuple = (128, 128, 96),
        zoom: tuple = (1, 1, 1),
        flip_prob: float = 0.5,
        transform_config: dict | None = None,
        include_com: bool = False,
        include_poi_list=None,
        include_vert_list=None,
        batch_size: int = 1,
        num_workers: int = 0,
        poi_file_ending: str = "poi.json",
        surface_erosion_iterations: int = 1,
        show_neighbors: bool = False,
        neighbor_drop_prob: float = 0.05,
    ):
        super().__init__(
            dataset=SPINE_DATASET,
            master_df=master_df,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            input_data_type=input_data_type,
            input_shape=input_shape,
            zoom=zoom,
            flip_prob=flip_prob,
            transform_config=transform_config,
            include_com=include_com,
            include_poi_list=include_poi_list,
            include_vert_list=include_vert_list,
            batch_size=batch_size,
            num_workers=num_workers,
            poi_file_ending=poi_file_ending,
            surface_erosion_iterations=surface_erosion_iterations,
            show_neighbors=show_neighbors,
            neighbor_drop_prob=neighbor_drop_prob,
        )

    def build_cutouts(
        self,
        bids_surgery_info: BIDS_Global_info,
        save_path: str,
        get_files: Callable | None = None,
        rescale_zoom: tuple | None = None,
        poi_source: str | None = None,
    ) -> None:
        """Build the per-vertebra cutouts this data module reads.

        Named ``build_cutouts`` rather than ``prepare_data``: the latter overrides
        PyTorch Lightning's own no-argument ``prepare_data()`` hook, which Lightning
        calls automatically whenever a data module is passed to ``Trainer.fit`` as
        ``datamodule=``. With this signature that call would raise TypeError.

        Note this crops to each vertebra's bounding box plus a margin, which is a
        different strategy from the fixed-size cutouts that ``verpex-prepare-data``
        writes. Prefer the CLI unless you specifically want this behaviour.
        """
        if get_files is None:
            if poi_source is None:
                raise ValueError(
                    "build_cutouts needs `poi_source`: the BIDS `source-` entity naming which POI "
                    "annotation set to read. Pass it, or supply your own `get_files`."
                )
            get_files = partial(
                _get_files,
                get_poi=partial(get_spine_poi, source=poi_source),
                get_ct=get_ct,
                get_subreg=get_subreg,
                get_vertseg=get_vertseg,
            )
        super().build_cutouts(bids_surgery_info, save_path, get_files, rescale_zoom)


class SpineNeighborDataModule(POIDataModule):
    """Data module that also serves each vertebra's neighbours."""

    #: Neighbour training defaults to no horizontal flip.
    #:
    #: The flip has to remap landmarks within each per-vertebra block, and that
    #: remapping was previously hardcoded to 35 landmarks per vertebra. It is now
    #: derived from the data, so flipping is safe to enable - but the default stays
    #: off, matching what the reference neighbour configs set explicitly and what
    #: PoiNeighborDataset itself defaults to.
    DEFAULT_FLIP_PROB = 0.0

    def __init__(self, **kwargs):
        # Previously this printed a warning and then did nothing - the line that
        # would have disabled flipping was commented out, so the base class default
        # of 0.5 silently applied. An explicit setting is now honoured; only the
        # default differs from the base class.
        kwargs.setdefault("flip_prob", self.DEFAULT_FLIP_PROB)
        super().__init__(dataset=SPINE_NEIGHBOR_DATASET, **kwargs)

    def build_cutouts(
        self,
        bids_surgery_info: BIDS_Global_info,
        save_path: str,
        get_files: Callable | None = None,
        rescale_zoom: tuple | None = None,
        poi_source: str | None = None,
    ) -> None:
        """Build the per-vertebra cutouts for this dataset layout."""
        if get_files is None:
            if poi_source is None:
                raise ValueError(
                    "build_cutouts needs `poi_source`: the BIDS `source-` entity naming which POI "
                    "annotation set to read. Pass it, or supply your own `get_files`."
                )
            get_files = partial(
                _get_files,
                get_poi=partial(get_spine_poi, source=poi_source),
                get_ct=get_ct,
                get_subreg=get_subreg,
                get_vertseg=get_vertseg,
            )
        super().build_cutouts(bids_surgery_info, save_path, get_files, rescale_zoom)


#: Config ``"type"`` string -> data module.
DATA_MODULES = {
    "SpineDataModule": SpineDataModule,
    "SpineNeighborDataModule": SpineNeighborDataModule,
}


def create_data_module(config) -> POIDataModule:
    """Create the data module a config describes.

    Args:
        config: A ``{"type": ..., "params": {...}}`` mapping.

    Returns:
        The constructed data module.

    Raises:
        UnknownTypeError: If the config names a data module that is not registered.
    """
    return build(DATA_MODULES, "data module", config)
