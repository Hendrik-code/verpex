"""Run this before training a model to prepare the data."""

import argparse
import os
from collections.abc import Callable
from functools import partial
from os import PathLike
from pathlib import Path

import pandas as pd
from pqdm.processes import pqdm
from TPTBox import BIDS_FILE, NII, BIDS_Global_info, Log_Type, No_Logger, Subject_Container, np_utils
from TPTBox.core.poi import POI

from verpex.data.quality_reports import (
    convert_agg_report_to_reported_bool_dict,
    is_vert_reported,
    load_agg_report_df,
)
from verpex.paths import get_path

logger = No_Logger(prefix="prepare_data")


def load_exclusion_dict(excel_path) -> dict[str, list[tuple[int, int]]]:
    """Load the POI exclusion list from an Excel file.

    Args:
        excel_path: Path to the exclusion spreadsheet. A missing file yields an
            empty dict, so the exclusion list stays optional.

    Returns:
        A mapping of subject id to the ``(vertebra, poi)`` pairs to exclude.
    """
    if not os.path.exists(excel_path):
        return {}

    df = pd.read_excel(excel_path)

    exclude_dict: dict[str, list[tuple[int, int]]] = {}

    for _, row in df.iterrows():
        subject = row["subject"]
        label = int(row["label"])

        for col in df.columns[2:]:  # columns : 'subject', 'label'
            val = str(row[col]).strip().lower()
            if val == "x":
                try:
                    poi_id = int(col.strip().split()[0])  # e.g. '124 \n(VertBodAntCenR)' → 124
                except ValueError:
                    continue  # is no valid POI ID can be extracted

                if subject not in exclude_dict:
                    exclude_dict[subject] = []
                exclude_dict[subject].append((label, poi_id))

    return exclude_dict


def get_bad_poi_list(subject_id: str, vert: int, exclude_dict: dict[str, list[tuple[int, int]]] | None) -> list[int]:
    """List the POIs to exclude for one vertebra of one subject.

    Args:
        subject_id: Subject id, as it appears in the exclusion spreadsheet.
        vert: Vertebra label.
        exclude_dict: Mapping of subject id to ``(vertebra, poi)`` pairs, as
            returned by :func:`load_exclusion_dict`. ``None`` excludes nothing.

    Returns:
        The POI ids to exclude for this vertebra.
    """
    if exclude_dict is None:
        return []
    bad_pois = exclude_dict.get(subject_id, [])
    filtered_pois = [poi_id for vert_id, poi_id in bad_pois if vert_id == vert]
    return filtered_pois


def get_spine_poi(container, extra_filter: dict[str, str] | None = None) -> tuple[POI, BIDS_FILE]:
    """Find a subject's POI annotation file.

    Args:
        container: The subject's BIDS container.
        extra_filter: Extra BIDS key/value filters to narrow the query.

    Returns:
        The loaded POI and the BIDS file it came from.
    """
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")
    poi_query.filter_filetype("json")  # only json files
    if extra_filter is not None:
        for key, value in extra_filter.items():
            poi_query.filter(key, value)

    if not poi_query.candidates:
        logger.print("ERROR: No POI candidates found!", Log_Type.FAIL)
        return None, None

    poi_candidate = poi_query.candidates[0]
    poi_p = poi_candidate.file["json"]
    logger.print(f"Loading POI from: {poi_p}")

    try:
        poi = POI.load(poi_p)
    except Exception as e:
        logger.print(f"Error loading POI: {e!s}", Log_Type.FAIL)
        return None, poi_candidate
    else:
        return poi, poi_candidate


def get_ct(container, extra_filter: dict[str, str] | None = None) -> tuple[NII, str] | tuple[None, None]:
    """Find a subject's CT image.

    Args:
        container: The subject's BIDS container.
        extra_filter: Extra BIDS key/value filters to narrow the query.

    Returns:
        The loaded image and its path.
    """
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")  # only nifti files
    if extra_filter is not None:
        for key, value in extra_filter.items():
            ct_query.filter(key, value)
    ct_candidate = ct_query.candidates[0]

    logger.print(f"Loading CT from: {ct_candidate.file['nii.gz']}")

    try:
        ct = ct_candidate.open_nii()
        return ct, ct_candidate.file["nii.gz"]
    except Exception as e:
        logger.print(f"Error opening CT: {e!s}", Log_Type.FAIL)
        return None, None


def get_subreg(container, extra_filter: dict[str, str] | None = None) -> NII:
    """Find a subject's vertebra subregion segmentation mask.

    Args:
        container: The subject's BIDS container.
        extra_filter: Extra BIDS key/value filters to narrow the query.

    Returns:
        The loaded mask, or None if it could not be opened.
    """
    subreg_query = container.new_query(flatten=True)
    subreg_query.filter_format("msk")
    subreg_query.filter_filetype("nii.gz")  # only nifti files
    subreg_query.filter("seg", "subreg")
    if extra_filter is not None:
        for key, value in extra_filter.items():
            subreg_query.filter(key, value)
    subreg_candidate = subreg_query.candidates[0]

    logger.print(f"Loading subreg from: {subreg_candidate.file['nii.gz']}")

    try:
        subreg = subreg_candidate.open_nii()
    except Exception as e:
        logger.print(f"Error opening subreg: {e!s}", Log_Type.FAIL)
        return None
    else:
        return subreg


def get_vertseg(container, extra_filter: dict[str, str] | None = None) -> NII:
    """Find a subject's vertebra instance segmentation mask.

    Args:
        container: The subject's BIDS container.
        extra_filter: Extra BIDS key/value filters to narrow the query.

    Returns:
        The loaded mask, or None if it could not be opened.
    """
    vertseg_query = container.new_query(flatten=True)
    vertseg_query.filter_format("msk")
    vertseg_query.filter_filetype("nii.gz")  # only nifti files
    vertseg_query.filter("seg", "vert")
    if extra_filter is not None:
        for key, value in extra_filter.items():
            vertseg_query.filter(key, value)
    vertseg_candidate = vertseg_query.candidates[0]

    logger.print(f"Loading vertseg from: {vertseg_candidate.file['nii.gz']}")

    try:
        vertseg = vertseg_candidate.open_nii()
    except Exception as e:
        logger.print(f"Error opening vertseg: {e!s}", Log_Type.FAIL)
        return None
    else:
        return vertseg


def get_files_withfilter(  # noqa: D103 - thin wrapper over the four getters above
    container,
    get_poi: Callable,
    get_ct_fn: Callable,
    get_subreg_fn: Callable,
    get_vertseg_fn: Callable,
) -> tuple[POI, NII, NII, NII]:
    poi, poi_bf = get_poi(container)
    extra_keys = ["split"]
    extra_filter = {key: value for key, value in poi_bf.info.items() if key in extra_keys} if poi is not None else None
    return (
        poi,
        get_ct_fn(container, extra_filter=extra_filter),
        get_subreg_fn(container, extra_filter=extra_filter),
        get_vertseg_fn(container, extra_filter=extra_filter),
    )


def process_container(  # noqa: ANN201
    subject,
    container,
    save_path: str | PathLike,
    rescale_zoom: tuple | None,
    get_files_fn: Callable[[Subject_Container], tuple[POI, NII, NII, NII]],
    exclusion_dict: dict | None = None,
    compute_surface_mask: bool = False,
    include_neighbouring_vertebrae: bool = False,  # noqa: ARG001 - kept for config compatibility
    report_der2dict: dict | None = None,
    ignore_outer: bool = False,
):
    """Build the per-vertebra cutouts for one subject.

    Crops each vertebra out of the full scan, reorients and rescales it, shifts the
    POI annotations to match, and writes one directory per vertebra.

    Args:
        subject: Subject id.
        container: The subject's BIDS container.
        save_path: Root to write cutouts under.
        rescale_zoom: Target voxel spacing, or ``None`` to keep the native spacing.
        get_files_fn: Callable returning this subject's POI, CT and masks.
        exclusion_dict: POIs to mark as excluded, from :func:`load_exclusion_dict`.
        compute_surface_mask: Also compute and store a surface mask per cutout.
        include_neighbouring_vertebrae: Unused; kept for config compatibility.
        report_der2dict: QA report index; flagged vertebrae are skipped.
        ignore_outer: Skip the outermost vertebrae of the scan.

    Returns:
        One ``master_df`` row per written cutout.
    """
    logger.print(f"Processing Subject: {subject}")
    poi, (ct, ct_p), subreg, vertseg = get_files_fn(container)

    if ct is None or subreg is None or vertseg is None or poi is None:
        logger.print(f"Skipping subject {subject} due to missing data.", Log_Type.FAIL)
        return []

    # reorient data to same orientation
    ct.reorient_(("L", "A", "S"))
    subreg.reorient_(("L", "A", "S")).map_labels_({50: 49}, verbose=False)
    vertseg.reorient_(("L", "A", "S"))
    poi.reorient_(axcodes_to=vertseg.orientation, _shape=vertseg.shape)

    surface_mask = None
    surface_subreg = None
    # if compute_surface_mask:
    #    try:
    #        surface_mask = vertseg.compute_surface_mask(connectivity=3, dilated_surface=False)
    #        surface_subreg = subreg.compute_surface_mask(connectivity=3, dilated_surface=False)
    #    except Exception as e:
    #        print(f"Error computing surface mask for subject {subject}: {str(e)}")
    #        surface_mask = None
    #        surface_subreg = None

    vertebrae = sorted({key[0] for key in poi.keys()})
    if ignore_outer:
        vertebrae = vertebrae[1:-1]  # ignore outermost vertebrae, which often have more artifacts and less reliable POI annotations

    if rescale_zoom:
        ct.rescale_(rescale_zoom)
        subreg.rescale_(rescale_zoom)
        vertseg.rescale_(rescale_zoom)
        poi.rescale_(rescale_zoom)

    vertseg_arr = vertseg.get_array()
    summary = []

    for index in range(len(vertebrae)):  # loops through each vertebra ID (extracted from POI keys)
        vert = vertebrae[index]
        if vert in vertseg_arr:  # vertebra found in segmentation mask
            subject_ct_id = ct_p.name.split(".")[0]
            if report_der2dict is not None and is_vert_reported(report_der2dict, subject_ct_id, vert):
                logger.print(f"Skipping {subject}, vert={vert} because it is reported in the report dict.", Log_Type.STRANGE)
                continue

            ## TODO: check whether the neighbours are present in vertseg_arr; if not, what should happen?
            # if include_neighbouring_vertebrae:
            #    vert_neighbours = [vert]
            #    if index > 0:
            #        vert_neighbours.insert(0, vertebrae[index - 1])
            #    if index < len(vertebrae) - 1:
            #        vert_neighbours.append(vertebrae[index + 1])
            #
            #    print(f"Vertebra {vert} neighbours: {vert_neighbours}")
            #
            #    # Initialize bounding box limits
            #    x_min, x_max = np.inf, -np.inf
            #    y_min, y_max = np.inf, -np.inf
            #    z_min, z_max = np.inf, -np.inf
            #
            #    for v in vert_neighbours:
            #        try:
            #            bounds = np_utils.calc
            #
            #        except ValueError as e:
            #            print(f"Error getting bounding box for vertebra {v}: {str(e)}")
            #            continue
            #
            #        x_min = min(x_min, bounds[0])
            #        x_max = max(x_max, bounds[1])
            #        y_min = min(y_min, bounds[2])
            #        y_max = max(y_max, bounds[3])
            #        z_min = min(z_min, bounds[4])
            #        z_max = max(z_max, bounds[5])
            #
            # else:
            #    try:
            #        x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box(vertseg_arr, vert)
            #    except ValueError as e:
            #        print(f"Error getting bounding box for vertebra {vert}: {str(e)}")
            #        continue

            # defines output paths for cropped files
            ct_path = os.path.join(save_path, subject, str(vert), "ct.nii.gz")
            subreg_path = os.path.join(save_path, subject, str(vert), "subreg.nii.gz")
            vertseg_path = os.path.join(save_path, subject, str(vert), "vertseg.nii.gz")
            poi_path = os.path.join(save_path, subject, str(vert), "poi.json")

            # create directories if they do not exist
            if not os.path.exists(os.path.join(save_path, subject, str(vert))):
                os.makedirs(os.path.join(save_path, subject, str(vert)))

            # Bound before the try so the diagnostic below cannot raise
            # UnboundLocalError and mask the original exception.
            crop = padding = None
            try:
                com = np_utils.np_center_of_mass(vertseg.extract_label(vert).get_seg_array())[1]
                ct_cropped, crop, padding = np_utils.np_calc_crop_around_centerpoint(
                    (round(com[0]), round(com[1]), round(com[2])), ct.get_array(), cutout_size=(128, 128, 144)
                )
                ct_cropped = ct.apply_crop(crop).apply_pad(padding, verbose=False)
                subreg_cropped = subreg.apply_crop(crop).apply_pad(padding, verbose=False)
                vertseg_cropped = vertseg.apply_crop(crop).apply_pad(padding, verbose=False)
                poi_cropped = poi.apply_crop(crop).resample_from_to(vertseg_cropped)

                # ct_cropped = ct.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
                # subreg_cropped = subreg.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
                # vertseg_cropped = vertseg.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
                # poi_cropped = poi.apply_crop(o_shift=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))

                # surface_mask_cropped = None
                # surfcae_subreg_cropped = None
                # if compute_surface_mask and surface_mask is not None and surface_subreg is not None:
                #    surface_mask_cropped = surface_mask.apply_crop(ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max)))
                #    surface_subreg_cropped = surface_subreg.apply_crop(
                #        ex_slice=(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))
                #    )

            except Exception as e:
                print(f"Error processing {subject}, vert={vert}: {e!s}")
                print(f"Crop dimensions: crop={crop}, padding={padding}")
                # print(f"Crop dimensions: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, z_min={z_min}, z_max={z_max}")
                # print(f"ex_slice: {(slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))}")
                print(f"ct shape: {ct.shape},\n subreg shape: {subreg.shape},\n vertseg shape: {vertseg.shape}, poi shape: {poi.shape}")
                raise

                # if compute_surface_mask and surface_mask_cropped is not None and surface_subreg_cropped is not None:
                #    surface_mask_cropped.rescale_(rescale_zoom)
                #    surface_subreg_cropped.rescale_(rescale_zoom)

            ct_cropped.save(ct_path, verbose=False)
            subreg_cropped.save(subreg_path, verbose=False)
            vertseg_cropped.save(vertseg_path, verbose=False)
            poi_cropped.save(poi_path, verbose=True)

            if compute_surface_mask:
                try:
                    surface_mask = vertseg_cropped.compute_surface_mask(connectivity=3, dilated_surface=False)
                    surface_subreg = subreg_cropped.compute_surface_mask(connectivity=3, dilated_surface=False)
                except Exception:
                    pass

                if compute_surface_mask and surface_mask is not None and surface_subreg is not None:
                    surface_mask_path = os.path.join(save_path, subject, str(vert), "surface_msk.nii.gz")
                    surface_subreg_path = os.path.join(save_path, subject, str(vert), "surface_subreg.nii.gz")
                    surface_mask.save(surface_mask_path, verbose=False)
                    surface_subreg.save(surface_subreg_path, verbose=True)

            # if compute_surface_mask and surface_mask_cropped is not None and surface_subreg_cropped is not None:
            #    surface_mask_cropped.save(surface_mask_path, verbose=False)
            #    surface_subreg_cropped.save(surface_subreg_path, verbose=False)
            #
            # Save the slice indices as json to reconstruct the original POI file (there probably is a more BIDS-like approach to this)
            # slice_indices = {
            #    "x_min": int(x_min),
            #    "x_max": int(x_max),
            #    "y_min": int(y_min),
            #    "y_max": int(y_max),
            #    "z_min": int(z_min),
            #    "z_max": int(z_max),
            # }
            # with open(
            #    os.path.join(save_path, subject, str(vert), "cutout_slice_indices.json"),
            #    "w",
            #    encoding="utf-8",
            # ) as f:
            #    json.dump(slice_indices, f)

            summary.append(
                {
                    "subject": subject,
                    "vertebra": vert,
                    # Written relative to save_path so the resulting master_df.csv is
                    # portable; PoiDataset resolves it against the configured
                    # cutout_root (see verpex.data.dataset.resolve_cutout_dir).
                    "file_dir": os.path.join(subject, str(vert)),
                    "bad_poi_list": get_bad_poi_list(f"sub-{subject}", vert, exclusion_dict),
                }
            )

        else:
            print(f"Vertebra {vert} has no segmentation for subject {subject}")

    return summary


def prepare_data(  # noqa: ANN201
    bids_surgery_info: BIDS_Global_info,
    save_path: str,
    get_files_fn: Callable,
    exclusion_path: str | None = None,
    rescale_zoom: tuple | None = None,
    n_workers: int = 8,
    compute_surface_mask: bool = False,
    ignore_outer: bool = False,
    include_neighbouring_vertebrae: bool = False,
    report_der2dict=None,
):
    """Build cutouts for every subject in a BIDS dataset, in parallel.

    Args:
        bids_surgery_info: The dataset to read.
        save_path: Root to write cutouts under.
        get_files_fn: Callable returning a subject's POI, CT and masks.
        exclusion_path: Optional Excel file listing POIs to mark as excluded.
        rescale_zoom: Target voxel spacing, or ``None`` to keep the native spacing.
        n_workers: Number of parallel worker processes.
        compute_surface_mask: Also compute and store a surface mask per cutout.
        ignore_outer: Skip the outermost vertebrae of each scan.
        include_neighbouring_vertebrae: Unused; kept for config compatibility.
        report_der2dict: QA report index; flagged vertebrae are skipped.

    Writes:
        One directory per vertebra under ``save_path``, plus ``master_df.csv``
        listing them with paths relative to ``save_path``.
    """
    master = []
    exclusion_dict = load_exclusion_dict(exclusion_path) if exclusion_path is not None else None

    partial_process_container = partial(
        process_container,
        save_path=save_path,
        rescale_zoom=rescale_zoom,
        get_files_fn=get_files_fn,
        exclusion_dict=exclusion_dict,  # Pass None if not provided
        compute_surface_mask=compute_surface_mask,
        include_neighbouring_vertebrae=include_neighbouring_vertebrae,
        report_der2dict=report_der2dict,
        ignore_outer=ignore_outer,
    )

    master = pqdm(
        bids_surgery_info.enumerate_subjects(),
        partial_process_container,
        n_jobs=n_workers,
        argument_type="args",
        # exception_behaviour="immediate",
        exception_behaviour="continue",
    )
    master = [item for sublist in master for item in sublist]
    master_df = pd.DataFrame(master)
    master_df.to_csv(os.path.join(save_path, "master_df.csv"), index=False)


def main() -> None:
    """Build per-vertebra cutouts from a BIDS dataset."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the BIDS dataset. Defaults to the configured data_root.",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--derivatives_name",
        type=str,
        help="The name of the derivatives folder",
        # required=True,
        nargs="+",
        default=["derivatives_seg", "derivatives_poi_new2g", "derivatives_combined", "derivatives_poi_automatic_correction-v4-onlygood"],
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The path to save the prepared data. Defaults to the configured cutout_root.",
        default=None,
    )
    parser.add_argument(
        "--no_rescale",
        action="store_true",
        help="Whether to skip rescaling the data to isotropic voxels",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        help="The number of workers to use for parallel processing",
        default=1,
    )

    parser.add_argument(
        "--set_zoom",
        type=lambda x: tuple(map(float, x.split(","))),
        help="Zoom for rescaling (format: x,y,z)",
        default=(1, 1, 1),
    )

    parser.add_argument("--exclude_path", type=str, help="Path to Excel file marking POIs to exclude", default=None)

    parser.add_argument(
        "--compute_surface_mask",
        action=argparse.BooleanOptionalAction,
        help="Whether to compute the surface mask for the vertebrae",
        default=True,
    )

    parser.add_argument(
        "--ignoreouter",
        action="store_true",
        help="If true, will not prepare top and bottom vertebrae",
        default=False,
    )

    parser.add_argument(
        "--include_neighbouring_vertebrae",
        action="store_true",
        help="Whether to include neighbouring vertebrae in the bounding box extraction",
    )

    parser.add_argument(
        "--report_root",
        type=str,
        help="Path to text file containing list of subjects to ignore in case of errors (one subject per line)",
        default=None,
    )
    parser.add_argument(
        "--ignore_err_list",
        type=str,
        help="Path to text file containing list of subjects to ignore in case of errors (one subject per line)",
        default=None,
        # default="TEST_derivatives_poi_automatic_correction-v3-6-onlygood",
    )

    args = parser.parse_args()
    print(args)

    parents = ["rawdata", args.derivatives_name] if not isinstance(args.derivatives_name, list) else ["rawdata", *args.derivatives_name]

    # Fall back to the configured paths when a location was not given on the command
    # line. get_path raises a PathConfigError naming the setting to fix, rather than
    # letting an unset path surface as a FileNotFoundError deep inside preprocessing.
    save_path = args.save_path or str(get_path("cutout_root"))
    data_path = args.data_path or [str(get_path("data_root"))]

    if args.ignore_err_list is not None:
        report_root = Path(args.report_root) if args.report_root else get_path("output_root")
        report_df = load_agg_report_df(report_root / args.ignore_err_list / "aggregated_poi_report.xlsx")
        report_der2dict = convert_agg_report_to_reported_bool_dict(report_df)
    else:
        report_der2dict = None

    data_path = data_path if isinstance(data_path, list) else [data_path]

    bids_gloabl_info = BIDS_Global_info(
        datasets=data_path,
        parents=parents,
    )

    get_data_files = partial(
        get_files_withfilter,
        get_poi=get_spine_poi,
        get_ct_fn=get_ct,
        get_subreg_fn=get_subreg,
        get_vertseg_fn=get_vertseg,
    )

    prepare_data(
        bids_surgery_info=bids_gloabl_info,
        save_path=save_path,
        exclusion_path=args.exclude_path,
        get_files_fn=get_data_files,
        rescale_zoom=None if args.no_rescale else args.set_zoom,
        n_workers=args.n_workers,
        compute_surface_mask=args.compute_surface_mask,
        include_neighbouring_vertebrae=args.include_neighbouring_vertebrae,
        ignore_outer=args.ignoreouter,
        report_der2dict=report_der2dict,
    )


if __name__ == "__main__":
    main()
