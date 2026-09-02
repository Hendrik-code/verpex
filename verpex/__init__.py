"""Prediction of anatomical points-of-interest (POIs) on vertebrae.

A two-stage coarse-to-fine 3D landmark-regression pipeline: a DenseNet backbone
predicts per-landmark heatmaps on a single-vertebra cutout, and a transformer
refines those coarse coordinates using local image patches.

See the README for the training and inference workflows.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    #: Derived from the latest git tag at build time by poetry-dynamic-versioning.
    __version__ = version("verpex")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
