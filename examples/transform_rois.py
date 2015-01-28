#!/usr/bin/env python
"""Example ROI transform script

To transform ROIs from a source dataset to target datasets:
>> python transform_rois.py /path/to/source/dataset /path/to/target/datasets
    SOURCE_LABEL

For help and additional arguments:
>> python transform_rois.py --help

"""

import os
import argparse

from sima import ImagingDataset
from sima.misc import TransformError

if __name__ == '__main__':
    """Calculate an affine transform between the source dataset and all
    datasets within the target directory.
    Apply this transform to all ROIs in the SOURCE_LABEL ROIList and copy them
    to the target datasets.
    Auto-generated ROILists are saved with the TARGET_LABEL label"""

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "source", action="store", type=str,
        help="Source SIMA directory for the ROIs")
    argParser.add_argument(
        "target", action="store", type=str,
        help="Target path to locate SIMA directories to copy ROIs to")
    argParser.add_argument(
        "source_label", action="store", type=str,
        help="ROIList label for the ROIs to be copied")
    argParser.add_argument(
        "-l", "--target_label", action="store", type=str,
        default="auto_transformed",
        help="Label to give the new transformed ROIs "
        + "(default: auto_transformed)")
    argParser.add_argument(
        "-c", "--channel", action="store", type=str, default="0",
        help="Channel of the datasets used to calculate the affine transform")
    argParser.add_argument(
        "-C", "--copy_properties", action="store_true",
        help="Copy ROI properties ")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="If target_label already exists, overwrite")
    args = argParser.parse_args()

    source_dataset = ImagingDataset.load(args.source)
    print "Beginning ROI transforms, source dataset: ", args.source
    print "-----------------------------------------"

    for directory, folders, files in os.walk(args.target):
        if directory.endswith('.sima'):
            try:
                target_dataset = ImagingDataset.load(directory)
            except IOError:
                continue

            if os.path.samefile(args.source, directory):
                continue

            if args.target_label in target_dataset.ROIs and not args.overwrite:
                print "Label already exists, skipping: ", directory
                continue

            try:
                channel = int(args.channel)
            except ValueError:
                channel = args.channel

            try:
                target_dataset.import_transformed_ROIs(
                    source_dataset=source_dataset,
                    source_channel=channel,
                    target_channel=channel,
                    source_label=args.source_label,
                    target_label=args.target_label,
                    copy_properties=args.copy_properties)
            except TransformError:
                print "FAIL, transform error: ", directory
            else:
                print "SUCCESS, ROIs transformed:", directory
