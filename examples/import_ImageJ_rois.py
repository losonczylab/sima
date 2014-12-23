#!/usr/bin/env python
"""Import ImageJ ROIs

To identify and import ZIP files within all SIMA directories:
>> python import_ImageJ_rois.py /path/to/sima/datasets rois

For help and additional arguments:
>> python import_ImageJ_rois.py --help

"""

import argparse
import os

import sima


def remove_empty_ROIs(rois, im_shape):
    """Identify and remove empty ROIs"""
    num_removed = 0
    for roi in reversed(list(rois)):
        roi.im_shape = im_shape
        if all([mask.nnz == 0 for mask in roi.mask]):
            rois.remove(roi)
            num_removed += 1
    if num_removed:
        print('Removed {} empty ROIs'.format(num_removed))

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "directory", action="store", type=str,
        help="path to search for SIMA folders with ROIs")
    argParser.add_argument(
        "label", action="store", type=str,
        help="label to give imported ROIs")
    argParser.add_argument(
        "-n", "--no_action", action="store_true",
        help="Do nothing, just report the changes that would be made")
    args = argParser.parse_args()

    for directory, folders, files in os.walk(args.directory):
        if directory.endswith('.sima'):
            zip_files = filter(
                lambda x: x.endswith('zip') and not x.startswith('._'), files)
            if len(zip_files) == 0:
                continue
            elif len(zip_files) > 1:
                print "Multiple ZIP files found, skipping directory: {}".format(
                    directory)
                continue
            try:
                dataset = sima.ImagingDataset.load(directory)
            except IOError:
                print "Unable to load ImagingDataset: {}".format(directory)
                continue
            print('Loaded {}'.format(dataset.savedir))
            zip_path = os.path.join(directory, zip_files[0])
            rois = sima.ROI.ROIList.load(zip_path, fmt='ImageJ')
            print('Loaded {} ROIs from {}'.format(len(rois), zip_path))
            remove_empty_ROIs(
                rois, im_shape=dataset.frame_shape[:-1])
            sima.misc.copy_label_to_id(rois)

            if not args.no_action:
                dataset.add_ROIs(rois, label=args.label)
            print("Added {} ROIs with label '{}' to {}".format(
                len(rois), args.label, dataset.savedir))
