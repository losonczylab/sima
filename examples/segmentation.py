#!/usr/bin/env python
"""Example segmentation script

To generate normalized cuts ROIs in all SIMA datasets:
>> python segmentation.py /path/to/sima/datasets

For help and additional arguments:
>> python segmentation.py --help

"""

import os
import argparse

from sima import ImagingDataset


def locate_datasets(search_directory):
    """Locates all SIMA directories below 'search_directory'"""
    for directory, folders, files in os.walk(search_directory):
        if directory.endswith('.sima'):
            try:
                dataset = ImagingDataset.load(directory)
            except IOError:
                continue
            else:
                yield dataset


if __name__ == '__main__':
    """Find all SIMA data within the directory and perform normalized cuts
       segmentation.  Auto-generated ROILists are saved with 'AUTO' label"""

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-c", "channel", action="store", type=int,
        help="Channel whose signals will be used for segmentation")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Re-segment datasets with pre-existing ROIs")
    argParser.add_argument(
        "directory", action="store", type=str, default=os.curdir,
        help="Segment all datasets below this folder")
    args = argParser.parse_args()

    #Define normalized cut parameters
    normcut_kwargs = {'channel': args.channel,
                      'num_pcs': 75,
                      'max_dist': (2, 2),
                      'spatial_decay': (2, 2),
                      'cut_max_pen': 0.01,
                      'cut_min_size': 40,
                      'cut_max_size': 200}

    for dataset in locate_datasets(args.directory):
        if args.overwrite or not len(dataset.ROIs()):
            dataset.segment(method='normcut',
                            label='AUTO',
                            kwargs=normcut_kwargs
                            )
            print('Normcut segmentation complete: {}'.format(dataset.savedir))
 