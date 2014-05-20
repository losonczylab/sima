#!/usr/bin/env python
"""Example extract script

To extract all ROI sets in all SIMA datasets:
>> python extraction.py /path/to/sima/datasets

For help and additional arguments:
>> python extraction.py --help

"""

import os
import argparse

from sima import ImagingDataset

# Never extract ROI sets with labels that contain any of these words
EXCLUDED_ROI_LABELS = ('auto', 'ignore', 'test', 'junk')


def labels_to_extract(dataset, signal_channel='Ch2', overwrite=False):
    """Looks through all of the ROI sets in an ImagingDataset and finds
    all of the sets that are either new or have been modified since they were
    last extracted.

    Parameters
    ----------
    dataset : sima.ImagingDataset
    signal_channel : string or integer
        label or index of channel to extract
    overwrite : boolean
        If True, returns all ROI set labels in the ImagingDataset

    Return
    ------
    list
        Returns a list of the ROI set labels to be extracted

    """
    labels = []
    for roi_label in dataset.ROIs:
        if roi_label not in EXCLUDED_ROI_LABELS:
            if overwrite or roi_label not in dataset.signals(signal_channel):
                labels.append(roi_label)
            else:
                roi_time = dataset.ROIs[roi_label].timestamp
                signals_time = dataset.signals(
                    signal_channel)[roi_label]['timestamp']
                if roi_time > signals_time:
                    labels.append(roi_label)
    return labels


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


def extract_dataset(dataset, signal_channel='Ch2', demix_channel=None,
                    overwrite=False, include_overlap=False):
    try:
        demix_channel = dataset._resolve_channel(demix_channel)
    except ValueError:
        demix_channel = None
    for roi_label in labels_to_extract(dataset, signal_channel, overwrite):
        print("Extracting label '{}' from {}".format(roi_label,
                                                     dataset.savedir))
        dataset.extract(rois=dataset.ROIs[roi_label],
                        label=roi_label,
                        signal_channel=signal_channel,
                        demix_channel=demix_channel,
                        remove_overlap=not include_overlap)

        print('Extraction complete: {}'.format(dataset.savedir))

if __name__ == '__main__':
    """Find all SIMA data within the directory and extract signals."""

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-x", "--demix_channel", action="store", type=str,
        help="Name of the channel to demix from the signal channel")
    argParser.add_argument(
        "-i", "--include_overlap", action="store_true",
        help="Include pixels that overlap between ROIs")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Re-extract datasets with pre-existing signals files")
    argParser.add_argument(
        "-s", "--signal_channel", action="store", type=str, default="Ch2",
        help="Name of the signal channel to extract, defaults to \'Ch2\'")
    argParser.add_argument(
        "directory", action="store", type=str, default=os.curdir,
        help="Extract all datasets below this folder")
    args = argParser.parse_args()

    for dataset in locate_datasets(args.directory):
        extract_dataset(
            dataset, signal_channel=args.signal_channel,
            demix_channel=args.demix_channel, overwrite=args.overwrite,
            include_overlap=args.include_overlap)
