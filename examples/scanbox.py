"""Example script to load Scanbox data as a SIMA sequence."""
from __future__ import division

import argparse
import fnmatch
import os
import numpy as np

import sima
from sima import imaging_parameters


def sbxread(path, info_path):
    """Read in an .sbx file and return a SIMA sequence.

    Based off of the sbxRead Matlab implementation and
    https://scanbox.org/2016/09/02/reading-scanbox-files-in-python/

    Parameters
    ----------
    path : str
        Path to the Scanbox data file (including the .sbx extension).
    info_path : str
        Path to the Scanbox info MAT file.

    """
    info = imaging_parameters.extract_imaging_parameters(
        info_path, format='Scanbox')

    nrows = info['recordsPerBuffer']
    ncols = info['sz'][1]
    nchannels = info['nchannels']
    nplanes = info['nplanes']
    nframes = (info['max_idx'] + 1) // nplanes
    shape = (nchannels, ncols, nrows, nplanes, nframes)

    seq = sima.Sequence.create(
        'memmap', path=path, shape=shape, dim_order='cxyzt', dtype='uint16',
        order='F')

    max_uint16_seq = sima.Sequence.create(
        'constant', value=np.iinfo('uint16').max, shape=seq.shape)

    return max_uint16_seq - seq


def initialize_sbx_datasets(path, calc_time_averages=False):
    """Locate and initialize a SIMA dataset for all Scanbox sbx files."""
    for directory, folders, files in os.walk(path):
        for sbx_file in fnmatch.filter(files, '*.sbx'):
            info_file = os.path.splitext(sbx_file)[0] + '.mat'
            sima_dir = os.path.splitext(sbx_file)[0] + '.sima'
            if info_file in files and sima_dir not in folders:
                print("Initializing SIMA dataset: {}".format(
                    os.path.join(directory, sima_dir)))
                seq = sbxread(
                    os.path.join(directory, sbx_file),
                    os.path.join(directory, info_file))
                dset = sima.ImagingDataset(
                    [seq], savedir=os.path.join(directory, sima_dir))
                if calc_time_averages:
                    print("Calculating time averages: {}".format(
                        os.path.join(directory, sima_dir)))
                    dset.time_averages


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-t", "--time_averages", action="store_true",
        help="Pre-calc time averages.")
    argParser.add_argument(
        "path", action="store", type=str, default=os.curdir,
        help="Locate all Scanbox files below this path.")

    args = argParser.parse_args()

    initialize_sbx_datasets(args.path, args.time_averages)
