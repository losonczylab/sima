#!/usr/bin/env python
"""Script to update from SIMA 0.x to 1.x

To convert old SIMA directories to the newest version:
>> python update_version.py /path/to/datasets

For help and additional arguments:
>> python update_version.py --help

NOTE: Conversion is not necessary to read datasets created with SIMA 0.x
SIMA 1.x is backwards-compatible as all SIMA 0.x datasets are converted to
read-only 1.x sets as needed. This process is slower than just directly
loading a 1.x dataset, so updating SIMA 0.x datasets can improve read times.

"""

import os
import shutil
import argparse
import datetime

from sima import ImagingDataset
from sima.misc.convert import _0_to_1

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "directory", action="store", type=str,
        help="Target directory to search for 0.x SIMA folders")
    argParser.add_argument(
        "-b", "--backup", action="store_true",
        help="Old dataset.pkl will be backed-up before conversion")
    args = argParser.parse_args()

    for directory, folders, files in os.walk(args.directory):
        if directory.endswith('.sima'):
            try:
                dataset = ImagingDataset(None, directory)
            except ImportError as error:
                if not error.args[0] == 'No module named iterables':
                    raise error
                # Possibly old SIMA directory, attempt convert
                if args.backup:
                    date_stamp = datetime.date.today().strftime('%Y%m%d')
                    shutil.copyfile(
                        os.path.join(directory, 'dataset.pkl'),
                        os.path.join(directory, 'dataset.pkl.{}.bak'.format(
                            date_stamp)))
                try:
                    # Convert and overwrite the current dataset
                    _0_to_1(directory, directory)
                except:
                    # Convert failed, possibly bad SIMA directory or path
                    pass
                else:
                    print "Dataset successfully update: " + directory
            except:
                # Possibly corrupt directory or an unsupported version
                pass
            else:
                # Dataset loaded successfully, already current version
                pass
