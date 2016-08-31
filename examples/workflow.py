#! /usr/bin/env python
"""
This file provides a demonstration of how to use the SIMA package.

In order to run this file, you will need to download the file
http://www.losonczylab.org/workflow_data.zip and extract it in your
current working directory.

"""
from __future__ import print_function
from builtins import input
from builtins import range

import os
import shutil

##############################################################################
#                                                                            #
#   PART 0: Import SIMA and necessary submodules.                            #
#                                                                            #
##############################################################################

import sima
import sima.motion
import sima.segment

##############################################################################
#                                                                            #
#   PART 1: Preparing the iterables.                                         #
#                                                                            #
##############################################################################

# Generate the filenames with Python list comprehensions.
tiff_filenames = [
    ['workflow_data/Cycle{n1:02d}_Ch{n2}.tif'.format(n1=cycle, n2=channel)
     for channel in range(1, 3)] for cycle in range(1, 16)
]

# The resulting filenames are printed for clarification.
print("TIFF filenames:\n", tiff_filenames)


# Finally, we construct a MultiPageTIFF iterable using each of the filenames.
sequences = [
    sima.Sequence.join(*[sima.Sequence.create('TIFF', chan) for chan in cycle])
    for cycle in tiff_filenames]

##############################################################################
#                                                                            #
#   PART 2: Running motion correction to create the dataset, and exporting   #
#           the corrected image data.                                        #
#                                                                            #
##############################################################################

dataset_path = 'workflow_data/dataset.sima'
correction_approach = sima.motion.HiddenMarkov2D(num_states_retained=30,
                                                 max_displacement=[20, 30])

if os.path.exists(dataset_path):
    while True:
        input_ = input("Dataset path already exists. Overwrite? (y/n) ")
        if input_ == 'n':
            exit()
        elif input_ == 'y':
            shutil.rmtree(dataset_path)
            break

print("Running motion correction.")
dataset = correction_approach.correct(
    sequences, dataset_path, channel_names=['tdTomato', 'GCaMP'],
    trim_criterion=0.95)

# Export the time averages for a manuscript figure.
print("Exporting motion-corrected time averages.")
dataset.export_averages(['workflow_data/tdTomato.tif',
                         'workflow_data/GCaMP.tif'])

# Generate the output filenames with Python list comprehensions.
output_filenames = [
    [[channel.replace('.tif', '_corrected.tif') for channel in cycle]]
    for cycle in tiff_filenames
]

# The resulting filenames are printed for clarification.
print("Output filenames:\n", output_filenames)

# Export the corrected frames for a presentation.
print("Exporting motion-corrected movies.")
dataset.export_frames(output_filenames, fill_gaps=True)

# At this point, one may wish to inspect the exported image data to evaluate
# the quality of the motion correction before continuing.
while True:
    input_ = input("Continue? (y/n): ")
    if input_ == 'n':
        exit()
    elif input_ == 'y':
        break

##############################################################################
#                                                                            #
#   PART 3: Running automated segmentation and editing results with the ROI  #
#           Buddy GUI.                                                       #
#                                                                            #
##############################################################################

# Segment the field of view into ROIs using the method for CA1 pyramidal cells
# and parameters that were determined based on the imaging magnification.
segmentation_approach = sima.segment.PlaneCA1PC(
    channel='GCaMP',
    num_pcs=30,
    max_dist=(3, 6),
    spatial_decay=(3, 6),
    cut_max_pen=0.10,
    cut_min_size=50,
    cut_max_size=150,
    x_diameter=14,
    y_diameter=7,
    circularity_threhold=.5,
    min_roi_size=20,
    min_cut_size=40
)

print("Running auto-segmentation.")
dataset.segment(segmentation_approach, 'auto_ROIs')

# At this point, one may wish to edit the automatically segmented ROIs using
# the ROI Buddy GUI before performing signal extraction.
while True:
    input_ = input("Continue? (y/n): ")
    if input_ == 'n':
        exit()
    elif input_ == 'y':
        break

##############################################################################
#                                                                            #
#   PART 4: Extracting fluorescence signals from the ROIs.                   #
#                                                                            #
##############################################################################

# Reload the dataset in case any changes have been made with ROI Buddy
dataset = sima.ImagingDataset.load(dataset_path)

# Extract the signals. By default, the most recently created ROIs are used.
print("Extracting signals.")
dataset.extract(signal_channel='GCaMP', label='GCaMP_signals')

# Export the extracted signals to a CSV file.
print("Exporting GCaMP time series.")
dataset.export_signals('example_signals.csv', channel='GCaMP',
                       signals_label='GCaMP_signals')

##############################################################################
#                                                                            #
#   PART 5: Visualizing data using Python.                                   #
#                                                                            #
##############################################################################

# import necessary functions from matplotlib
from matplotlib.pyplot import plot, show

# plot the signal from an ROI object, with a different color for each cycle
print("Displaying example calcium trace.")
raw_signals = dataset.signals('GCaMP')['GCaMP_signals']['raw']
for sequence in range(3):  # plot data from the first 3 cycles
    plot(raw_signals[sequence][3])  # plot the data from ROI #3
show(block=True)
