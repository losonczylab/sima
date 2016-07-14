#!/root/anaconda/bin/python

"""
Written by Vijay Namboodiri and Randall Ung of the Stuber lab in
UNC Chapel Hill.

This program runs SIMA motion correction or ROI extraction in parallel for
multiple files stored on S3. Since multiple simultaneous SSL connections
are not alloweed on to S3 from the same computer with different processes,
the downloads and uploads of files to S3 are done serially. The motion
correction/extraction themselves are run in parallel. The intent of the
program is set by global variables defined below between "From Here" and
"To Here". Also, this program is tailored currently to the workflow of the
 Stuber lab. Users are free to adapt or modify it for their own needs.

Note: Users are required to set their AWS credentials in this script.
 Hence, it is highly recommended to move this file to a local filesystem
 before running it. We thought that setting the AWS credentials explicitly
 in this text file is safer than setting an environment variable automatically
 on the EC2 instance. This way, users will hopefully remember that their
 credentials are explicit in this file and be careful with its sharing.

To run the program, you need to set a source directory on Amazon S3.
The program finds all files in the source directory that need to be analyzed
 and analyzes them. Only files contained in the source directory and not in
 subdirectories are analyzed. Also, if you want to run motion correction, if
 a data file named 'originalfilename.h5' exists, it will be not be run if
 another blank text file named 'originalfilename_mc.txt' exists. This is our
 mechanism to filter out files that have already been analyzed from being
 re-analyzed. For this automation to work, you have to make sure that every
 single file directly contained in the source directory are data files with
 either a .tif or .h5 extension. After motion correction, the script uploads
 the motion correction SIMA datasets (directories with .sima extension) to the
 source directory. Also, it uploads the standard deviation time projection of
 the frames to a subdirectory named Results. If you are performing subsequent
 ROI extraction on hand-drawn ROIs from ImageJ, name those ROIs as
 originalfilename_mc_RoiSet.zip where originalfilename was the original data
 file's name. These should be placed in the Results folder before running
 extraction.

Assumptions:
1. We assume that every data file that needs processed has the same extension
 (either .tif or .h5). This is calculated automatically by the script.
2. We assume that there is a directory on S3 named logfiles directly in the
 bucket of interest. This is where the log of the script will be uploaded. If
 the script fails, you can check the logfile to see what went wrong.
3. If there are individual tif files for individual frames, we assume that
 they are numbered according to the frame number such that there is a base
 filename, then an '_' and then a number. We sort them using the number after
 the last underscore. For instance, filenames could be
 experiment151006_001.tif etc. For this case, use is_single_file = False
 in the global variables.
4. If any filename ends with _CH1, we assume that this file was from a dual
 channel imaging session such that there is also another corresponding file
 whose name is the same as the above file but ends with _CH2.


Things to note:
1. Set the global variables right below the import statements to set the
 intent of the script.
2. If motion corrected videos are to be exported, they will be exported in
 hdf5 format to avoid any 4gb memory issues with tif files. Users are free
 to modify the motion_correction() function to change this behavior.
3. The script removes all pre-existing SIMA directories on the EC2 machine so
 as to avoid SIMA from throwing an error.


Our lab's workflow:
1. Take raw .oir files collected by our Olympus microscope and convert them
 to tiff stacks using an Olympus software.
2. Combine all the tiff stacks per experimental session into a single HDF5
 file using a custom Python script. If you'd like to have this script,
 let us know.
3. Upload the HDF5 files to S3. Say an exampled file is named
 originalfilename.h5
4. Run this script on EC2 by giving it the path to the S3 source directory
 containing the different HDF5 files that need to be motion corrected.
5. The results are uploaded to S3 in the form of motion corrected SIMA dataset
 in a directory named originalfilename_mc.sima in the source directory, and,
 the standard deviation (std) frame projected across time named
 originalfilename_mc_std.tif in a directory named Results in the source
 directory.
6. Once motion correction is done, make sure to upload a blank text file named
 originalfilename_mc.txt to let the program know that originalfilename.h5 has
 already been motion corrected. This way, the next time you run motion
 correction on the source directory, this file will not be motion corrected.
 Below, once extraction is also completed on this file, rename the above text
 file to originalfilename_mc_extract.txt to let the program know that both
 motion correction and extraction have been run on originalfilename.h5.
7. Using the std frame, draw ROIs in ImageJ and upload a file named
 originalfilename_mc_RoiSet.zip for each session into the Results
 sub-directory.
8. Using this script, extract the traces for each ROI by setting action to
 'extract' below.
9. The results are uploaded to S3 as a numpy file. This is used for all
 subsequent processing.
10. In case the motion corrected video needs to be reconstructed, the motion
 corrected SIMA objects are downloaded from S3 and the paths of the SIMA
 objects are reassigned to the local path of the raw data file
 (HDF5 file in step2). Once the paths have been appropriately modified,
 SIMA can be run on the local machine using export_frames() function.

NOTE: the output from the instance while running the script is redirected
 to a local file named analysislog.txt. So users can look at this text
 file to keep track of the printed output.
"""

import sima
import sima.motion
import numpy as np
import math
import datetime
import logging
import os
import time
import sys
from itertools import repeat
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from sima.ROI import ROIList
from multiprocessing import Pool, cpu_count
from shutil import rmtree
from glob import glob

# These global variables between "From Here" and "To Here" define the
# program's intent. Change these depending on the intended use.
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# --------------------------------FROM HERE--------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------

# Since this is executed on the EC2 instance, it is easier to set the
# credentials here compared to reading from the environmental variables
# on the local machine and porting it to the remote instance.
# It is also probably safer since the user is fully aware that the
# keys are in this file and hence, will likely be more responsible
# for handling this info.
AWS_ACCESS_KEY = ''
AWS_SECRET_ACCESS_KEY = ''

# Parameters
# Define data, file type of data ('.tif' or '.h5') and if single file
# (==true if a single tif stack or h5 file represents the whole data;
# ==false if single tiff file for each frame).
s3_bucket_name = ''
s3_source_dir = ''  # Every file in this directory will be analyzed
is_single_file = True

# Declare analysis method and if segmentation is needed.
# 'action' options: motioncorrect, extract, both
action = 'motioncorrect'
to_segment = False  # This value does not affect analysis
# if action == 'motioncorrect'
export_mc_frames_or_not = False  # Should the motion corrected video be
# exported? Yes if True, no if False
if not to_segment:
    roi_filename = 'RoiSet.zip'  # File of ROI masks

dim_order_for_hdf5 = 'tyx'  # dim_order for HDF5 file. It states what the axes
# in your data array represents. So tyx means the data array in the HDF5 file
# is shaped "Time x Y pixels x X pixels". Used in sima.Sequence.create()

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# --------------------------------TO HERE----------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


# Additional global variables. Do not change any of these
base_path_on_ec2 = '/mnt/analysis/DATA/'
# This is the working directory where the analysis script is run.
conn = S3Connection(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)
bucket = conn.get_bucket(s3_bucket_name)


def create_sequence(full_path_to_file, individualframename):
    # Retrieve S3 filename without path
    temp = os.path.splitext(os.path.basename(full_path_to_file))
    filename = temp[0]
    if filename[-4:] == '_CH1':
        dual_channel = True
    else:
        dual_channel = False

    if is_single_file:
        file_format = temp[-1]

        if dual_channel:
            # Identify file format to create SIMA Sequence object
            if file_format == '.tif':
                import_format = 'TIFF'
                sequences1 = sima.Sequence.create(import_format,
                                                  filename +
                                                  file_format)
                sequences2 = sima.Sequence.create(import_format,
                                                  filename[:-1] +
                                                  '2' + file_format)
                sequences = [sima.Sequence.join(sequences1, sequences2)]
            else:  # i.e., if file_format == '.h5':
                import_format = 'HDF5'
                sequences1 = sima.Sequence.create(import_format,
                                                  filename +
                                                  file_format,
                                                  dim_order_for_hdf5)
                sequences2 = sima.Sequence.create(import_format,
                                                  filename[:-1] +
                                                  '2' + file_format,
                                                  dim_order_for_hdf5)
                sequences = [sima.Sequence.join(sequences1, sequences2)]

        else:
            # Identify file format to create SIMA Sequence object
            if file_format == '.tif':
                import_format = 'TIFF'
                sequences = [sima.Sequence.create(import_format,
                                                  os.path.basename(
                                                     full_path_to_file))]
            else:  # i.e., if file_format == '.h5':
                import_format = 'HDF5'
                sequences = [sima.Sequence.create(import_format,
                                                  os.path.basename(
                                                     full_path_to_file),
                                                  dim_order_for_hdf5)]

    else:  # file format is tiff
        import_format = 'TIFFs'
        if dual_channel:
            # 'filename' is really the directory. An individual frame's
            # file name prefix is defined by 'individualframename'
            sequences1 = sima.Sequence.create(import_format,
                                              [[base_path_on_ec2 +
                                                filename + '/' +
                                                individualframename +
                                                '_*.tif']])
            sequences2 = sima.Sequence.create(import_format,
                                              [[base_path_on_ec2 +
                                                filename[:-1] + '2' + '/' +
                                                individualframename +
                                                '_*.tif']])
            sequences = [sima.Sequence.join(sequences1, sequences2)]

        else:
            # 'filename' is really the directory. An individual frame's
            # file name prefix is defined by 'individualframename'
            sequences = [sima.Sequence.create(import_format,
                                              [[base_path_on_ec2 + filename +
                                                '/' + individualframename +
                                                '_*.tif']])]
    sima.ImagingDataset(sequences, filename + '.sima')
    return filename, dual_channel


def motion_correction((full_path_to_file, individualframename)):
    # Create SIMA Sequence object from file(s)
    filename, dual_channel = create_sequence(full_path_to_file,
                                             individualframename)
    print "Starting motion correction for %s" % filename
    init_data = sima.ImagingDataset.load(filename + '.sima')

    # Define motion correction method and run.
    mc_approach = sima.motion.HiddenMarkov2D(granularity='row',
                                             verbose=True,
                                             max_displacement=[50, 100])
    if dual_channel:
        dataset = mc_approach.correct(init_data, filename + '_mc.sima',
                                      channel_names=['1', '2'],
                                      correction_channels=['1'])
    else:
        dataset = mc_approach.correct(init_data, filename + '_mc.sima')
    logging.info("Done motion correction for %s. Saving results" % filename)
    print "Done motion correction for %s. Saving results" % filename
    sys.stdout.flush()

    if dual_channel:
        dataset.export_averages([filename + '_mc_std.tif',
                                 filename[:-1] + '2' + '_mc_std.tif'],
                                projection_type='std')
    else:
        dataset.export_averages([filename + '_mc_std.tif'],
                                projection_type='std')
    # Export motion corrected video as .hdf5 to avoid any memory issues
    if export_mc_frames_or_not:
        if dual_channel:
            dataset.export_frames([filename + '_mc.h5',
                                   filename[:-1] + '2' + '_mc.h5'],
                                  fmt='HDF5')
        else:
            dataset.export_frames([filename + '_mc.h5'],
                                  fmt='HDF5')

    logging.info("Done exporting motion corrected files for %s" % filename)
    print "Done exporting motion corrected files for %s" % filename
    sys.stdout.flush()


def extract_rois((full_path_to_file, individualframename)):
    # Create SIMA Sequence & ImagingDataset objects from image file(s) or
    # motion correction if action == 'extract', assume that motion
    # correction was done on EC2 previously
    if action == 'both':
        try:
            motion_correction((full_path_to_file, individualframename))
        except Exception as e:
            print('Motion correction failed')
            print e
            logging.Exception('Motion correction failed')

    filename = os.path.splitext(os.path.basename(full_path_to_file))[0]
    dataset = sima.ImagingDataset.load(filename + '_mc.sima')

    # Obtain ROIs
    if to_segment:
        logging.info("Segmenting images for %s..." % filename)

        # Automated segmentation
        # Define segmentation method and post processing.
        segment_approach = sima.segment.PlaneNormalizedCuts()
        segment_approach.append(sima.segment.SparseROIsFromMasks())
        segment_approach.append(sima.segment.SmoothROIBoundaries())
        segment_approach.append(sima.segment.MergeOverlapping(threshold=0.5))

        # Apply segmentation to dataset
        rois = dataset.segment(segment_approach)

        logging.info("Done segmenting images for %s" % filename)

        print("Done segmenting images for %s" % filename)
    else:
        logging.info("Importing ROIs from ImageJ for %s..." % filename)
        print("Importing ROIs from ImageJ for %s..." % filename)

        # Load ROIs from ImageJ
        rois = ROIList.load(filename + '_mc_' + roi_filename, fmt='ImageJ')
        dataset.add_ROIs(rois, 'from_ImageJ')

        logging.info("Done importing ROIs for %s" % filename)
        print("Done importing ROIs for %s" % filename)

    # Extract signals from ROIs into numpy file
    signals = dataset.extract(rois)
    extracted_signals = np.asarray(signals['raw'])
    np.save(filename + '_extractedsignals', extracted_signals)

    logging.info("Done extracting signals")
    print("Done extracting signals")


def download_all_files_sequentially(s3_full_filenames):
    k = Key(bucket)

    # Download all files sequentially
    logging.info('Downloading files')
    print('Downloading files')

    for s3_full_filename in s3_full_filenames:
        filename = os.path.basename(s3_full_filename)
        print('Downloading %s' % (filename))
        temp = os.path.splitext(filename)
        if temp[0][-4:] == '_CH1':
            dual_channel = True
        else:
            dual_channel = False

        if is_single_file:
            # Check if the file has already been downloaded before
            # downloading it.
            if not os.path.isfile(filename):
                k.key = s3_full_filename
                k.get_contents_to_filename(filename)
            individualframename = None
            filenames_to_check_extension = s3_full_filenames
            if dual_channel:
                if not os.path.isfile(temp[0][:-1] + '2' + temp[1]):
                    temp1 = os.path.splitext(s3_full_filename)
                    k.key = temp1[0][:-1] + '2' + temp1[1]
                    k.get_contents_to_filename(temp[0][:-1] + '2' + temp[1])
        else:
            # 'filename' is the name of the directory with the multiple
            # files in this case. Create directory on EC2 machine.
            os.system('mkdir -p "%s"' % filename)

            filenames_to_check_extension = []
            # Import each file from S3 directory into corresponding
            # EC2 directory
            for key in bucket.list(prefix=s3_full_filename):
                individualframename = os.path.basename(str(key.name))
                filenames_to_check_extension.append(individualframename)
                if not os.path.isfile(os.path.join(base_path_on_ec2,
                                                   filename,
                                                   individualframename)):
                    key.get_contents_to_filename(
                                 os.path.join(base_path_on_ec2,
                                              filename,
                                              individualframename))
            individualframename = '_'.join(
                                  individualframename.split('_')[:-1])
            # Set the prefix for a frame's filename

            if not check_for_known_extensions(filenames_to_check_extension):
                raise Exception(
                      'Data files do not have a .tif or .h5 extension!')

        # Import ROI masks if appropriate.
        if action == 'extract':
            # download previous motion corrected sima objects
            filename = os.path.splitext(os.path.basename(s3_full_filename))[0]
            s3_path = os.path.dirname(s3_full_filename)
            os.system('mkdir -p "%s_mc.sima"' % filename)
            for key in bucket.list(prefix=os.path.join(s3_path,
                                                       filename +
                                                       '_mc.sima')):
                temp = os.path.basename(str(key.name))
                if temp.split('/')[-1]:
                    # remove the empty key returned by bucket.list
                    key.get_contents_to_filename(os.path.join(
                                                    base_path_on_ec2,
                                                    filename + '_mc.sima',
                                                    temp))
            if not to_segment:
                # also download RoiSet.zip if rois need to be extracted
                k.key = os.path.join(s3_path, 'Results',
                                     filename + '_mc_' + roi_filename)
                k.get_contents_to_filename(os.path.join(base_path_on_ec2,
                                           filename + '_mc_' +
                                           roi_filename))

    if not check_for_known_extensions(filenames_to_check_extension):
        raise Exception('Data files do not have a .tif or .h5 extension!')

    logging.info('Done downloading files')
    print('Done downloading files')
    return individualframename


def upload_all_files_sequentially(s3_full_filenames):
    k = Key(bucket)

    # Upload all files sequentially
    logging.info('Uploading files')
    print('Uploading files')

    for s3_full_filename in s3_full_filenames:
        if action == 'motioncorrect' or action == 'both':
            upload_mc_files(s3_full_filename)
        elif action == 'extract' or action == 'both':
            s3_path, filename = upload_mc_sima_objects(s3_full_filename)
            k.key = os.path.join(s3_path, 'Results',
                                 filename + '_extractedsignals.npy')
            k.set_contents_from_filename(filename + '_extractedsignals.npy')
            # Upload the empty text file showing that both motion correction
            # and extraction have been performed for this file. We will also
            # delete the previous motion correction status file.
            os.system('touch "%s_mc_extract.txt"' % filename)
            k.key = os.path.join(s3_path, filename + '_mc_extract.txt')
            k.set_contents_from_filename(filename + '_mc_extract.txt')
            k.key = os.path.join(s3_path, filename + '_mc.txt')
            bucket.delete_key(k)

    logging.info('Done uploading')
    print('Done uploading')


def upload_mc_files(s3_full_filename):
    s3_path, filename = upload_mc_sima_objects(s3_full_filename)
    if filename[-4:] == '_CH1':
        dual_channel = True
    else:
        dual_channel = False

    k = Key(bucket)
    k.key = os.path.join(s3_path, 'Results', filename + '_mc_std.tif')
    k.set_contents_from_filename(filename + '_mc_std.tif')
    # Upload the empty text file showing that motion correction
    # has been performed for this file
    os.system('touch "%s_mc.txt"' % filename)
    k.key = os.path.join(s3_path, filename + '_mc.txt')
    k.set_contents_from_filename(filename + '_mc.txt')
    if dual_channel:
        k.key = os.path.join(s3_path, 'Results',
                             filename[:-1] + '2' + '_mc_std.tif')
        k.set_contents_from_filename(filename[:-1] + '2' + '_mc_std.tif')
        # Upload the empty text file showing that motion correction
        # has been performed for this file
        os.system('touch "%s2_mc.txt"' % filename[:-1])
        k.key = os.path.join(s3_path, filename[:-1] + '2' + '_mc.txt')
        k.set_contents_from_filename(filename[:-1] + '2' + '_mc.txt')

    if export_mc_frames_or_not:
        file_path_on_ec2 = os.path.join(base_path_on_ec2, filename + '_mc.h5')
        if os.stat(file_path_on_ec2).st_size < 5000 * 1024 * 1024:
            # if file size less than 5GB, use single part upload
            name_on_s3 = os.path.splitexit(s3_full_filename)[0]
            k.key = name_on_s3 + '_mc.h5'
            k.set_contents_from_filename(filename + '_mc.h5')
            if dual_channel:
                k.key = name_on_s3[:-1] + '2' + '_mc.h5'
                k.set_contents_from_filename(filename[:-1] + '2' + '_mc.h5')
        else:
            upload_file(file_path_on_ec2)
            if dual_channel:
                temp = os.path.join(base_path_on_ec2,
                                    filename[:-1] + '2' + '_mc.h5')
                upload_file(temp)


def upload_mc_sima_objects(s3_full_filename):
    k = Key(bucket)
    s3_path = os.path.dirname(s3_full_filename)
    filename = os.path.splitext(os.path.basename(s3_full_filename))[0]
    for path, directory, files in os.walk(os.path.join(base_path_on_ec2,
                                                       filename +
                                                       '_mc.sima')):
        for afile in files:
            relpath = os.path.relpath(os.path.join(path, afile))
            logging.info('Uploading ' + relpath)
            print('Uploading ' + relpath)
            k.key = os.path.join(s3_path, relpath)
            k.set_contents_from_filename(relpath)
    return s3_path, filename


# The following function does multi-part upload to circumvent the 5GB limit
# for uploading to S3. It chunks the file into 5GB parts for a multi-part
# upload in boto, stores the location of the read/write pointer of the last
# uploaded chunk and resumes from there iteratively until the whole file is
# uploaded.
def upload_file(file_path_on_ec2):
    filename = os.path.basename(file_path_on_ec2)

    mp = bucket.initiate_multipart_upload(filename)

    sourcesize = os.stat(file_path_on_ec2).st_size
    bytes_per_chunk = 5000 * 1024 * 1024
    num_chunks = int(math.ceil(sourcesize / float(bytes_per_chunk)))

    for i in range(num_chunks):
        offset = i * bytes_per_chunk
        remainingbytes = sourcesize - offset
        size = min([bytes_per_chunk, remainingbytes])
        part_num = i + 1

        logging.info("uploading part " + str(part_num) + " of " +
                     str(num_chunks))
        print("uploading part " + str(part_num) + " of " + str(num_chunks))

        with open(file_path_on_ec2, 'r') as fp:
            fp.seek(offset)
            mp.upload_part_from_file(fp=fp, part_num=part_num, size=size)

    if len(mp.get_all_parts()) == num_chunks:
        mp.complete_upload()
        logging.info("Multipart upload done")
        print("Multipart upload done")
    else:
        mp.cancel_upload()
        logging.info("Multipart upload failed")
        print("Multipart upload failed")


def delete_previous_sima_directories():
    # Remove all directories ending with .sima - error encountered
    # if directory already exists.
    path = os.getcwd()
    pattern = os.path.join(path, "*.sima")
    for item in glob(pattern):
        if os.path.isdir(item):
            rmtree(item)


def check_for_known_extensions(filenames):
    accepted_extensions = {'.tif', '.h5'}
    appropriate_extension = True
    for filename in filenames:
        file_format = os.path.splitext(filename)[-1]
        if file_format not in accepted_extensions:
            appropriate_extension = False
            break
    return appropriate_extension


def get_filenames_to_analyze():
    # This function gets the names of all files that need to be analyzed.
    # This function checks all the files in the source_dir and classifies
    # them as one with .txt extension or with .tif or .h5 extensions.
    # If present, the .txt files are used to set the status of previous
    # analysis on the files. For instance, if the text file has a name
    # filename_mc.txt, it is assumed that the file with name filename.h5
    # or filename.tif has been motion corrected. So this file is not going
    # to be included for analysis if action==motioncorrect. Similarly, if
    # the text file has a name filename_mc_extract.txt, it is assumed that
    # the file with name filename.h5 or filename.tif has been motion corrected
    # and the ROIs have been extracted. So this file is not going to be
    # included for analysis if action==motioncorrect or if action==extract.

    s3_full_filenames_temp = []
    # List of all potential filenames to be analyzed
    s3_full_filenames_done = []
    # List of all filenames that have been analyzed.
    # These are the ones where with .txt extensions. The filenames to analyze
    # are the ones in s3_full_filenames_temp but not in s3_full_filenames_done
    for key in bucket.list(prefix=s3_source_dir+'/'):
        temp = str(key.name)
        if (temp.split('/')[-1] and
           len(temp.split('/')) == len(s3_source_dir.split('/')) + 1 and
           os.path.splitext(temp)[1] != '.txt'):
            # filenames to analyze don't include files more than one
            # level deep in the source_dir
            s3_full_filenames_temp.append(temp)
        if (temp.split('/')[-1] and
           len(temp.split('/')) == len(s3_source_dir.split('/')) + 1 and
           os.path.splitext(temp)[1] == '.txt'):
            # filenames that are done have an associated .txt file
            s3_full_filenames_done.append(temp)

    if action == 'motioncorrect' or action == 'both':
        s3_full_filenames = [filename for filename in s3_full_filenames_temp
                             if ((os.path.splitext(filename)[0] + '_mc.txt'
                                 not in s3_full_filenames_done) and
                                 (os.path.splitext(filename)[0] +
                                  '_mc_extract.txt' not in
                                  s3_full_filenames_done))]
    elif action == 'extract':
        s3_full_filenames = [filename for filename in s3_full_filenames_temp
                             if (os.path.splitext(filename)[0] +
                                 '_mc_extract.txt' not in
                                 s3_full_filenames_done)]
    # Now check for dual channel files. If any file ends with _CH1, there
    # should be a corresponding _CH2. After making sure of this, remove the
    # _CH2 file from this list since it will be taken care of in the later
    # functions.
    for afile in s3_full_filenames:
        temp = os.path.splitext(afile)
        basename = temp[0]
        file_format = temp[-1]
        if basename[-4:] == '_CH1':
            if not basename[:-1] + '2' + file_format in s3_full_filenames:
                raise Exception('CH2 file not found for %s' % afile)
            s3_full_filenames.remove(basename[:-1] + '2' + file_format)
    return s3_full_filenames


# This is just a debugging function. Not needed for analysis
def ping_for_a_while(num_minutes, start_time):
    temp_start_time = start_time
    while (time.time() - start_time) < num_minutes*60:
        if (time.time() - temp_start_time) > 30:
            print('%s seconds later!' % (time.time() - start_time))
            sys.stdout.flush()
            temp_start_time = time.time()


def main():
    accepted_actions = {'motioncorrect', 'extract', 'both'}
    if action not in accepted_actions:
        raise Exception('Action not recognized!')

    s3_full_filenames = get_filenames_to_analyze()
    # temp = get_filenames_to_analyze()
    # s3_full_filenames = [f for f in temp if 'D4' in f]

    if not s3_full_filenames:
        raise Exception('No file(s) to analyze')

    print('Files to analyze: %s' % ('\n'.join(s3_full_filenames)))

    print('Number of cores available: %d, number of files being analyzed: %d'
          % (cpu_count(), len(s3_full_filenames)))

    print('You are using %d %% of your available processing power'
          % (100*len(s3_full_filenames)/cpu_count()))

    sys.stdout.flush()

    start_time = time.time()

    delete_previous_sima_directories()

    individualframename = download_all_files_sequentially(s3_full_filenames)
    # individualframename is the prefix of an individual frame's name if
    # the data is stored in multiple tiff files.
    # It's set to none for a single file representing the data.
    # For the other case with multiple tiff files, one for each frame,
    # this will be calculated while downloading the files from S3.
    # This variable is used for the SIMA function that loads images.

    # Run operation in parallel
    print "Running action: " + action
    logging.info("Running action: %s", action)

    num_processes = np.minimum(len(s3_full_filenames), cpu_count())
    print "Number of processes initiated = %d" % (num_processes)
    p = Pool(num_processes)
    if action == 'motioncorrect':
        p.map(motion_correction, zip(s3_full_filenames,
              repeat(individualframename)))
    elif action == 'extract' or action == 'both':
        # extract_rois performs motioncorrection prior to extraction if
        # action == both
        p.map(extract_rois, zip(s3_full_filenames,
              repeat(individualframename)))
    logging.info('Done running action: %s!' % (action))
    p.close()
    p.join()
    print('Done running action: %s! Now uploading results to S3' % (action))
    sys.stdout.flush()

    upload_all_files_sequentially(s3_full_filenames)

    print("----Time taken: %s seconds----" % (time.time()-start_time))

if __name__ == '__main__':

    # Set the current working directory.
    # If this is /mnt/DATA, this directory has already been created
    # by the set up program that initiates the EC2 instance
    os.system('mkdir -p "%s"' % base_path_on_ec2)
    os.chdir(base_path_on_ec2)
    if os.path.isfile('/tmp/analysisfailed.txt'):
        os.system('rm /tmp/analysisfailed.txt')
    if os.path.isfile('/tmp/analysissuccess.txt'):
        os.system('rm /tmp/analysissuccess.txt')

    # Create log file
    there_was_any_error = False
    log_filename = 'logfile' + datetime.datetime.now().isoformat(' ') + '.txt'
    logging.getLogger('').handlers = []
    logging.basicConfig(filename=log_filename,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s \
                                %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.info("Logging starts here:")

    # Run analysis
    try:
        main()
        logging.info("Everything worked! Hurray!")
        print("Everything worked! Hurray!")
    except Exception as e:
        logging.exception(
                "The following error occured during the analysis script:")
        there_was_any_error = True
        print("The following error occured during the analysis script:")
        print e

    # Upload logs
    k = Key(bucket)
    k.key = 'logfiles/' + log_filename
    k.set_contents_from_filename(log_filename)

    # We are handling server drops in a clunky way by writing one of two
    # status files below indicating whether the analysis successfully
    # completed or failed. This is because if the instance drops its
    # connection for even a little bit, the loop checking for exit flag
    # from the instance in ConnectToEC2.py hangs indefinitely. Thus, in
    # this case, we created a solution  by timing out the connection every
    # once in a while and reconnecting. This way, if the instance drops
    # its connection but the analysis actually completes, the following
    # files get written. So then when the reconnection to the instance
    # happens, we can check for the existence of these files to know
    # if the analysis completed prior to the time out.

    # The elif clause is added because if the /tmp/runanalysis socket
    # gets deleted somehow (may be intentionally by the user), the try
    # catch loop above calling main() exits without an error and it would
    # appear as if the analysis successfully completed, when in fact, it did
    # not.
    if there_was_any_error:
        os.system('touch /tmp/analysisfailed.txt')
        raise Exception('Analysis failed. Check log file')
    elif os.path.exists('/tmp/runanalysis'):
        os.system('touch /tmp/analysissuccess.txt')
