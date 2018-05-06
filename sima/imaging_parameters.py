from __future__ import division
import os
from xml.etree import ElementTree
from distutils.version import LooseVersion
import numpy as np

from .misc import loadmat


def prairie_imaging_parameters(filepath):
    """Parse imaging parameters from Prairie/Bruker."""
    def _get_prairieview_version(xml_filepath):
        """Return Prairieview version number."""
        for _, elem in ElementTree.iterparse(xml_filepath, events=("start",)):
            if elem.tag == 'PVScan':
                return LooseVersion(elem.get('version'))

    if filepath.endswith('.tif'):
        xml_filename = os.path.basename(
            os.path.realpath(filepath)).split('_')[0] + '.xml'
        directory = os.path.dirname(os.path.realpath(filepath))
        filepath = os.path.join(directory, xml_filename)
    assert filepath.endswith('.xml')

    prairieview_version = _get_prairieview_version(filepath)

    if prairieview_version >= LooseVersion('5.2'):
        params = {}
        for _, elem in ElementTree.iterparse(filepath, events=("start",)):
            if elem.tag == 'PVStateShard':
                for key in elem.findall('PVStateValue'):
                    if len(key.findall('IndexedValue')):
                        k = key.get('key')
                        params[k] = {}
                        for indexedValue in key.findall('IndexedValue'):
                            field = indexedValue.get('index')
                            value = indexedValue.get('value')
                            try:
                                params[k][field] = float(value)
                            except ValueError:
                                params[k][field] = value
                    elif len(key.findall('SubindexedValues')):
                        k = key.get('key')
                        params[k] = {}
                        for subindexedValue in key.findall('SubindexedValues'):
                            i = subindexedValue.get('index')
                            params[k][i] = {}
                            for subval in subindexedValue.findall(
                                    'SubindexedValue'):
                                if subval.get('description', None):
                                    field = subval.get('description')
                                else:
                                    field = subval.get('subindex')
                                value = subval.get('value')
                                try:
                                    params[k][i][field] = float(value)
                                except ValueError:
                                    params[k][i][field] = value
                    else:
                        field = key.get('key')
                        value = key.get('value')
                        try:
                            params[field] = float(value)
                        except ValueError:
                            params[field] = value
                break
    else:
        params = {}
        for _, elem in ElementTree.iterparse(filepath, events=("start",)):
            if elem.tag == 'PVStateShard':
                for key in elem.findall('Key'):
                    field = key.get('key')
                    value = key.get('value')
                    try:
                        params[field] = float(value)
                    except ValueError:
                        params[field] = value
                break
    return params


def scanbox_imaging_parameters(filepath):
    """Parse imaging parameters from Scanbox.

    Based off of the sbxRead Matlab implementation and
    https://scanbox.org/2016/09/02/reading-scanbox-files-in-python/

    """
    assert filepath.endswith('.mat')
    data_path = os.path.splitext(filepath)[0] + '.sbx'
    info = loadmat(filepath)['info']

    # Fix for old scanbox versions
    if 'sz' not in info:
        info['sz'] = np.array([512, 796])

    if 'scanmode' not in info:
        info['scanmode'] = 1  # unidirectional
    elif info['scanmode'] == 0:
        info['recordsPerBuffer'] *= 2  # bidirectional

    if info['channels'] == 1:
        # both PMT 0 and 1
        info['nchannels'] = 2
        # factor = 1
    elif info['channels'] == 2 or info['channels'] == 3:
        # PMT 0 or 1
        info['nchannels'] = 1
        # factor = 2

    # Bytes per frame (X * Y * C * bytes_per_pixel)
    info['nsamples'] = info['sz'][1] * info['recordsPerBuffer'] * \
        info['nchannels'] * 2

    # Divide 'max_idx' by the number of plane to get the number of time steps
    if info.get('scanbox_version', -1) >= 2:
        # last_idx = total_bytes / (Y * X * 4 / factor) - 1
        # info['max_idx'] = os.path.getsize(data_path) // \
        #     info['recordsPerBuffer'] // info['sz'][1] * factor // 4 - 1
        info['max_idx'] = os.path.getsize(data_path) // info['nsamples'] - 1
    else:
        if info['nchannels'] == 1:
            factor = 2
        elif info['nchannels'] == 2:
            factor = 1
        info['max_idx'] = os.path.getsize(data_path) \
            // info['bytesPerBuffer'] * factor - 1

    # Check optotune planes
    if ('volscan' in info and info['volscan'] > 0) or \
       ('volscan' not in info and len(info.get('otwave', []))):
        info['nplanes'] = len(info['otwave'])
    else:
        info['nplanes'] = 1

    return info


def extract_imaging_parameters(filepath, format):
    """Get the imaging parameters used during the session.

    Parameters
    ----------
    filepath : string
        The path to the file which contains the parameters
    format : {'Prairie', 'Scanbox'}
        The format of the parameters file.
    **kwargs
        Additional keyword arguments are passed to the specific imaging
        parameter parser.

    Returns
    -------
    dictionary
        A dictionary of imaging parameters and values

    """
    if format == 'Prairie':
        params = prairie_imaging_parameters(filepath)
    elif format == 'Scanbox':
        params = scanbox_imaging_parameters(filepath)
    return params
