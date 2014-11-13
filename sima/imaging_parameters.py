import os
from xml.etree import ElementTree
from distutils.version import LooseVersion


def prairie_imaging_parameters(filepath):

    def _get_prairieview_version(xml_filepath):
        """Return Prairieview version number"""
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
                            field = indexedValue.get('key')
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


def extract_imaging_parameters(filepath, format):
    """Get the imaging parameters used during the session

    Parameters
    ----------
    filepath : string
        The path to the file which contains the parameters
    format : {'prairie'}
        The format of the parameters file.

    Returns
    -------
    dictionary
        A dictionary of imaging parameters and values

    """

    if format == 'Prairie':
        params = prairie_imaging_parameters(filepath)
    return params
