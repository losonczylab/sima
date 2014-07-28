import os
from xml.etree import ElementTree


def prairie_imaging_parameters(filepath):
    if filepath.endswith('.tif'):
        xml_filename = os.path.basename(
            os.path.realpath(filepath)).split('_')[0] + '.xml'
        directory = os.path.dirname(os.path.realpath(filepath))
        filepath = os.path.join(directory, xml_filename)

    assert filepath.endswith('.xml')

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
