#NOTE: NEED TO COPY .DLLs FROM SHAPELY INTO THE DIST DIRECTORY AFTER BUILDING
from guidata import disthelpers as dh
from shutil import copy, copytree
from os.path import join

TARGET_DIR = './dist'

INCLUDES = ['numpy', 'cPickle', 'sima', 'sima.iterables', 'sima.ROI', 'sima.segment', 'sima.imaging',
            'scipy', 'scipy.cluster.vq', 'scipy.misc', 'scipy.ndimage',
            'scipy.ndimage.filters', 'skimage', 'skimage.color', 'skimage.draw',
            'skimage.feature', 'skimage.filter', 'skimage.measure', 'skimage.morphology',
            'skimage._shared.interpolation',
            'skimage.filter.rank.core_cy', 'scipy.special._ufuncs_cxx', 'skimage._shared.geometry',
            'scipy.sparse.csgraph._validation', 'skimage._shared.transform',
            'skimage.transform', 'sip', 'PyQt4', 'PyQt4.QtSvg', 'guidata', 'guiqwt']

dist = dh.Distribution()

dist.setup(name='ROI_Buddy',
           version='0.1',
           description='ROI Buddy GUI',
           script='roi_buddy.py',
           target_name='ROI_Buddy.exe',
           target_dir=TARGET_DIR,
           icon='icon.ico',
           includes=INCLUDES)

dist.add_modules('guidata', 'guiqwt')
dist.build_py2exe()

copy('geos.dll', TARGET_DIR)
copy('geos_c.dll', TARGET_DIR)
copytree('icons', join(TARGET_DIR, 'icons'))
