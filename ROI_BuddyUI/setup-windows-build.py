#NOTE: NEED TO COPY .DLLs FROM SHAPELY INTO THE DIST DIRECTORY AFTER BUILDING
"""
Setup script for building the ROI Buddy GUI

Things to note:
-- This should be build from a machine with a patched version of guiqwt.baseplot.py:
        def add_item_with_z_offset(self, item, zoffset):
        ""
        Add a plot *item* instance within a specified z range, over *zmin*
        ""
        zlist = sorted([_it.z() for _it in self.items
                        if _it.z() >= zoffset]+[zoffset-1])
        dzlist = np.argwhere(np.diff(zlist) > 1)
        if len(dzlist) == 0:
            z = max(zlist)+1
        else:
            z = zlist[dzlist[0,0]]+1
        self.add_item(item, z=z)

-- Due to a shapely error, the geos.dll and geos_c.dll files need to be copied into dist manually


--CTYPES PATCH HERE
"""

from guidata import disthelpers as dh
from shutil import copy, copytree
from os.path import join
import zipfile as zf

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
           version='0.2.0',
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
copy('libtiff.dll', TARGET_DIR)
#copy('tiff.h', TARGET_DIR)
copytree('icons', join(TARGET_DIR, 'icons'))

f = zf.ZipFile(join('dist', 'library.zip'), 'a')
f.write('tiff_h_4_0_3.py', join('libtiff', 'tiff_h_4_0_3.py'))
f.write('tiff_h_4_0_3.pyo', join('libtiff', 'tiff_h_4_0_3.pyo'))
f.write('tiff_h_4_0_3.pyc', join('libtiff', 'tiff_h_4_0_3.pyc'))
f.close()
