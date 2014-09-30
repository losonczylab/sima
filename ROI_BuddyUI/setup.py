from setuptools import setup
setup(
    name='ROIBuddy',
    version='0.2.1',
    author_email='software@losonczylab.org',
    url='http://www.losonczylab.org/sima/roibuddy',
    license='GNU General Public License (GPL)',
    packages=['ROIBuddy'],
    package_dir={'ROIBuddy': '.'},
    entry_points={'gui_scripts': ['roibuddy = ROIBuddy.roi_buddy:main']}
)
