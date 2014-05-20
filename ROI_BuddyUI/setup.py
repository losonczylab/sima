from setuptools import setup
# import py2exe

setup(
    name="ROIBuddy",
    version="0.1.0",
    author_email="software@losonczylab.org",
    url="http://www.losonczylab.org/roibuddy",
    license="GNU General Public License (GPL)",
    packages=['ROIBuddy'],
    package_dir={'ROIBuddy': '.'},
    # packages=['liftr'],
    # package_data={"liftr": ["ui/*"]},
    # scripts=["bin/liftr"],
    # windows=[{"script": "bin/liftr"}],
    # options={"py2exe": {"skip_archive": True, "includes": ["sip"]}},
    scripts=['ROIBuddy']
)
