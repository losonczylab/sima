import sima
import sima.motion
from sima.iterables import MultiPageTIFF
from matplotlib.pyplot import plot, show
import shutil

CORRECT = True
SEGMENT = True
RECORD_PATH = '/home/patrick/tmp-test-data/TSeries-Loc6-control-006/test3.sima'
if CORRECT:
    try:
        shutil.rmtree(RECORD_PATH)
    except:
        pass

    # CONSTRUCT ITERABLES
    d = '/home/patrick/tmp-test-data/TSeries-Loc6-control-006/'
    b = d + 'TSeries-Loc6-control-006_Cycle000'
    filenames = [
        [b + '01_CurrentSettings_Ch1.tif', b + '01_CurrentSettings_Ch2.tif'],
        [b + '02_CurrentSettings_Ch1.tif', b + '02_CurrentSettings_Ch2.tif'],
        [b + '03_CurrentSettings_Ch1.tif', b + '03_CurrentSettings_Ch2.tif'],
        [b + '04_CurrentSettings_Ch1.tif', b + '04_CurrentSettings_Ch2.tif'],
        [b + '05_CurrentSettings_Ch1.tif', b + '05_CurrentSettings_Ch2.tif'],
        [b + '06_CurrentSettings_Ch1.tif', b + '06_CurrentSettings_Ch2.tif'],
        [b + '07_CurrentSettings_Ch1.tif', b + '07_CurrentSettings_Ch2.tif'],
        [b + '08_CurrentSettings_Ch1.tif', b + '08_CurrentSettings_Ch2.tif'],
        [b + '09_CurrentSettings_Ch1.tif', b + '09_CurrentSettings_Ch2.tif'],
        [b + '10_CurrentSettings_Ch1.tif', b + '10_CurrentSettings_Ch2.tif'],
        [b + '11_CurrentSettings_Ch1.tif', b + '11_CurrentSettings_Ch2.tif'],
        [b + '12_CurrentSettings_Ch1.tif', b + '12_CurrentSettings_Ch2.tif'],
        [b + '13_CurrentSettings_Ch1.tif', b + '13_CurrentSettings_Ch2.tif'],
        [b + '14_CurrentSettings_Ch1.tif', b + '14_CurrentSettings_Ch2.tif'],
        [b + '15_CurrentSettings_Ch1.tif', b + '15_CurrentSettings_Ch2.tif']
    ]
    clip = ((0, 0), (20, 0))
    iterables = [
        [MultiPageTIFF(chan, clip) for chan in cycle]
        for cycle in filenames
    ]

    dataset = sima.motion.hmm(
        iterables, RECORD_PATH, ['tdTomato', 'GCaMP'],
        30, [20, 30], trim_criterion=0.95
    )

else:
    dataset = sima.ImagingDataset.load(RECORD_PATH)

if SEGMENT:
    dataset.segment()

""" USE IMAGING BUDDY"""

dataset = sima.ImagingDataset.load(RECORD_PATH)
dataset.extract(signal_channel=0)

"""
plot(dataset.signals[0])
show()
"""
