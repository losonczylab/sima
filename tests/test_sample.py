from os.path import join, basename, dirname
from unittest import TestCase
import cPickle
import numpy as np

import sima.motion
from sima.iterables import MultiPageTIFF

class MotionCorrectionTest(TestCase):
    def test(self):
        RECORRECT = True

        d = '/home/patrick/tmp-test-data/TSeries-Loc6-control-006/'
        b = d + 'TSeries-Loc6-control-006_Cycle000'

        with open('test.sima/displacements_.pkl', 'rb') as f:
            old_displacements = cPickle.load(f)

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
        if RECORRECT:
            sima.motion.hmm(
                iterables, 'test.sima', 30, artifact_channels=[0], verbose=False
            )
        """
        else:
            with open('displacements.pkl', 'rb') as f:
                saved_data = cPickle.load(f)
            save_filenames = [[join(dirname(chan), 'corrected', basename(chan))
                              for chan in cycle] for cycle in filenames]
            time_avg_names = [join(dirname(chan), 'corrected', basename(chan))
                              for chan in filenames[0]]
        """

        with open('test.sima/displacements.pkl', 'rb') as f:
            displacements = cPickle.load(f)
        min_displacement = np.amin(
            [x.min(axis=0) for x in displacements], axis=0)
        displacements = [x - min_displacement for x in displacements]

        min_displacement = np.amin(
            [x.min(axis=0) for x in old_displacements], axis=0)
        old_displacements = [x - min_displacement for x in old_displacements]
        """
        for x, y in zip(pkl_data_test['shifts'], pkl_data['shifts']):
            assert np.all(x == y)
        print 'checked shifts'

        for x, y in zip(pkl_data_test['references'], pkl_data['references']):
            np.allclose(x[np.isfinite(x)], y[np.isfinite(y)])
        print 'checked references'

        for s in ('gains', 'cov_matrix_est', 'max_displacements'):
            assert np.allclose(pkl_data_test[s], pkl_data[s])
            print 'checked', s

        """
        i = 0
        for x, y in zip(displacements, old_displacements):
            assert np.allclose(x, y)
            i += 1
        print 'checked displacements'
