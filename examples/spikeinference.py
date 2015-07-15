from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range

from scipy import signal
from scipy.stats import uniform, norm
import numpy as np
import seaborn as sns
import matplotlib.mlab as ml
import matplotlib.pyplot as plt

from sima import spikes

#########
# PART 1: Make model calcium data
#########

# Data parameters
RATE = 1       # mean firing rate of poisson spike train (Hz)
STEPS = 5000   # number of time steps in data
TAU = 0.6      # time constant of calcium indicator (seconds)
DELTAT = 1 / 30  # time step duration (seconds)
SIGMA = 0.1    # standard deviation of gaussian noise
SEED = 2222    # random number generator seed
NTRACE = 5     # number of data traces to generate

# Make a poisson spike trains
SPIKES = [spikes.get_poisson_spikes(deltat=DELTAT, rate=RATE,
                                    steps=STEPS, seed=SEED + i)
          for i in range(NTRACE)]
SPIKES = np.asarray(SPIKES)

# Convolve with kernel to make calcium signal
np.random.seed(SEED)
GAMMA = 1 - (DELTAT / TAU)
CALCIUM = signal.lfilter([1], [1, -GAMMA], SPIKES)
TIME = np.linspace(0, STEPS * DELTAT, STEPS)

# Make fluorescence traces with random gaussian noise and baseline
FLUORS = [CALCIUM[i, ] + norm.rvs(scale=SIGMA, size=STEPS) + uniform.rvs()
          for i in range(NTRACE)]
FLUORS = np.asarray(FLUORS)

#########
# PART 2:  Estimate model parameters and perform spike inference
#########

# Perform spike inference on all simulated fluorescence traces
INFERENCE = np.zeros([STEPS, NTRACE])
FITS = np.zeros([STEPS, NTRACE])

# Jointly estimate gamma on traces concatenated together
[joint_gamma_est, joint_sigma_est] = spikes.estimate_parameters(
    FLUORS.reshape(FLUORS.size), mode="correct")

for x in range(NTRACE):

    # Estimate noise and decay parameters
    [gamma_est, sigma_est] = spikes.estimate_parameters(
        FLUORS[x, ], mode="correct", gamma=joint_gamma_est)
    print("tau = {tau},  sigma = {sigma}".format(
        tau=DELTAT / (1 - gamma_est), sigma=sigma_est))

    # Run spike inference
    INFERENCE[:, x], FITS[:, x], params = spikes.spike_inference(
        FLUORS[x, ], sigma=sigma_est, gamma=joint_gamma_est, verbose=True)

#########
# PART 3: Plot results
#########

# Close all open figures
plt.close("all")

# Set up plotting style
sns.set(context="talk", rc={"figure.figsize": [20, 6]}, style="white")
sns.set_palette("muted", desat=.6)
tck = [0, .5, 1]
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, facecolor='w')

# Which cell to plot in first figure
cp = 0

# Plot the simulated data and model fit from the first result
plt.axes(ax1)
sns.tsplot(FLUORS[cp, ], ax=ax1, time=TIME)
sns.tsplot(FITS[:, cp], ax=ax1, time=TIME, color="red")
ax1.set_ylabel("Data and Fit")
plt.yticks(
    np.round([FLUORS[cp].min(), FLUORS[cp].mean(), FLUORS[cp].max()], 1))

# Plot the true spike train
plt.axes(ax2)
plt.bar(TIME, SPIKES[cp, ], color="DimGray", width=DELTAT)
ax2.set_ylabel("True Spikes")
plt.yticks(tck)
plt.ylim(-.1, 1.1)

# Get true positives and false positives
spike_cutoff = 0.1
i_times = ml.find(INFERENCE[:, cp] > spike_cutoff)  # inferred spikes
t_times = ml.find(SPIKES[cp, :])  # true spikes
sInds = np.intersect1d(i_times, t_times)  # indices of true positives
wInds = np.setdiff1d(i_times, t_times)   # indices of false positives
tp = float(sInds.size) / float(i_times.size)  # true positive rate
fp = float(wInds.size) / \
    (STEPS - float(t_times.size))  # false positive rate

# Plot the spike inference
plt.axes(ax3)
plt.bar(
    TIME[sInds], np.ones(sInds.size),
    color="LightGrey", edgecolor="LightGrey", width=DELTAT)
plt.bar(
    TIME[wInds], np.ones(wInds.size),
    color="Red", edgecolor="Red", width=DELTAT)
plt.bar(
    TIME, INFERENCE[:, 0] / INFERENCE[:, 0].max(),
    color="DimGray", edgecolor="DimGray", width=DELTAT)
ax3.set_xlabel("Time (Seconds)")
ax3.set_ylabel("Spike Inference")
sns.despine(bottom=True, left=True)
plt.yticks(tck)
plt.ylim(-.1, 1.1)
plt.title(
    "TP rate = " + str(round(tp, 2)) + "; FP rate = " + str(round(fp, 2)))

# Plot all traces and inference
plt.figure(5, facecolor='w')
plt.subplot(211)
plt.imshow(FLUORS, aspect="auto", interpolation="none")
plt.colorbar()
plt.subplot(212)
plt.imshow(INFERENCE.transpose(), aspect="auto", interpolation="none")
plt.colorbar()

plt.show()
