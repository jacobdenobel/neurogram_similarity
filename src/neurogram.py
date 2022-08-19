import os
from itertools import combinations
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity


DATA_DIR = os.path.join(
    os.path.realpath(os.path.dirname(os.path.dirname(__file__))), "data"
)
DROPBOX_DIR = os.path.join(DATA_DIR, "dropbox")

KERNELS = [
    ("FULL", 0, 1.0),
    ("TFS", 32, 1),  # TFS
    ("ENV", 128, 0.1),  # ENV
]


def load_spike_matrix(query: str) -> np.ndarray:
    folder = os.path.join(DROPBOX_DIR, "spike_matrices")
    matches = glob(f"{folder}/*{query}*")
    if not any(matches):
        raise FileNotFoundError(f"{query} not found in {DROPBOX_DIR}")
    path, *_ = matches[::-1]
    return np.load(path, allow_pickle=True)


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)


def smooth(neurogram: np.ndarray, window_size=32, overlap=0.5) -> np.ndarray:
    if not window_size:
        return neurogram

    padding = np.zeros(int(window_size * overlap * 0.5))
    window = np.r_[padding, np.hamming(window_size), padding]
    return np.vstack(
        [np.convolve(ni, window, mode="same") / window.sum() for ni in neurogram]
    )


def rebin(x: np.ndarray, factor: float = 0.1):
    shape = x.shape[0], int(x.shape[1] * factor)
    sh = shape[0], x.shape[0] // shape[0], shape[1], x.shape[1] // shape[1]
    return x.reshape(sh).sum(-1).sum(1)

def plot_spectrogram(x: np.ndarray, ax: plt.axis = None, dt=100e-6) -> None:
    if ax is None:
        _, ax = plt.subplots()
    n = x.shape[1]
    t = np.linspace(0, n * dt, n)
    y = np.arange(0, 32000, 100)

    img = ax.pcolormesh(t, y, x, cmap="cividis")
    ax.set_ylabel("fiber id")
    ax.set_xlabel("time [s]")
    return img


def transform(psth, factor, window_size):
    data = rebin(psth, factor)
    data = standardize(data)
    data = smooth(data, window_size)
    return normalize(data) * 255


def show_neurograms(query: str):
    psth = load_spike_matrix(query)
    f, axes = plt.subplots(1, len(KERNELS), figsize=(20, 5), sharey=True)
    f.suptitle(query)
    for (name, window_size, binsize), ax in zip(KERNELS, axes.ravel()):
        ax.set_title(f"{name} window: {window_size} bin: {int(10/binsize)}" + r"$\mu$s")
        data = transform(psth, binsize, window_size)
        im = plot_spectrogram(data, ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(im, cax=cax, orientation="vertical")


if __name__ == "__main__":
    correct_files = [
        "2022-08-10_12h57 spike_matrix_smrt_rate_5_depth_20_width_3p0_phase_0.npy",
        "2022-08-10_13h37 spike_matrix_smrt_rate_5_depth_20_width_2p0_phase_0.npy",
        "2022-08-12_14h57 spike_matrix_smrt_rate_5_depth_20_width_5p0_phase_0.npy",
        "2022-08-16_14h54 spike_matrix_smrt_rate_5_depth_20_width_1p4_phase_0.npy",
    ]

    neurograms = dict()
    for f in correct_files:
        name = f.split("20_")[1].split("_phase")[0]
        psth = load_spike_matrix(f)
        neurograms[name] = [
            transform(psth, binsize, window_size)
            for (_, window_size, binsize) in KERNELS
        ]

    names = sorted(list(neurograms.keys()))
    distances = np.zeros((3, len(names), len(names)))
    for (a,b) in combinations(names, 2):
        ai = names.index(a)
        bi = names.index(b)
        for i in range(len(KERNELS)):
            distances[i, ai, bi] = structural_similarity(neurograms[a][i], neurograms[b][i])
            distances[i, bi, ai] = distances[i, ai, bi]

    for i in range(len(KERNELS)):
        data = pd.DataFrame(distances[i], index=names, columns=names)
        print(KERNELS[i][0])
        print(data)
