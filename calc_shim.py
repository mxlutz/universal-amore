"""Optimize universal AMORE shims and save training/validation comparisons.

Run from the repository root. Missing data are downloaded from Zenodo.
"""

# %%
from pathlib import Path
import hashlib
import json
import shutil
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dataloader import MrfGreDataset
from shimming import (
    amore_roi_values,
    amore_roi_values_per_slice,
    apply_shim_weights,
    phase_shimming_torch,
)


def ensure_data(data_dir=Path("data")):
    """Download the Zenodo dataset if train/val data are not already present."""
    data_dir = Path(data_dir).resolve()
    if all(any((data_dir / split).glob("*.h5")) for split in ("train", "val")):
        return
    if data_dir.exists():
        raise FileExistsError(
            f"{data_dir} exists but is missing train/val HDF5 files. "
            "Move it aside or complete it before downloading."
        )

    record_url = "https://zenodo.org/api/records/23692144"
    print("Fetching dataset information from Zenodo (10.5281/zenodo.23692144)...")
    with urlopen(record_url, timeout=60) as response:
        record = json.load(response)
    archive_info = next(
        (item for item in record["files"] if item["key"] == "data.zip"), None
    )
    if archive_info is None:
        raise RuntimeError("The Zenodo record does not contain data.zip.")

    # Stage both download and extraction so failures leave no partial data/.
    with TemporaryDirectory(prefix="amore-data-", dir=data_dir.parent) as temporary:
        staging = Path(temporary)
        archive_path = staging / "data.zip"
        print(f"Downloading data.zip ({archive_info['size'] / 1e6:.1f} MB)...")
        with urlopen(archive_info["links"]["self"], timeout=60) as response:
            with archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)

        algorithm, expected = archive_info["checksum"].split(":", 1)
        checksum = hashlib.new(algorithm)
        with archive_path.open("rb") as downloaded:
            for chunk in iter(lambda: downloaded.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise RuntimeError("Dataset checksum mismatch. Please retry the download.")

        unpacked = staging / "unpacked"
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (unpacked / member.filename).resolve()
                if not target.is_relative_to(unpacked.resolve()):
                    raise ValueError(f"Unsafe ZIP entry: {member.filename}")
            archive.extractall(unpacked)
        extracted_data = unpacked / "data"
        if not all(
            any((extracted_data / split).glob("*.h5")) for split in ("train", "val")
        ):
            raise RuntimeError("Expected data/train and data/val HDF5 files in data.zip.")
        extracted_data.rename(data_dir)
    print(f"Dataset extracted to {data_dir}")


def show_or_close(figure):
    """Display saved figures interactively; close them in batch runs."""
    try:
        interactive = get_ipython() is not None
    except NameError:
        interactive = False
    if interactive or sys.flags.interactive or plt.isinteractive():
        plt.show()
    else:
        plt.close(figure)


ensure_data()
Path("figures").mkdir(parents=True, exist_ok=True)

dataset = MrfGreDataset("data/train")
dataset_val = MrfGreDataset("data/val")
b1_channelwise_deg, mask_liver = dataset[:]
b1_channelwise_deg_val, mask_liver_val = dataset_val[:]
print(b1_channelwise_deg.shape)
print(mask_liver.shape)
print(b1_channelwise_deg_val.shape)
print(mask_liver_val.shape)

# %% Baseline shims
shim_0_cp = np.ones((2, 8), dtype=np.complex64)
shim_0_cp[1] = np.exp(1j * np.arange(0, 2 * np.pi, 2 * np.pi / 8))

# %%
obj = 'amore'
b1_threshold = 15.0
shim, score = phase_shimming_torch(b1_channelwise_deg, mask_liver, objective=obj, b1_threshold=b1_threshold)
amore_values_roi = amore_roi_values(b1_channelwise_deg, shim, mask_liver)
amore_values_roi_0_cp = amore_roi_values(b1_channelwise_deg, shim_0_cp, mask_liver)
amore_values_roi_per_slice, shims_per_slice, scores_per_slice = amore_roi_values_per_slice(
    b1_channelwise_deg,
    mask_liver,
    b1_threshold=b1_threshold,
)
print(amore_values_roi.shape)
print(np.nanmin(amore_values_roi))
print(amore_values_roi_per_slice.shape)
if amore_values_roi_per_slice.size:
    print(np.nanmin(amore_values_roi_per_slice))

# %% validation
amore_values_roi_0_cp_val = amore_roi_values(b1_channelwise_deg_val, shim_0_cp, mask_liver_val)
amore_values_roi_val = amore_roi_values(b1_channelwise_deg_val, shim, mask_liver_val)
amore_values_roi_val_per_slice, shims_val_per_slice, scores_val_per_slice = amore_roi_values_per_slice(
    b1_channelwise_deg_val,
    mask_liver_val,
    b1_threshold=b1_threshold,
)
print(amore_values_roi_val.shape)
print(np.nanmin(amore_values_roi_val))
print(amore_values_roi_val_per_slice.shape)
if amore_values_roi_val_per_slice.size:
    print(np.nanmin(amore_values_roi_val_per_slice))

# %%
def percentage_below_threshold(values, threshold):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    return 100 * np.count_nonzero(finite_values < threshold) / finite_values.size


def mean_lowest_percent(values, percent):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    count = max(1, int(np.ceil(finite_values.size * percent / 100)))
    return np.mean(np.sort(finite_values)[:count])


vmax = 40
cmap = "jet"
slice_idx = 3

universal_b1 = apply_shim_weights(b1_channelwise_deg_val[:,:,slice_idx], shim, combine=None)
baseline_b1 = apply_shim_weights(b1_channelwise_deg_val[:,:,slice_idx], shim_0_cp, combine=None)
tailored_b1 = apply_shim_weights(b1_channelwise_deg_val[:,:,slice_idx], shims_val_per_slice[slice_idx], combine=None)

slice_mask = mask_liver_val[:, :, slice_idx].astype(bool)
slice_values = {
    "Baseline": np.max(np.abs(baseline_b1), axis=2)[slice_mask],
    "Universal AMORE": np.max(np.abs(universal_b1), axis=2)[slice_mask],
    "Tailored AMORE": np.max(np.abs(tailored_b1), axis=2)[slice_mask],
}
threshold = 10
lowest_percent = 2
print(f"Percentage below {threshold:g} deg in validation slice {slice_idx}:")
for label, values in slice_values.items():
    print(f"{label}: {percentage_below_threshold(values, threshold):.2f}%")
print(f"Mean of the lowest {lowest_percent:g}% in validation slice {slice_idx}:")
for label, values in slice_values.items():
    print(f"{label}: {mean_lowest_percent(values, lowest_percent):.2f} deg")

plt.style.use("dark_background")
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)
background_color = plt.get_cmap(cmap)(0)
fig, axs = plt.subplots(
    3,
    3,
    figsize=(8, 5),
    constrained_layout=True,
    facecolor=background_color,
)
images = [
    [np.abs(baseline_b1[:, :, 0]), np.abs(baseline_b1[:, :, 1]), np.max(np.abs(baseline_b1), axis=2)],
    [np.abs(universal_b1[:, :, 0]), np.abs(universal_b1[:, :, 1]), np.max(np.abs(universal_b1), axis=2)],
    [np.abs(tailored_b1[:, :, 0]), np.abs(tailored_b1[:, :, 1]), np.max(np.abs(tailored_b1), axis=2)],
]
column_titles = ["shim 1", "shim 2", "max (shim 1, shim 2)"]
row_titles = ["Baseline", "Universal\nAMORE", "Tailored\nAMORE"]
x_limits = (15, 113)
y_limits = (35, 103)
mask = mask_liver_val[:, :, slice_idx].T.astype(bool)
outside_roi_alpha = 0.5

for row, row_images in enumerate(images):
    for column, image in enumerate(row_images):
        display_image = image.T
        axs[row, column].imshow(
            display_image,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            origin="lower",
            alpha=outside_roi_alpha,
        )
        roi_image = np.ma.masked_where(~mask, display_image)
        plot = axs[row, column].imshow(roi_image, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        axs[row, column].set_facecolor(background_color)
        axs[row, column].set_xlim(*x_limits)
        axs[row, column].set_ylim(*y_limits)
        axs[row, column].set_xticks([])
        axs[row, column].set_yticks([])
        masked_min = np.nanmin(display_image[mask])
        axs[row, column].text(
            0.97,
            0.05,
            f"min: {masked_min:.1f} deg",
            transform=axs[row, column].transAxes,
            color="white",
            fontsize=12,
            ha="right",
            va="bottom",
            bbox={"facecolor": background_color, "edgecolor": "none", "alpha": 0.7, "pad": 2},
        )
        if row == 0:
            axs[row, column].set_title(column_titles[column])
    axs[row, 0].set_ylabel(row_titles[row], rotation=90, labelpad=10, fontsize=14)

fig.colorbar(plot, ax=axs, location="right", shrink=0.85, label="|B1| (deg)")
for ax in axs.flat:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.contour(mask_liver_val[:, :, slice_idx].T, levels=[0.5], colors="white", linewidths=2)

fig.savefig(f"figures/shim_comparison_val_idx_{slice_idx}.png", dpi=300, bbox_inches="tight")
show_or_close(fig)
# %% ROI histograms

vmax = 30
roi_mask = mask_liver_val.astype(bool)
universal_b1_all = apply_shim_weights(b1_channelwise_deg_val, shim, combine=None)
baseline_b1_all = apply_shim_weights(b1_channelwise_deg_val, shim_0_cp, combine=None)
tailored_b1_all = np.stack(
    [
        apply_shim_weights(b1_channelwise_deg_val[:, :, z_idx], shims_val_per_slice[z_idx], combine=None)
        for z_idx in range(b1_channelwise_deg_val.shape[2])
    ],
    axis=2,
)
histogram_values = {
    "Baseline": [
        np.abs(baseline_b1_all[..., 0])[roi_mask],
        np.abs(baseline_b1_all[..., 1])[roi_mask],
        amore_values_roi_0_cp_val[np.isfinite(amore_values_roi_0_cp_val)],
    ],
    "Universal AMORE": [
        np.abs(universal_b1_all[..., 0])[roi_mask],
        np.abs(universal_b1_all[..., 1])[roi_mask],
        amore_values_roi_val[np.isfinite(amore_values_roi_val)],
    ],
    "Tailored AMORE": [
        np.abs(tailored_b1_all[..., 0])[roi_mask],
        np.abs(tailored_b1_all[..., 1])[roi_mask],
        amore_values_roi_val_per_slice[np.isfinite(amore_values_roi_val_per_slice)],
    ],
}

histogram_titles = ["shim 1", "shim 2", "max (shim 1, shim 2)"]
histogram_colors = {
    "Baseline": "slategray",
    "Universal AMORE": "deepskyblue",
    "Tailored AMORE": "darkorange",
}
lower_threshold = 10.0
histogram_bins = np.linspace(0, vmax, 31)
sns.set_theme(style="white")
hist_fig, hist_axes = plt.subplots(
    3,
    1,
    figsize=(6, 6),
    sharex=True,
    constrained_layout=True,
    facecolor="white",
)
hist_fig.suptitle("Validation data", color="black")

for row, title in enumerate(histogram_titles):
    hist_axes[row].set_facecolor("white")
    for label, color in histogram_colors.items():
        sns.histplot(
            histogram_values[label][row],
            bins=histogram_bins,
            stat="density",
            element="step",
            fill=True,
            alpha=0.35,
            color=color,
            label=label,
            ax=hist_axes[row],
        )
    hist_axes[row].set_title(title)
    hist_axes[row].set_ylabel("Density")
    hist_axes[row].set_xlim(0, vmax)
    hist_axes[row].axvline(
        lower_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="lower threshold",
    )
    hist_axes[row].legend(loc="upper right")
    hist_axes[row].tick_params(colors="black")
    hist_axes[row].xaxis.label.set_color("black")
    hist_axes[row].yaxis.label.set_color("black")
    hist_axes[row].title.set_color("black")

hist_axes[-1].set_xlabel("|B1| in ROI (deg)")
hist_fig.savefig(f"figures/hist_shim_comparison_val_idx_{slice_idx}.png", dpi=300, bbox_inches="tight")
show_or_close(hist_fig)

# %% Training ROI histograms
universal_b1_train_all = apply_shim_weights(b1_channelwise_deg, shim, combine=None)
baseline_b1_train_all = apply_shim_weights(b1_channelwise_deg, shim_0_cp, combine=None)
tailored_b1_train_all = np.stack(
    [
        apply_shim_weights(b1_channelwise_deg[:, :, z_idx], shims_per_slice[z_idx], combine=None)
        for z_idx in range(b1_channelwise_deg.shape[2])
    ],
    axis=2,
)
train_roi_mask = mask_liver.astype(bool)
training_histogram_values = {
    "Baseline": [
        np.abs(baseline_b1_train_all[..., 0])[train_roi_mask],
        np.abs(baseline_b1_train_all[..., 1])[train_roi_mask],
        amore_values_roi_0_cp[np.isfinite(amore_values_roi_0_cp)],
    ],
    "Universal AMORE": [
        np.abs(universal_b1_train_all[..., 0])[train_roi_mask],
        np.abs(universal_b1_train_all[..., 1])[train_roi_mask],
        amore_values_roi[np.isfinite(amore_values_roi)],
    ],
    "Tailored AMORE": [
        np.abs(tailored_b1_train_all[..., 0])[train_roi_mask],
        np.abs(tailored_b1_train_all[..., 1])[train_roi_mask],
        amore_values_roi_per_slice[np.isfinite(amore_values_roi_per_slice)],
    ],
}

train_hist_fig, train_hist_axes = plt.subplots(
    3,
    1,
    figsize=(6, 6),
    sharex=True,
    constrained_layout=True,
    facecolor="white",
)
train_hist_fig.suptitle("Training data", color="black")

for row, title in enumerate(histogram_titles):
    train_hist_axes[row].set_facecolor("white")
    for label, color in histogram_colors.items():
        sns.histplot(
            training_histogram_values[label][row],
            bins=histogram_bins,
            stat="density",
            element="step",
            fill=True,
            alpha=0.35,
            color=color,
            label=label,
            ax=train_hist_axes[row],
        )
    train_hist_axes[row].set_title(title)
    train_hist_axes[row].set_ylabel("Density")
    train_hist_axes[row].set_xlim(0, vmax)
    train_hist_axes[row].axvline(
        lower_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="lower threshold",
    )
    train_hist_axes[row].legend(loc="upper right")
    train_hist_axes[row].tick_params(colors="black")
    train_hist_axes[row].xaxis.label.set_color("black")
    train_hist_axes[row].yaxis.label.set_color("black")
    train_hist_axes[row].title.set_color("black")

train_hist_axes[-1].set_xlabel("|B1| in ROI (deg)")
train_hist_fig.savefig("figures/hist_shim_comparison_train.png", dpi=300, bbox_inches="tight")
show_or_close(train_hist_fig)

# %% Max-shim ROI histograms
max_histogram_values = {
    "Training data": {
        "Baseline": amore_values_roi_0_cp[np.isfinite(amore_values_roi_0_cp)],
        "Universal AMORE": amore_values_roi[np.isfinite(amore_values_roi)],
        "Tailored AMORE": amore_values_roi_per_slice[np.isfinite(amore_values_roi_per_slice)],
    },
    "Validation data": {
        "Baseline": amore_values_roi_0_cp_val[np.isfinite(amore_values_roi_0_cp_val)],
        "Universal AMORE": amore_values_roi_val[np.isfinite(amore_values_roi_val)],
        "Tailored AMORE": amore_values_roi_val_per_slice[np.isfinite(amore_values_roi_val_per_slice)],
    },
}
max_hist_fig, max_hist_axes = plt.subplots(
    2,
    1,
    figsize=(6, 4),
    sharex=True,
    constrained_layout=True,
    facecolor="white",
)

for row, (dataset_label, dataset_values) in enumerate(max_histogram_values.items()):
    max_hist_axes[row].set_facecolor("white")
    for label, color in histogram_colors.items():
        sns.histplot(
            dataset_values[label],
            bins=histogram_bins,
            stat="density",
            element="step",
            fill=True,
            alpha=0.35,
            color=color,
            label=label,
            ax=max_hist_axes[row],
        )
    max_hist_axes[row].set_title(dataset_label)
    max_hist_axes[row].set_ylabel("Density")
    max_hist_axes[row].set_xlim(0, vmax)
    max_hist_axes[row].axvline(
        lower_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label="lower threshold",
    )
    max_hist_axes[row].legend(loc="upper right")
    max_hist_axes[row].tick_params(colors="black")
    max_hist_axes[row].xaxis.label.set_color("black")
    max_hist_axes[row].yaxis.label.set_color("black")
    max_hist_axes[row].title.set_color("black")

max_hist_axes[-1].set_xlabel("max(shim 1, shim 2) in ROI (deg)")
max_hist_fig.savefig("figures/hist_shim_comparison_max_roi.png", dpi=300, bbox_inches="tight")
show_or_close(max_hist_fig)

# %% Percentage below threshold
def mean_lowest_percent_per_dataset(values, percent):
    values = np.asarray(values)
    return np.array(
        [mean_lowest_percent(values[:, dataset_idx], percent) for dataset_idx in range(values.shape[1])]
    )


threshold = 10
lowest_percent = 2
threshold_values = {
    "Baseline": (amore_values_roi_0_cp, amore_values_roi_0_cp_val),
    "Universal AMORE": (amore_values_roi, amore_values_roi_val),
    "Tailored AMORE": (amore_values_roi_per_slice, amore_values_roi_val_per_slice),
}

print(f"Percentage below {threshold:g} deg for max(shim 1, shim 2):")
for label, (training_values, validation_values) in threshold_values.items():
    training_percentage = percentage_below_threshold(training_values, threshold)
    validation_percentage = percentage_below_threshold(validation_values, threshold)
    print(
        f"{label}: training = {training_percentage:.2f}%, "
        f"validation = {validation_percentage:.2f}%"
    )

print(f"Mean of the lowest {lowest_percent:g}% for max(shim 1, shim 2):")
for label, (training_values, validation_values) in threshold_values.items():
    training_mean = mean_lowest_percent(training_values, lowest_percent)
    validation_mean = mean_lowest_percent(validation_values, lowest_percent)
    print(
        f"{label}: training = {training_mean:.2f} deg, "
        f"validation = {validation_mean:.2f} deg"
    )

# %% Lowest-percent radar plot per dataset index
radar_data = {
    "Training data": {
        label: mean_lowest_percent_per_dataset(training_values, lowest_percent)
        for label, (training_values, _) in threshold_values.items()
    },
    "Validation data": {
        label: mean_lowest_percent_per_dataset(validation_values, lowest_percent)
        for label, (_, validation_values) in threshold_values.items()
    },
}
radar_max = max(
    np.nanmax(values)
    for split_values in radar_data.values()
    for values in split_values.values()
)
radar_colors = {
    "Baseline": "slategray",
    "Universal AMORE": "deepskyblue",
    "Tailored AMORE": "darkorange",
}

radar_fig, radar_axes = plt.subplots(
    1,
    2,
    figsize=(13, 7),
    subplot_kw={"polar": True},
    constrained_layout=False,
    facecolor="white",
)
radar_fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.4)
for radar_ax, (split_label, split_values) in zip(radar_axes, radar_data.items()):
    dataset_count = min(values.size for values in split_values.values())
    radar_angles = np.linspace(0, 2 * np.pi, dataset_count, endpoint=False)
    radar_angles = np.concatenate([radar_angles, radar_angles[:1]])
    radar_labels = [str(dataset_idx) for dataset_idx in range(dataset_count)]
    radar_ax.set_facecolor("white")
    for method, values in split_values.items():
        values = values[:dataset_count]
        closed_values = np.concatenate([values, values[:1]])
        radar_ax.plot(
            radar_angles,
            closed_values,
            color=radar_colors[method],
            linewidth=2,
            label=method,
        )
        radar_ax.fill(radar_angles, closed_values, color=radar_colors[method], alpha=0.15)
    radar_ax.set_xticks(radar_angles[:-1])
    radar_ax.set_xticklabels(radar_labels, color="black")
    radar_ax.set_ylim(0, radar_max * 1.15)
    radar_ax.set_title(
        f"{split_label}\nMean lowest {lowest_percent:g}% of max(shim 1, shim 2)",
        color="black",
        pad=22,
    )
    radar_ax.tick_params(colors="black")
handles, labels = radar_axes[0].get_legend_handles_labels()
radar_fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04))
radar_fig.supxlabel("Dataset index", color="black", y=0.12)
radar_fig.savefig(f"figures/radar_lowest_percent_{lowest_percent:g}.png", dpi=300, bbox_inches="tight")
show_or_close(radar_fig)

# %%
print(np.abs(shim))
print(np.angle(shim))

np.save("shim.npy", shim)
# %%
