# Universal AMORE

Optimize universal AMORE RF shims from channelwise B1 maps and liver masks,
and compare them with baseline and slice-tailored shims.

## Installation

Python 3.10 or newer. Install dependencies from the repository root:

```bash
python -m pip install -e .
```

## Data

Data are available on [Zenodo](https://doi.org/10.5281/zenodo.23692144).

`calc_shim.py` automatically downloads and extracts `data.zip` into the
repository root when the data are missing. Existing train/val data are reused.
Files in `data/train/` and `data/val/` are named `subj<number>_<position>.h5`
and contain:

- `b1_map_channelwise_deg`: complex B1 maps in degrees, shape `(x, y, channels)`.
- `mask_liver`: liver ROI mask, shape `(x, y)`.

## Usage

Run from the repository root:

```bash
python calc_shim.py
```

- Saves comparison figures in `figures/` and universal shim weights in `shim.npy`.
- Uses CUDA when available, otherwise CPU.
- Assumes eight transmit channels; the example map plot uses validation index 3.
- Optimization and plotting parameters are set in `calc_shim.py`.

To prepare original files from `data_original/train/` and `data_original/val/`:

```bash
python anonymize_data.py
```

Creates `data/` with anonymous filenames and only the two datasets above.
The output directory must not already exist.

## References

- AMORE method: [10.1002/mrm.29318](https://doi.org/10.1002/mrm.29318).
- Universal AMORE idea: [10.1002/mrm.30032](https://doi.org/10.1002/mrm.30032).

## License

[MIT](LICENSE).
