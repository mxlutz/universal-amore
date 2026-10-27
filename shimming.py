import torch
import numpy as np
from einops import rearrange

def phase_shimming_torch(b1_maps, mask=None, n_inits=100, n_iter=500, lr=0.05,
                         device=None, seed=42, allow_amplitude=False,
                         objective="hom", lambda_=1.0,
                         b1_threshold=10.0):
    """
    RF shimming using PyTorch autodiff with vectorized optimization.

    All random initializations are optimized simultaneously in a single batched
    forward/backward pass per iteration, making full use of GPU parallelism.
    Minimizes the chosen objective across all initializations simultaneously,
    then returns the weights corresponding to the best individual result.

    Parameters
    ----------
    b1_maps : np.ndarray, complex, shape (x, y, n_Tx) or (x, y, z, n_Tx)
        Per-channel B1+ maps. Supports both 2D and 3D spatial dimensions.
        The transmit channel dimension is always last.
    mask : np.ndarray, bool, shape (x, y) or (x, y, z), or None, optional
        Binary ROI mask. Must match the spatial dimensions of b1_maps.
        If None, all voxels are included. Default: None.
    n_inits : int, optional
        Number of random phase initializations, optimized in parallel.
        Default: 100.
    n_iter : int, optional
        Number of Adam gradient steps. Default: 500.
    lr : float, optional
        Adam learning rate. Default: 0.05.
    device : str or torch.device or None, optional
        Device to run on ('cpu', 'cuda', 'mps'). None auto-selects.
        Default: None.
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    allow_amplitude : bool, optional
        If True, per-channel magnitudes in [0, 1] are optimized jointly with
        phases (amplitude-and-phase shimming). If False, magnitudes are fixed
        at 1 (phase-only shimming). Default: False.
    objective : {"hom", "eff", "homeff", "hommin", "effmin", "amore"}, optional
        Loss function to minimize. "hom" minimizes the coefficient of variation
        of |B1+| across ROI voxels (lower is better). "eff" maximizes
        the mean ratio |B1+| / sum(|B1_n|) across ROI voxels (minimizes the
        negative efficiency). "homeff" minimizes CoV - lambda_ * efficiency,
        trading off homogeneity against efficiency. "hommin" minimizes CoV
        plus a soft penalty for voxels below b1_threshold. "effmin" minimizes efficiency
        plus a soft penalty for voxels below b1_threshold. "amore" returns two
        shims and minimizes the squared penalty for voxels where the maximum
        of the two shimmed |B1+| maps is below b1_threshold.
        Default: "hom".
    lambda_ : float, optional
        Weighting factor for the secondary term: efficiency weight in "homeff",
        penalty weight in "hommin". Ignored for "hom" and "eff". Default: 1.0.
    b1_threshold : float, optional
        |B1+| threshold for the "hommin" objective. Voxels below this value
        are penalized. Default: 10.0.

    Returns
    -------
    weights : np.ndarray, complex, shape (n_Tx,) or (2, n_Tx)
        Optimal complex weights. When allow_amplitude is False, all weights
        have unit magnitude (|w_n| = 1). When True, magnitudes lie in [0, 1].
        For objective="amore", two shims are returned with shape (2, n_Tx).
    best_score : float
        Best achieved value of the chosen objective across ROI voxels.

    Notes
    -----
    - Complex B1+ maps are split into real/imaginary parts to ensure
      well-defined autograd gradients throughout the computation graph.
    - torch.sqrt(x**2 + y**2 + eps) is used instead of torch.abs() on
      complex tensors to avoid gradient instabilities near zero.
    - All n_inits initializations are batched along a leading dimension,
      so a single optimizer.step() updates all of them simultaneously.
    - On GPU, the batched approach is significantly faster than looping
      over initializations sequentially.
    - b1_maps with 3D spatial dimensions (x, y, z, n_Tx) are supported;
      spatial dimensions are flattened internally before optimization.
    - When allow_amplitude is True, magnitudes are parameterized via sigmoid
      to constrain them to (0, 1) while keeping gradients smooth.

    Examples
    --------
    >>> # Phase-only shimming
    >>> weights, cov = phase_shimming_torch(b1_maps, mask=mask)
    >>> # Amplitude-and-phase shimming
    >>> weights, cov = phase_shimming_torch(b1_maps, mask=mask, allow_amplitude=True)
    >>> print(f"Best CoV: {cov:.4f}")
    >>> b1_shimmed = np.abs(np.sum(b1_maps * weights, axis=-1))
    """
    if device is None:
        device = ('cuda' if torch.cuda.is_available()
                  else 'cpu')
    device = torch.device(device)

    # --- Validate dimensions ---
    # b1_maps must be (..., n_Tx) where ... is 2D or 3D spatial dims
    if b1_maps.ndim not in (3, 4):
        raise ValueError(
            f"b1_maps must be 2D (x, y, n_Tx) or 3D (x, y, z, n_Tx), "
            f"got shape {b1_maps.shape}."
        )
    spatial_shape = b1_maps.shape[:-1]     # (x, y) or (x, y, z)
    n_Tx = b1_maps.shape[-1]

    # --- Apply or build mask ---
    if mask is None:
        mask = np.ones(spatial_shape, dtype=bool)
    else:
        if mask.shape != spatial_shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match spatial dimensions "
                f"of b1_maps {spatial_shape}."
            )
            
    mask = mask.astype(bool)
    
    if objective not in ("hom", "eff", "homeff", "hommin", "effmin", "amore"):
        raise ValueError(
            "objective must be 'hom', 'eff', 'homeff', 'hommin', 'effmin', "
            f"or 'amore', got '{objective}'"
        )

    # Flatten spatial dims: b1_roi -> (n_voxels, n_Tx)
    b1_roi = b1_maps[mask]

    n_voxels, _ = b1_roi.shape
    b1_real = torch.tensor(b1_roi.real, dtype=torch.float32, device=device)
    b1_imag = torch.tensor(b1_roi.imag, dtype=torch.float32, device=device)

    # Per-channel magnitudes at each voxel: (n_voxels, n_Tx) -> sum over Tx: (n_voxels,)
    b1_abs_sum = torch.tensor(
        np.abs(b1_roi).sum(axis=-1), dtype=torch.float32, device=device
    )
    # b1_real, b1_imag: (n_voxels, n_Tx)

    n_shims = 2 if objective == "amore" else 1

    # --- Initialize all phases (and optionally amplitudes) at once ---
    torch.manual_seed(seed)
    phases = torch.nn.Parameter(
        torch.empty(n_inits, n_shims, n_Tx, device=device).uniform_(-torch.pi, torch.pi)
    )
    # phases: (n_inits, n_shims, n_Tx)

    params = [phases]
    if allow_amplitude:
        # Unconstrained parameter; sigmoid maps it to (0, 1).
        # Initialized at 0 so sigmoid(0) = 0.5 (mid-range).
        raw_magnitudes = torch.nn.Parameter(
            torch.zeros(n_inits, n_shims, n_Tx, device=device)
        )
        params.append(raw_magnitudes)

    # --- Define objective functions: return score per init (n_inits,) ---
    def obj_hom(abs_b1):
        return abs_b1.std(dim=1) / abs_b1.mean(dim=1)

    def obj_eff(abs_b1):
        return -(abs_b1 / b1_abs_sum.unsqueeze(0)).mean(dim=1)

    def obj_homeff(abs_b1):
        cov = abs_b1.std(dim=1) / abs_b1.mean(dim=1)
        eff = (abs_b1 / b1_abs_sum.unsqueeze(0)).mean(dim=1)
        return (1-lambda_) * cov - lambda_ * eff

    def obj_hommin(abs_b1):
        cov = abs_b1.std(dim=1) / abs_b1.mean(dim=1)
        penalty = torch.relu(b1_threshold - abs_b1).mean(dim=1)
        return (1-lambda_) * cov + lambda_ * penalty
    
    def obj_effmin(abs_b1):
        eff = (abs_b1 / b1_abs_sum.unsqueeze(0)).mean(dim=1)
        penalty = torch.relu(b1_threshold - abs_b1).mean(dim=1)
        return -(1-lambda_) *eff + lambda_ * penalty
    
    def obj_amore(abs_b1):
        abs_b1_amore = abs_b1.max(dim=1).values
        penalty = torch.relu(b1_threshold - abs_b1_amore)
        return (penalty**2).sum(dim=1)

    score_fns = {"hom": obj_hom, "eff": obj_eff, "homeff": obj_homeff, "hommin": obj_hommin, "effmin": obj_effmin, "amore": obj_amore}
    score_fn = score_fns[objective]

    optimizer = torch.optim.Adam(params, lr=lr)

    # --- Optimization loop ---
    for _ in range(n_iter):
        optimizer.zero_grad()

        magnitudes = torch.sigmoid(raw_magnitudes) if allow_amplitude else 1.0

        # w = magnitude * exp(i*phi), split into real/imag: (n_inits, n_shims, n_Tx)
        w_real = magnitudes * torch.cos(phases)
        w_imag = magnitudes * torch.sin(phases)

        # |B1+| per voxel per init per shim: (n_inits, n_shims, n_voxels)
        b1_sum_real = torch.einsum("ist,vt->isv", w_real, b1_real) - torch.einsum("ist,vt->isv", w_imag, b1_imag)
        b1_sum_imag = torch.einsum("ist,vt->isv", w_real, b1_imag) + torch.einsum("ist,vt->isv", w_imag, b1_real)
        abs_b1 = torch.sqrt(b1_sum_real**2 + b1_sum_imag**2 + 1e-12)
        if objective != "amore":
            abs_b1 = abs_b1[:, 0, :]

        score_fn(abs_b1).mean().backward()
        optimizer.step()

    # --- Extract best result ---
    with torch.no_grad():
        magnitudes = torch.sigmoid(raw_magnitudes) if allow_amplitude else 1.0
        w_real = magnitudes * torch.cos(phases)
        w_imag = magnitudes * torch.sin(phases)
        b1_sum_real = torch.einsum("ist,vt->isv", w_real, b1_real) - torch.einsum("ist,vt->isv", w_imag, b1_imag)
        b1_sum_imag = torch.einsum("ist,vt->isv", w_real, b1_imag) + torch.einsum("ist,vt->isv", w_imag, b1_real)
        abs_b1 = torch.sqrt(b1_sum_real**2 + b1_sum_imag**2 + 1e-12)
        if objective != "amore":
            abs_b1 = abs_b1[:, 0, :]
        score_per_init = score_fn(abs_b1)
        best_idx = score_per_init.argmin().item()

    best_score = score_per_init[best_idx].item()
    best_phases = phases[best_idx].detach().cpu().numpy()

    if allow_amplitude:
        best_mags = torch.sigmoid(raw_magnitudes[best_idx]).detach().cpu().numpy()
    else:
        best_mags = np.ones((n_shims, n_Tx))

    weights = best_mags * np.exp(1j * best_phases)
    if objective != "amore":
        weights = weights[0]
    return weights, best_score


def shim_on_pred_apply_on_target(pred, target, mask_liver, objective = "hom", lambda_ = 1.0, b1_threshold = 10.0, shim_weights = None, allow_amplitude = False):
    """
    Shim the prediction and apply the weights to the target.
    """
    if shim_weights is None:
        weights, _ = phase_shimming_torch(pred, mask_liver, objective = objective, lambda_ = lambda_, b1_threshold = b1_threshold, allow_amplitude = allow_amplitude)
    else:
        weights = shim_weights
    pred_shim = apply_shim_weights(pred, weights)
    target_shim = apply_shim_weights(target, weights)
    return pred_shim, target_shim, weights

def apply_shim_weights(b1_maps, weights, combine="max"):
    """
    Apply one shim or multiple shims to B1 maps.
    """
    weights = np.asarray(weights)
    if weights.ndim == 1:
        return np.abs(b1_maps @ weights)
    if weights.ndim != 2:
        raise ValueError(f"weights must have shape (n_Tx,) or (n_shims, n_Tx), got {weights.shape}")

    shimmed = np.abs(np.einsum("...c,sc->...s", b1_maps, weights))
    if combine == "max":
        return shimmed.max(axis=-1)
    if combine is None:
        return shimmed
    raise ValueError(f"combine must be 'max' or None, got {combine}")

def amore_roi_values(b1_maps, weights, roi):
    """
    Return max(Shim 1, Shim 2) B1 values inside the ROI.

    For (x, y, z) spatial maps, return shape (max_roi_voxels, z),
    packing each slice's ROI values into a column and padding with NaN.
    For (x, y) spatial maps, return a flat array of ROI values.
    """
    roi = roi.astype(bool)
    max_map = apply_shim_weights(b1_maps, weights)
    if max_map.shape != roi.shape:
        raise ValueError(f"ROI shape {roi.shape} does not match shimmed map shape {max_map.shape}")
    if max_map.ndim == 2:
        return max_map[roi].ravel()

    values_per_slice = [
        max_map[:, :, z_idx][roi[:, :, z_idx]]
        for z_idx in range(max_map.shape[2])
    ]
    max_roi_voxels = max((values.size for values in values_per_slice), default=0)
    values = np.full(
        (max_roi_voxels, max_map.shape[2]),
        np.nan,
        dtype=np.result_type(max_map.dtype, np.float32),
    )
    for z_idx, slice_values in enumerate(values_per_slice):
        values[:slice_values.size, z_idx] = slice_values
    return values

def amore_roi_values_per_slice(b1_maps, roi, **shim_kwargs):
    """
    Calculate AMORE shims per z slice.

    Return ROI values of shape (max_roi_voxels, z), shims of shape
    (z, 2, n_Tx), and scores of shape (z,). ROI columns are padded with
    NaN. Empty slices retain their z index with NaN values, shims, and scores.
    """
    if b1_maps.ndim != 4:
        raise ValueError(f"b1_maps must have shape (x, y, z, n_Tx), got {b1_maps.shape}")
    if roi.shape != b1_maps.shape[:-1]:
        raise ValueError(f"ROI shape {roi.shape} does not match B1 spatial shape {b1_maps.shape[:-1]}")

    roi = roi.astype(bool)
    n_slices = b1_maps.shape[2]
    max_roi_voxels = roi.sum(axis=(0, 1)).max(initial=0)
    values = np.full(
        (max_roi_voxels, n_slices),
        np.nan,
        dtype=np.result_type(b1_maps.real.dtype, np.float32),
    )
    shims = np.full((n_slices, 2, b1_maps.shape[-1]), np.nan, dtype=complex)
    scores = np.full(n_slices, np.nan)
    for z_idx in range(n_slices):
        roi_slice = roi[:, :, z_idx]
        if not np.any(roi_slice):
            continue

        shim, score = phase_shimming_torch(
            b1_maps[:, :, z_idx, :],
            roi_slice,
            objective="amore",
            **shim_kwargs,
        )
        slice_values = amore_roi_values(b1_maps[:, :, z_idx, :], shim, roi_slice)
        values[:slice_values.size, z_idx] = slice_values
        shims[z_idx] = shim
        scores[z_idx] = score

    return values, shims, scores

def calc_metrics_shim(b1shim, b1_abs_sum, roi):
    eps = 1e-12
    roi = roi.astype(bool)
    b1shim_roi = b1shim[roi]
    eff_roi = (b1shim / (b1_abs_sum + eps))[roi]
    return {
        "cv": np.std(b1shim_roi) / (np.mean(b1shim_roi) + eps),
        "min": np.min(b1shim_roi),
        "mean_efficiency": np.mean(eff_roi),
    }

def relative_phases(weights):
    """Return weights with phases made relative to the first channel (angle[0] == 0)."""
    weights = np.asarray(weights)
    if weights.ndim == 1:
        return weights * np.exp(-1j * np.angle(weights[0]))
    if weights.ndim == 2:
        return weights * np.exp(-1j * np.angle(weights[:, :1]))
    raise ValueError(f"weights must have shape (n_Tx,) or (n_shims, n_Tx), got {weights.shape}")


def eval_metrics_over_roi(pred, target, mask_liver, objective = "hom", lambda_ = 1.0, b1_threshold = 10.0, shim_weights = None, allow_amplitude = False):
    """
    Evaluate the metrics over the ROI.
    """
    target = rearrange(target, "c h w -> h w c")
    pred = rearrange(pred, "c h w -> h w c")
    weights_gt, _ = phase_shimming_torch(target, mask_liver, objective = objective, lambda_ = lambda_, b1_threshold = b1_threshold, allow_amplitude = allow_amplitude)
    shim_gt = apply_shim_weights(target, weights_gt)
    pred_shim, target_shim, weights_pred = shim_on_pred_apply_on_target(pred, target, mask_liver, objective, lambda_, b1_threshold, shim_weights, allow_amplitude)
    pred_abs_sum = np.sum(np.abs(pred), axis=-1)
    target_abs_sum = np.sum(np.abs(target), axis=-1)
    weights_pred_rel = relative_phases(weights_pred)
    weights_gt_rel = relative_phases(weights_gt)
    return {
        "shim_pred_on_pred": {**calc_metrics_shim(pred_shim, pred_abs_sum, mask_liver), "weights": weights_pred_rel},
        "shim_pred_on_target": {**calc_metrics_shim(target_shim, target_abs_sum, mask_liver), "weights": weights_pred_rel},
        "shim_target_on_target": {**calc_metrics_shim(shim_gt, target_abs_sum, mask_liver), "weights": weights_gt_rel},
    }
