# Dynamic Time Integration (DTI) Pipeline for SMILE SXI

This repository contains the Python implementation of the Dynamic Time Integration (DTI) method, a statistical quality-control framework for the Soft X-ray Imager (SXI) aboard the upcoming SMILE mission.

The DTI method systematically optimizes the analysis of SXI observations by dynamically determining whether the magnetopause is visible within a given field of view, and calculating the minimum integration time required to achieve an exploitable signal-to-noise ratio.

## How it works

The intrinsically weak soft X-ray emissivity of the magnetosheath can frequently be obscured by background and instrumental noise, so standard boundary-detection algorithms risk analyzing pure noise. The DTI pipeline guards against this by running three automated tests on a PCA (truncated-SVD) denoised image, searching for the shortest integration window that passes all three:

1. **Structure visibility (Test 1):** statistically evaluates whether a valid structural signal exists above the noise floor, and finds the minimum integration time (in stacked frames) at which it does, verified over a few subsequent frames to reject one-off noise spikes.
2. **Cusp rejection (Test 2):** identifies and excludes highly emissive magnetospheric cusps, evaluated over a longer, fixed integration window, to prevent boundary misidentification.
3. **Magnetopause FOV verification (Test 3):** fits an asymmetric Gaussian to the azimuthal (phi) profile, evaluated over a separate fixed integration window, and checks that the fitted peak falls within the field of view -- i.e. the magnetopause boundary itself is visible, not just cut off at the edge.

By dynamically adjusting the integration window (up to a configurable maximum, 10 frames by default), the DTI method balances signal clarity against preserving the boundary's temporal dynamics.

## Installation

The module has no dependencies beyond `numpy`, `scipy`, and `astropy`:

```bash
pip install numpy scipy astropy
```

Then either drop `SXI_Dynamic_Time_Integration_method.py` next to your analysis code, or add its directory to `sys.path`.

## Usage

```python
import SXI_Dynamic_Time_Integration_method as dti

# `directories` is a list of paths, one per time step, each containing the
# three FITS files (background, total, exposure) for that observation.
status, min_integration_time, final_image = dti.run_DTI_pipeline(
    index=1850,                # Central index into `directories`
    directories=my_dir_list,
    n_components=10,           # Number of PCA (truncated-SVD) components used to denoise
    max_ni=10,                 # Max integration time Test 1 will search up to (frames)
    ni_verif=3,                # Number of subsequent frames required to verify stability
    ni_cusp=30,                # Fixed integration window for cusp detection (Test 2)
    ni_mp=10,                  # Fixed integration window for the magnetopause-in-FOV test (Test 3)
)

if status:
    print(f"Magnetopause is visible! Required integration time: {min_integration_time} frames.")
    # Proceed to boundary detection algorithms (e.g. tangent fitting) on `final_image`
else:
    print("Magnetopause is obscured, absent, or a cusp. Interpret downstream analysis with caution.")
```

### Individual tests

`run_DTI_pipeline` is the recommended entry point, but each test is also exposed independently for custom pipelines:

| Function | Purpose |
|---|---|
| `stack_images(center_index, num_images, directories)` | Stack and exposure-normalize a window of frames |
| `remove_noise_with_tsvd(image, n_components)` | Truncated-SVD denoising |
| `find_min_integration_time(index, directories, ...)` | Test 1: minimum integration time for a visible structure |
| `test2_is_cusp(image)` | Test 2: is this a magnetospheric cusp? |
| `test3_is_mp_in_fov(image)` | Test 3: is the magnetopause contained in the field of view? |

## Verification

`examples.ipynb` runs the module against real SMILE SXI simulated data (the SMILE_challenge dataset) and its noise-free ground truth, and documents the results: a sweep of detections across PCA component counts (best accuracy ~91% around `n_components = 9-11`), a confusion matrix and precision/recall/accuracy curve against ground truth, standalone accuracy for Test 2 (~95%) and Test 3, and a worked experiment on making Test 3's curve fit more robust (concluding it should *not* be changed, since forcing convergence on ambiguous images trades honest failures for confidently wrong answers).
