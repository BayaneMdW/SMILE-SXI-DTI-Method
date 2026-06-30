# Dynamic Time Integration (DTI) Pipeline for SMILE SXI

This repository contains the Python implementation of the Dynamic Time Integration (DTI) method, a statistical quality-control framework designed for the Soft X-ray Imager (SXI) aboard the upcoming SMILE mission. 

The DTI method systematically optimizes the analysis of SXI observations by dynamically determining whether the magnetopause is visible within a given field of view, and calculating the minimum integration time required to achieve an exploitable signal-to-noise ratio. 

## Features

Because the intrinsically weak soft X-ray emissivity of the magnetosheath can frequently be obscured by background and instrumental noise, standard boundary detection algorithms risk analyzing pure noise. The DTI pipeline prevents this by running three automated tests:
1. **Visibility Assessment:** Statistically evaluates if a valid structural signal exists above the noise floor using PCA (Truncated SVD) denoising.
2. **Cusp Rejection:** Identifies and excludes highly emissive magnetospheric cusps to prevent boundary misidentification.
3. **Magnetopause FOV Verification:** Fits an asymmetric Gaussian to ensure the magnetopause boundary is properly contained within the image field of view.

By dynamically adjusting the integration window (up to a 10-minute maximum), the DTI method ensures an optimal balance between signal clarity and the preservation of the boundary's temporal dynamics.




```
from dti_pipeline import run_dti_pipeline
import numpy as np



# Run the pipeline for a specific time index
status_valid, min_integration_time, final_image = run_dti_pipeline(
    index=1850,                # Central index of the directory list
    directories=my_dir_list,   # List of paths to your FITS directories
    n_components=10,           # Recommended number of PCA components
    max_ni=10,                 # Max integration time (minutes/frames)
    ni_verif=3,                # Frames required to verify stability
    ni_cusp=30                 # Window for slow-moving cusp detection
)

if status_valid:
    print(f"Magnetopause is visible! Required integration time: {min_integration_time} min.")
    # Proceed to boundary detection algorithms (e.g., Tangent Fitting)
else:
    print("Magnetopause is obscured or absent. Interpret downstream analysis with caution.")

```
