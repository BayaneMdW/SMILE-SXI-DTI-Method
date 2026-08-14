import os
import glob 
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from typing import Tuple, List, Optional


def load_smile_images(directory_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the background, total, and velocity (vcy) FITS images from a given directory.
    
    Args:
        directory_path (str): The path to the directory containing the FITS files.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the background (ibkg), 
                                                   total (itot), and velocity (ivcy) image arrays.
    """
    file_paths = glob.glob(os.path.join(directory_path, '*'))
    file_paths.sort()
    
    if len(file_paths) != 3:
        raise ValueError(f"Expected exactly 3 files in {directory_path}, found {len(file_paths)}.")
        
    bkg_file, tot_file, vcy_file = file_paths
    
    ibkg = fits.getdata(bkg_file)
    itot = fits.getdata(tot_file)
    ivcy = fits.getdata(vcy_file)
    
    return ibkg, itot, ivcy


def stack_images(center_index: int, num_images: int, directories: List[str]) -> np.ndarray:
    """
    Stacks and averages images over a given time window.
    
    Args:
        center_index (int): The target index in the directory list.
        num_images (int): The number of images to stack (integration window).
        directories (List[str]): List of all available directory paths.
        
    Returns:
        np.ndarray: The stacked and normalized image.
    """
    # Get shape from the first image in the target directory
    base_shape = load_smile_images(directories[center_index])[0].shape
    ibkg = np.zeros(base_shape)
    itot = np.zeros(base_shape)
    ivcy = np.zeros(base_shape)
    
    # Calculate safe start and end indices to prevent out-of-bounds errors
    start_idx = max(0, int(center_index - num_images / 2))
    end_idx = min(len(directories), int(center_index + num_images / 2))
    
    count = 0
    for j in range(start_idx, end_idx):
        count += 1
        bkg, tot, vcy = load_smile_images(directories[j])
        ibkg += bkg
        itot += tot
        ivcy += vcy
        
    if count == 0:
        raise ValueError("No images were found in the specified window.")
        
    # Calculate the final stacked image
    image = (itot - ibkg) / (ivcy / count)
    return image


def remove_noise_with_tsvd(image: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Reduces noise in an image using Truncated Singular Value Decomposition (SVD).
    
    Args:
        image (np.ndarray): The input image array.
        n_components (int): The number of principal components to retain.
        
    Returns:
        np.ndarray: The noise-reduced image.
    """
    img_float = image.astype(np.float64)
    mean = img_float.mean(axis=0)
    centered = img_float - mean
    
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    reconstructed = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    
    return reconstructed + mean


def average_along_phi_axis(image: np.ndarray) -> np.ndarray:
    """Averages the image along the phi (azimuthal) axis (axis 0)."""
    return np.mean(image, axis=0)


def average_along_theta_axis(image: np.ndarray) -> np.ndarray:
    """Averages the image along the theta (elevation) axis (axis 1)."""
    return np.mean(image, axis=1)


def test1_is_structure_in_fov(image: np.ndarray) -> bool:
    """
    Test 1: Determines if a significant structure is visible in the field of view.
    """
    imean = average_along_phi_axis(image)
    ipic = imean.max() - imean.min()
    imax = imean.max()
    istd = image.std()
    
    return (ipic >= istd) and (imax >= istd)


def test2_is_cusp(image: np.ndarray) -> bool:
    """
    Test 2: Determines if the visible structure is a magnetospheric cusp.
    """
    itheta = average_along_theta_axis(image)
    theta_pic = itheta.max() - itheta.min()
    
    iphi = average_along_phi_axis(image)
    phi_pic = iphi.max() - iphi.min()
    
    return theta_pic >= phi_pic


def asymmetric_gaussian(x: np.ndarray, amplitude: float, mean: float, 
                        sigma_left: float, sigma_right: float, min_val: float) -> np.ndarray:
    """
    Calculates an asymmetric Gaussian function.
    """
    x = np.asarray(x, dtype=float)
    y = np.where(
        x < mean,
        amplitude * np.exp(-0.5 * ((x - mean) / sigma_left) ** 2),
        amplitude * np.exp(-0.5 * ((x - mean) / sigma_right) ** 2),
    )
    return y + min_val


# Physical field-of-view grid for the phi (azimuthal) axis, in degrees.
# Matches the SMILE SXI instrument geometry: 64 pixels spanning +/-7.75 deg.
PHI_FOV_DEG = np.linspace(-7.75, 7.75, 64)


def test3_is_mp_in_fov(image: np.ndarray) -> bool:
    """
    Test 3: Determines if the magnetopause (MP) is contained within the field of view.
    Fits an asymmetric Gaussian to the azimuthal profile and checks that its peak
    falls within +/-8 deg of the field-of-view center.
    """
    av_phi = average_along_phi_axis(image)
    phi = PHI_FOV_DEG if len(av_phi) == len(PHI_FOV_DEG) else np.linspace(-7.75, 7.75, len(av_phi))

    initial_guess = [av_phi.max(), phi[av_phi.argmax()], 3.0, 2.0, av_phi.min()]

    try:
        popt, _ = curve_fit(asymmetric_gaussian, phi, av_phi, p0=initial_guess)
    except RuntimeError:
        # Curve fitting failed to converge
        return False
    return abs(popt[1]) <= 8


def find_min_integration_time(index: int, directories: List[str], 
                              n_components: int = 10, max_ni: int = 10, 
                              min_ni: int = 1, ni_verif: int = 3) -> Tuple[bool, int]:
    """
    Calculates the minimum integration time required to achieve a reliable signal-to-noise ratio.
    
    Args:
        index (int): The current time index in the directory list.
        directories (List[str]): List of all available directory paths.
        n_components (int): Number of PCA components for denoising.
        max_ni (int): Maximum integration time limit (in minutes/frames).
        min_ni (int): Minimum integration time.
        ni_verif (int): Number of subsequent frames required to verify the structure.
        
    Returns:
        Tuple[bool, int]: (Visibility boolean, Required integration time in minutes)
    """
    for ni in range(min_ni, max_ni + 1):
        image = stack_images(index, ni, directories)
        image = remove_noise_with_tsvd(image, n_components=n_components)
        
        if test1_is_structure_in_fov(image):
            valid_frames = 0
            
            # Verify the structure remains visible for the next few integration steps.
            # This intentionally may probe windows beyond max_ni: max_ni only bounds
            # where the search for the minimum ni starts from, not how far ahead
            # verification is allowed to look.
            for v_ni in range(ni + 1, ni + ni_verif + 1):
                v_image = stack_images(index, v_ni, directories)
                v_image = remove_noise_with_tsvd(v_image, n_components=n_components)
                
                if test1_is_structure_in_fov(v_image):
                    valid_frames += 1
                    
            if valid_frames >= ni_verif:
                return (True, ni)
                
    return (False, max_ni)

def run_DTI_pipeline(index: int, directories: List[str], n_components: int,  max_ni: int = 10,
                     ni_verif: int = 3, ni_cusp: int = 30, ni_mp: int = 10) -> Tuple[bool, int, np.ndarray]:
    """
    Executes the full Dynamic Time Integration (DTI) method pipeline.

    Evaluates an image sequence to determine if the magnetopause is visible,
    ensures the structure is not a magnetospheric cusp, and calculates the
    minimum required integration time.

    Args:
        index (int): The current central time index in the directory list.
        directories (List[str]): List of all available image directory paths.
        n_components (int): Number of PCA components for TSVD denoising.
        max_ni (int): Maximum integration time limit (in frames/minutes).
        ni_verif (int): Number of subsequent frames required to verify the structure.
        ni_cusp (int): Large integration window used specifically to identify slow-moving cusps.
        ni_mp (int): Fixed integration window used for the magnetopause-in-FOV test.
            Kept fixed rather than using the optimized `ni`, so the test always runs
            at a consistent SNR regardless of how short the detected integration
            time is (matches the original analysis pipeline).

    Returns:
        Tuple[bool, int, np.ndarray]: 
            - bool: Final DTI status (True if Magnetopause is valid and visible, False otherwise).
            - int: The determined optimal integration time (ni).
            - np.ndarray: The final denoised and integrated image.
    """
    
    # ---------------------------------------------------------
    # TEST 1: Is a structure visible & what is the min integration time?
    # ---------------------------------------------------------
    status_structure, ni = find_min_integration_time(
        index, directories, n_components=n_components, 
        max_ni=max_ni, min_ni=1, ni_verif=ni_verif
    )
    
    # OPTIMIZATION: Short-circuit. If no structure is found, do not waste CPU 
    # stacking 30 images for the cusp test. Return immediately.
    if not status_structure:
        # Generate the standard max_ni integrated image to return
        image = stack_images(index, ni, directories)
        image = remove_noise_with_tsvd(image, n_components=n_components)
        return False, ni, image

    # ---------------------------------------------------------
    # TEST 2: Is the structure a magnetospheric cusp?
    # ---------------------------------------------------------
    # Use the longer ni_cusp integration window specifically for cusp detection
    cusp_image = stack_images(index, ni_cusp, directories)
    cusp_image = remove_noise_with_tsvd(cusp_image, n_components=n_components)
    status_cusp = test2_is_cusp(cusp_image)
    
    # OPTIMIZATION: Short-circuit. If it is a cusp, DTI fails.
    if status_cusp:
        image = stack_images(index, ni, directories)
        image = remove_noise_with_tsvd(image, n_components=n_components)
        return False, ni, image

    # ---------------------------------------------------------
    # TEST 3: Is the magnetopause properly contained in the FOV?
    # ---------------------------------------------------------
    # Uses the fixed ni_mp-frame window, not the optimized ni.
    mp_image = stack_images(index, ni_mp, directories)
    mp_image = remove_noise_with_tsvd(mp_image, n_components=n_components)

    status_mp = test3_is_mp_in_fov(mp_image)

    # ---------------------------------------------------------
    # FINAL EVALUATION
    # ---------------------------------------------------------
    # DTI is valid IF: Structure is found AND it is NOT a cusp AND MP is in FOV.
    status_dti = status_structure and (not status_cusp) and status_mp

    # Report the image reconstructed at the optimized integration time.
    image = stack_images(index, ni, directories)
    image = remove_noise_with_tsvd(image, n_components=n_components)

    return status_dti, ni, image