import numpy as np
from astropy.io import fits
import os
import pandas as pd
from pathlib import Path

from hotpants import Hotpants, HotpantsConfig, SubstampStatus

print("HOTPanTS Python Wrapper - Complete Implementation")
print("=================================================")

# Load real example data (required)
print("Loading example real data...")

# Search current working directory and parents for example/data
cwd = Path.cwd()
example_path = None
for p in [cwd] + list(cwd.parents):
    candidate = p / "example" / "data"
    if candidate.exists():
        example_path = candidate
        break

if example_path is None:
    raise FileNotFoundError("example/data directory not found. Please add real FITS files before running this notebook.")

print(f"Found example data directory: {example_path}")
files = os.listdir(example_path)
ref_files = [f for f in files if "ref_output" in f and f.endswith(".fits")]
sci_files = [f for f in files if "sci_output" in f and f.endswith(".fits")]
if not (ref_files and sci_files):
    raise FileNotFoundError("Suitable reference/science FITS files not found in example/data.")

ref_file = str(example_path / ref_files[0])
ref_err_file = str(example_path / ref_files[1])
ref_mask_file = str(example_path / ref_files[2])
sci_file = str(example_path / sci_files[0])
sci_err_file = str(example_path / sci_files[1])
# sci_mask_file = str(example_path / sci_files[2])
print(f"  Reference: {ref_file}, {ref_err_file}, {ref_mask_file}")
print(f"  Science:   {sci_file}, {sci_err_file}")

with fits.open(ref_file) as hdul:
    template = hdul[0].data.astype(np.float32)
with fits.open(ref_err_file) as hdul:
    template_err = hdul[0].data.astype(np.float32)
with fits.open(ref_mask_file) as hdul:
    template_mask = hdul[0].data.astype(np.int16)
with fits.open(sci_file) as hdul:
    science = hdul[0].data.astype(np.float32)
with fits.open(sci_err_file) as hdul:
    science_err = hdul[0].data.astype(np.float32)
science_mask = np.zeros_like(science)  # Placeholder for science mask

catalog = pd.read_csv(example_path / "stamp_catalog.txt", sep=" ")
catalog_list = catalog[["X", "Y"]].to_numpy()

# Simple one-line usage
print("\nRunning HOTPanTS pipeline...")

hotpants_config = HotpantsConfig(ko=4, bgo=3, ssig=3.0, ks=5.0, kfm=0.9, nstampx=10, nstampy=10, nss=10, ft=2.0, force_convolve="t", normalize="i", rkernel=3, rss=3, verbose=0)
print(f"HOTPanTS configuration created: {hotpants_config}")
hotpants = Hotpants(template_data=template, image_data=science, t_error=template_err, i_error=science_err, t_mask=template_mask, i_mask=science_mask, config=hotpants_config, star_catalog=catalog_list)

print(f"HOTPanTS instance created: {hotpants}")
hotpants.find_stamps()
conv_direction = hotpants.fit_and_select_direction()
print(f"Convolution direction: {conv_direction}")
hotpants.iterative_fit_and_clip()
hotpants.convolve_and_difference()
results = hotpants.get_final_outputs()

diff = results["diff_image"]
conv = results["convolved_image"]
diff_error = results["noise_image"]
diff_mask = results["output_mask"]
diff_stats = results["stats"]
conv_direction = results["conv_direction"]
kernel_solution = results["kernel_solution"]
fit_stats = results.get("fit_stats")

out_dir = example_path / "out"  # Path object from above
out_dir.mkdir(exist_ok=True)

# Add a small header with metadata
hdr = fits.Header()
hdr["CONVDIR"] = conv_direction
hdr["COMMENT"] = "Produced by HOTPanTS Python wrapper"

# Ensure types are suitable for FITS and write files
fits.PrimaryHDU(diff.astype(np.float32), header=hdr).writeto(str(out_dir / "diff_image.fits"), overwrite=True)
fits.PrimaryHDU(conv.astype(np.float32), header=hdr).writeto(str(out_dir / "convolved_image.fits"), overwrite=True)
fits.PrimaryHDU(diff_error.astype(np.float32), header=hdr).writeto(str(out_dir / "diff_noise.fits"), overwrite=True)
# Masks are typically integer-valued
fits.PrimaryHDU(diff_mask.astype(np.int16), header=hdr).writeto(str(out_dir / "diff_mask.fits"), overwrite=True)

print(f"Saved diff, convolved image, noise and mask to: {out_dir}")


print("\n--- Showcasing Substamp Details ---")
substamp_details = hotpants.get_substamp_details()

# Find the first substamp that was actually used in the final fit
chosen_substamp = None
substamp_source_list = substamp_details["template_substamps"] if conv_direction == "t" else substamp_details["image_substamps"]

for s in substamp_source_list:
    if s.status == SubstampStatus.USED_IN_FINAL_FIT:
        chosen_substamp = s
        break

if chosen_substamp:
    print("\nInspecting a single substamp used in the final fit:")
    print(f"  Details: {chosen_substamp}")
    print(f"  Fit Results: {chosen_substamp.fit_results}")

    # Save the cutouts for this specific substamp
    if chosen_substamp.template_cutout is not None:
        fits.PrimaryHDU(chosen_substamp.template_cutout.astype(np.float32)).writeto(str(out_dir / f"substamp_{chosen_substamp.id}_template.fits"), overwrite=True)
    if chosen_substamp.image_cutout is not None:
        fits.PrimaryHDU(chosen_substamp.image_cutout.astype(np.float32)).writeto(str(out_dir / f"substamp_{chosen_substamp.id}_image.fits"), overwrite=True)
    if chosen_substamp.noise_variance_cutout is not None:
        fits.PrimaryHDU(chosen_substamp.noise_variance_cutout.astype(np.float32)).writeto(str(out_dir / f"substamp_{chosen_substamp.id}_noise_var.fits"), overwrite=True)

    print(f"  Saved image, template, and noise cutouts for substamp ID {chosen_substamp.id} to: {out_dir}")
else:
    print("\nCould not find a substamp that was used in the final fit to display.")


print("\nPIPELINE COMPLETE.")
