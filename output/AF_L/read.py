# import os
# import pickle 

# with open('/tracto/TractoDiff/output/AF_L/sub-1030.pkl', 'rb') as file:
#     data = pickle.load(file)

# print((data[2]['observations'].shape))

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Path to your NIfTI WM mask
nifti_path = '/tracto/TractoDiff/data/trainset/sub-1030/sub-1030-generated_approximated_mask_1mm.nii.gz'


# Load the image
img = nib.load(nifti_path)
data = img.get_fdata()  # float64 by default

# Print details
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Unique values:", np.unique(data))

# Extract middle axial slice
mid_slice = data.shape[2] // 2
slice_data = data[:, :, mid_slice]

# Save as PNG
plt.imshow(slice_data, cmap='gray')
plt.title(f"WM Mask - Slice {mid_slice}")
plt.axis('off')
plt.savefig(f'wm_mask_slice_{mid_slice}.png', bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Saved: wm_mask_slice_{mid_slice}.png")