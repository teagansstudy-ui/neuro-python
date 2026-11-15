from nilearn import datasets
import nibabel as nib

template = datasets.load_mni152_template()

print("Shape:", template.shape)
print("Voxel size:", template.header.get_zooms())

nib.save(template, "mni_template.nii.gz")
print("Saved mni_template.nii.gz")
import os
print("Saved in:", os.getcwd())