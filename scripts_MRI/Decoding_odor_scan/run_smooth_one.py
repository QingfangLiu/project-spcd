import os
from nilearn.image import smooth_img, load_img

def smooth_file(input_path, fwhm=6):
    img = load_img(input_path)
    smoothed = smooth_img(img, fwhm=fwhm)
    output_path = input_path.replace(".nii.gz", f"_smooth{fwhm}.nii.gz")
    smoothed.to_filename(output_path)
    print(f"Smoothed: {input_path} -> {output_path}")

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    smooth_file(file_path)
