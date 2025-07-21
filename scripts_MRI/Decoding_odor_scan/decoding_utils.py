import os
import pandas as pd
import numpy as np
from nilearn.image import load_img

def load_betas(sub, run, base_dir):
    """
    Load beta images and binary labels (0 = odor_1, 1 = odor_2) for one run.
    """
    tsv_dir = os.path.join(base_dir, 'nii', sub, 'func_PAID', 'odor_scan')
    beta_dir = os.path.join(base_dir, 'GLM', 'odor_scan_trialwise_2mm', sub)
    tsv_file = os.path.join(tsv_dir, f'{sub}_task-odor_run-{run}_events.tsv')

    events = pd.read_csv(tsv_file, sep='\t')
    events = events[events['trial_type'].isin(['odor_1', 'odor_2'])].copy()
    events['label'] = events['trial_type'].map({'odor_1': 0, 'odor_2': 1})

    imgs = []
    for i in range(len(events)):
        trial_num = i + 1
        beta_file = os.path.join(beta_dir, f'beta_run{run}_trial{trial_num:02d}.nii')
        imgs.append(load_img(beta_file))

    return imgs, events['label'].values

def load_decoding_data(sub, base_dir, run_list=[1, 2]):
    """
    Load and stack beta images and labels for decoding across multiple runs.

    Parameters
    ----------
    sub : str
        Subject ID (e.g., "sub-01")
    base_dir : str
        Base path containing 'GLM' and 'nii' folders
    run_list : list of int
        List of runs to load (default: [1, 2])

    Returns
    -------
    all_imgs : 4D Nifti image
        Concatenated beta images
    all_y : np.array
        Combined labels (0 for odor_1, 1 for odor_2)
    groups : np.array
        Group labels indicating which run each trial came from
    """
    from nilearn.image import concat_imgs
    import numpy as np
    
    all_imgs_list = []
    all_labels = []
    all_groups = []

    for run in run_list:
        imgs, y = load_betas(sub, run, base_dir)
        all_imgs_list.extend(imgs)
        all_labels.extend(y)
        all_groups.extend([run] * len(y))

    all_imgs = concat_imgs(all_imgs_list)
    all_y = np.array(all_labels)
    groups = np.array(all_groups)

    return all_imgs, all_y, groups


