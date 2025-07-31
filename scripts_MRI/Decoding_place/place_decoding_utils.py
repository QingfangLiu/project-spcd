
import os
import pandas as pd
import numpy as np
from nilearn.image import load_img, concat_imgs, get_data, new_img_like

def load_betas(sub, run, base_dir, normalization='none'):
    beta_dir = os.path.join(base_dir, 'GLM', 'place_trialwise_2mm', sub)
    tsv_dir = os.path.join(base_dir, 'nii', sub, 'func_PAID', 'place')
    tsv_file = os.path.join(tsv_dir, f'{sub}_task-place_run-{run}_events.tsv')

    events = pd.read_csv(tsv_file, sep='\t')
    trial_types = events['trial_type'].values

    imgs = []
    for i in range(len(trial_types)):
        beta_file = os.path.join(beta_dir, f'beta_run{run}_trial{i+1:02d}.nii')
        imgs.append(load_img(beta_file))

    # Normalization
    if normalization == 'across_trials':
        data_stack = np.stack([get_data(img) for img in imgs])
        mean = data_stack.mean(axis=0)
        std = data_stack.std(axis=0)
        imgs = [new_img_like(img, (get_data(img) - mean) / std) for img in imgs]
    elif normalization == 'across_voxels':
        imgs = [new_img_like(img, (get_data(img).flatten() - get_data(img).mean()) / get_data(img).std()) for img in imgs]
    elif normalization != 'none':
        raise ValueError(f"Unknown normalization: {normalization}")

    return imgs, trial_types

def load_decoding_data(sub, base_dir, run_list=[1, 2, 3, 4], normalization='none'):
    all_imgs = []
    all_labels = []
    all_groups = []

    for run in run_list:
        imgs, labels = load_betas(sub, run, base_dir, normalization)
        all_imgs.extend(imgs)
        all_labels.extend(labels)
        all_groups.extend([run] * len(labels))

    # Keep only top 6 most frequent labels
    label_counts = pd.Series(all_labels).value_counts()
    top_labels = label_counts.nlargest(6).index.tolist()
    mask = [label in top_labels for label in all_labels]

    filtered_imgs = [img for img, m in zip(all_imgs, mask) if m]
    filtered_labels = [label for label in all_labels if label in top_labels]
    filtered_groups = [group for group, m in zip(all_groups, mask) if m]

    label_map = {label: i for i, label in enumerate(sorted(top_labels))}
    numeric_labels = [label_map[label] for label in filtered_labels]

    return concat_imgs(filtered_imgs), np.array(numeric_labels), np.array(filtered_groups)
