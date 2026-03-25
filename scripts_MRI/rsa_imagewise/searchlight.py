import os
import numpy as np

from nilearn.image import concat_imgs, new_img_like, load_img

from .io_imagewise import build_imagewise_patterns
from .estimator import SingleModelRSA
from .models import canonical_condition_order, create_identity_model_rdm


def _expand_base_rdm_to_imagewise(base_rdm, imagewise_labels, base_order):
    base_lookup = {c: i for i, c in enumerate(base_order)}

    def base_cond(label):
        return label.rsplit("_img", 1)[0]

    n = len(imagewise_labels)
    out = np.zeros((n, n), dtype=float)

    for i in range(n):
        bi = base_cond(str(imagewise_labels[i]))
        for j in range(i):
            bj = base_cond(str(imagewise_labels[j]))
            out[i, j] = base_rdm[base_lookup[bi], base_lookup[bj]]
            out[j, i] = out[i, j]

    np.fill_diagonal(out, 0.0)
    return out


def _create_button_rdm(button_labels):
    labels = [str(x) for x in button_labels]
    n = len(labels)
    rdm = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i):
            rdm[i, j] = 0.0 if labels[i] == labels[j] else 1.0
            rdm[j, i] = rdm[i, j]

    np.fill_diagonal(rdm, 0.0)
    return rdm


def searchlight_rsa_single_subject(
    sub: str,
    base_dir: str,
    session_folder: str,
    session_tag: str,
    radius: float,
    n_jobs: int,
    button_scope: str = "both_images",
    center_within_run: bool = False,
):
    from nilearn.decoding import SearchLight

    patterns, run_ids, cond_ids, button_labels, sample_labels, img_list = build_imagewise_patterns(
        sub=sub,
        base_dir=base_dir,
        session_folder=session_folder,
        session_tag=session_tag,
        button_scope=button_scope,
    )

    n_samples = len(img_list)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for searchlight RSA.")

    if (
        len(run_ids) != n_samples
        or len(cond_ids) != n_samples
        or len(button_labels) != n_samples
        or len(sample_labels) != n_samples
    ):
        raise ValueError("Loader output lengths do not match number of image samples.")

    identity_base = create_identity_model_rdm()
    identity_rdm = _expand_base_rdm_to_imagewise(
        identity_base,
        imagewise_labels=sample_labels,
        base_order=canonical_condition_order,
    )
    button_rdm = _create_button_rdm(button_labels)

    model_rdms = {
        "identity": identity_rdm,
        "button": button_rdm,
    }

    img_4d = concat_imgs(img_list)

    mask_file = os.path.join(base_dir, "masks", "gm_0.1.nii")
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    mask_img = load_img(mask_file)

    estimator = SingleModelRSA(
        model_rdms=model_rdms,
        target_name="identity",
        odor_mask=None,
        run_ids=run_ids,
        cond_ids=cond_ids,
        center_within_run=center_within_run,
    )

    searchlight = SearchLight(
        mask_img=mask_img,
        radius=radius,
        estimator=estimator,
        cv=[(np.arange(n_samples), np.arange(n_samples))],
        n_jobs=n_jobs,
        verbose=1,
    )

    y_dummy = np.arange(n_samples)
    searchlight.fit(img_4d, y_dummy)

    scores_img = new_img_like(mask_img, searchlight.scores_)
    return {"identity": scores_img}
    