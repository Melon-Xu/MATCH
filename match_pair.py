import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from ripser import ripser
import cripser as cr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import os

def get_persistence_diagram(likelihood, threshold=0.03):
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth")
    pd_arr_lh = pd[pd[:, 0] == 0]  # 0-dim topological features
    pd_lh = pd_arr_lh[:, 1:3]  # birth time and death time

    # birth critical points
    bcp_lh = pd_arr_lh[:, 3:5]
    # death critical points
    dcp_lh = pd_arr_lh[:, 6:8]

    # Handle infinite death times
    for i in range(len(pd_lh)):
        if pd_lh[i, 1] > 1.0:
            pd_lh[i, 1] = 1.0
            # Check if we need to set a default death critical point
            if np.isnan(dcp_lh[i]).any() or np.isinf(dcp_lh[i]).any():
                print(f"Warning: Invalid death point at index {i}, setting default")
                dcp_lh[i] = np.array([0.0, 0.0])  # Set a default value

    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])
    # Filter out the persistence pairs with too small persistence
    valid_idx = np.where(pd_pers > threshold)[0]
    noisy_idx = np.where(pd_pers <= threshold)[0]

    pd_lh_valid = pd_lh[valid_idx]
    bcp_lh_valid = bcp_lh[valid_idx]
    dcp_lh_valid = dcp_lh[valid_idx]

    pd_lh_noisy = pd_lh[noisy_idx]
    bcp_lh_noisy = bcp_lh[noisy_idx]
    dcp_lh_noisy = dcp_lh[noisy_idx]

    #return pd_lh, bcp_lh, dcp_lh
    return pd_lh_valid, bcp_lh_valid, dcp_lh_valid, pd_lh_noisy, bcp_lh_noisy, dcp_lh_noisy


def _grow_component_mask(likelihood, birth_pt, death_val, 
                         bin_eps=1e-6, connectivity=8):
    """
    Reconstruct an approximate mask of the connected component that
    corresponds to a persistence pair.

    Parameters
    ----------
    likelihood : (H,W) array   – original *likelihood* (not inverted).
    birth_pt   : (y,x)         – pixel where the component is born.
    death_val  : float         – death threshold (on 'lh' = 1-likelihood).
                                 In original likelihood domain that value is
                                 1 - death_val.
    bin_eps    : float         – small epsilon to stay strictly above the
                                 death threshold.
    connectivity : int         – 1 (4‑nbhd) or 2 (8‑nbhd).

    Returns
    -------
    mask : bool (H,W) – True for pixels in the component.
    """
    H, W = likelihood.shape
    # Component is alive for likelihood >= (1 - death_val)
    thresh = 1.0 - death_val - bin_eps
    bin_img = likelihood >= thresh

    # Flood‑fill from the birth pixel within the binary image
    mask = np.zeros_like(bin_img, dtype=bool)
    if not bin_img[birth_pt]:
        # safety: birth pixel might be filtered out by eps – relax threshold
        bin_img[birth_pt] = True
    ndi.binary_fill_holes(bin_img, structure=np.ones((3,3))).astype(bool)
    # label connected components
    labeled, num = ndi.label(bin_img, structure=ndi.generate_binary_structure(2, connectivity))
    comp_label = labeled[birth_pt]
    if comp_label == 0:          # should not happen, but guard
        mask[birth_pt] = True
    else:
        mask = labeled == comp_label
    return mask

def match_by_spatial_overlap(lh1, lh2, iou_thr=0.1, pers_thr=0.03):
    """
    Parameters
    ----------
    lh1, lh2 : (H,W) arrays  – likelihood maps of two segmentations.
    iou_thr  : float         – minimum IoU to be considered a match.
    pers_thr : float         – persistence filter used in get_PD.

    Returns
    -------
    matched_pairs : list[(i,j,float)]  – (index_in_1 , index_in_2 , IoU)
    unmatched_1   : list[int]
    unmatched_2   : list[int]
    """
    # --- extract diagrams ---------------------------------------------
    pd1, bcp1, dcp1, pd1_n, bcp1_n, dcp1_n = get_persistence_diagram(lh1, threshold=pers_thr)
    pd2, bcp2, dcp2, pd2_n, bcp2_n, dcp2_n = get_persistence_diagram(lh2, threshold=pers_thr)

    n1, n2 = len(pd1), len(pd2)

    # If no structures on either side, skip
    if n1 == 0 or n2 == 0:
        matched_pairs = np.empty((0, 2), dtype=int)
        unmatched_1 = list(range(n1))  # all in 1 are unmatched
        unmatched_2 = list(range(n2))  # all in 2 are unmatched
        return matched_pairs, unmatched_1, unmatched_2

    # --- build masks ---------------------------------------------------
    masks1 = []
    for i in range(n1):
        mask_i = _grow_component_mask(lh1, tuple(bcp1[i].astype(int)),
                                      death_val=pd1[i,1])
        masks1.append(mask_i)

    masks2 = []
    for j in range(n2):
        mask_j = _grow_component_mask(lh2, tuple(bcp2[j].astype(int)),
                                      death_val=pd2[j,1])
        masks2.append(mask_j)

    # --- compute IoU matrix -------------------------------------------
    iou_mat = np.zeros((n1, n2))
    for i in range(n1):
        m1 = masks1[i]
        area1 = m1.sum()
        if area1 == 0:
            continue
        for j in range(n2):
            inter = np.logical_and(m1, masks2[j]).sum()
            if inter == 0:
                continue
            union = area1 + masks2[j].sum() - inter
            iou_mat[i, j] = inter / union

    # --- greedy matching by highest IoU -------------------------------
    matched_pairs = []
    used_i = set()
    used_j = set()
    while True:
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        max_iou = iou_mat[idx]
        if max_iou < iou_thr:
            break
        i, j = idx
        #matched_pairs.append((i, j, float(max_iou)))
        matched_pairs.append([i, j])
        used_i.add(i)
        used_j.add(j)
        iou_mat[i, :] = -1     # invalidate row
        iou_mat[:, j] = -1     # invalidate col

    unmatched_1 = [i for i in range(n1) if i not in used_i]
    unmatched_2 = [j for j in range(n2) if j not in used_j]

    if len(matched_pairs) > 0:
        matched_pairs = np.asarray(matched_pairs, dtype=int)
        matched_pairs = matched_pairs[np.argsort(matched_pairs[:, 0])]
    else:
        matched_pairs = np.empty((0, 2), dtype=int) 

    return matched_pairs, unmatched_1, unmatched_2



def multi_phase_overlap_matching(
    masks1, pers1, bcp1,
    masks2, pers2, bcp2,
    # Primary matching thresholds
    tau_primary=0.10,
    # Secondary relaxed matching thresholds
    tau_secondary=0.02,
    # Optional relative distance cutoff in the primary pass
    relative_dist=None,
    # Optionally incorporate a continuous distance penalty
    use_distance_penalty=False
):
    """
    Performs a two-phase matching of persistent components:
      1) Global Hungarian assignment with a stricter Score threshold.
      2) A relaxed many-to-many pass to catch partial matches (merges/splits)
         that have lower overlap but are still valid.

    Parameters
    ----------
    masks1, masks2 : list of bool arrays, shape (H, W)
        Binary masks for each persistent component in map1, map2.
    pers1, pers2   : list of floats
        Persistence (topological significance) for each component in map1, map2.
    bcp1, bcp2     : arrays of shape (n1,2) and (n2,2)
        Birth-critical-point coordinates (y, x) for each component in map1, map2.
    tau_primary    : float
        Minimum Score for the Hungarian 1‑to‑1 assignment.
    tau_secondary  : float
        Minimum Score for the relaxed second pass (many‑to‑many).
    relative_dist  : float or None
        If provided, we compute the max pairwise distance among (bcp1,bcp2),
        multiply by relative_dist, and disallow matches beyond that threshold.
    use_distance_penalty : bool
        If True, instead of a hard cutoff, incorporate a continuous penalty
        that lowers the Score for more distant pairs.

    Returns
    -------
    primary_pairs : ndarray of shape (k, 2)
        Indices (i, j) matched in the primary pass (Hungarian).
    secondary_pairs : ndarray of shape (m, 2)
        Indices (i, j) matched in the secondary relaxed pass.
    unpaired_1 : list of int
        Unmatched structure indices from map1 (after both passes).
    unpaired_2 : list of int
        Unmatched structure indices from map2 (after both passes).
    """

    n1, n2 = len(masks1), len(masks2)
    # If either map has no components, nothing to match
    if n1 == 0 or n2 == 0:
        return (np.empty((0,2),dtype=int),
                np.empty((0,2),dtype=int),
                list(range(n1)), list(range(n2)))

    #---------------------------------------------------
    # (1) Compute IoU
    #---------------------------------------------------
    IoU = np.zeros((n1, n2), dtype=float)
    area1 = np.array([m.sum() for m in masks1])
    area2 = np.array([m.sum() for m in masks2])
    for i in range(n1):
        if area1[i] == 0:
            continue
        m1 = masks1[i]
        for j in range(n2):
            inter = (m1 & masks2[j]).sum()
            if inter == 0:
                continue
            union = area1[i] + area2[j] - inter
            IoU[i, j] = inter / union

    #---------------------------------------------------
    # (2) Persistence weighting
    #---------------------------------------------------
    # If not provided or empty, default them to 1 so as not to break code
    if pers1 is None or len(pers1) == 0:
        pers1 = np.ones(n1)
    if pers2 is None or len(pers2) == 0:
        pers2 = np.ones(n2)

    w1 = pers1 / (pers1.max() + 1e-9)
    w2 = pers2 / (pers2.max() + 1e-9)
    Score = IoU * (w1[:, None] * w2[None, :])

    #---------------------------------------------------
    # (3) Distance factor / cutoff
    #---------------------------------------------------
    # Use a distance matrix from birth points
    dist_mat = cdist(bcp1, bcp2, metric='euclidean')
    max_dist = dist_mat.max() + 1e-9

    if relative_dist is not None:
        # Hard cutoff based on fraction of max distance
        dist_cutoff = relative_dist * max_dist
        Score[dist_mat > dist_cutoff] = 0.0

    if use_distance_penalty and max_dist > 0:
        # Example penalty: multiply Score by (1 - dist/max_dist)
        dist_factor = 1.0 - (dist_mat / max_dist)
        Score *= dist_factor

    #---------------------------------------------------
    # (4) Hungarian assignment (primary pass)
    #     Minimizing cost = (1 - Score)
    #---------------------------------------------------
    Cost = 1.0 - Score
    row_ind, col_ind = linear_sum_assignment(Cost)

    used_rows = set()
    used_cols = set()
    primary_pairs = []
    for (i, j) in zip(row_ind, col_ind):
        sc = Score[i, j]
        if sc >= tau_primary:
            primary_pairs.append([i, j])
            used_rows.add(i)
            used_cols.add(j)

    #---------------------------------------------------
    # (5) Secondary pass (many-to-many)
    #     Relaxed threshold for partial matches
    #     This ensures we don't miss smaller or partial
    #     overlaps if they've been left unmatched.
    #---------------------------------------------------
    secondary_pairs = []
    for i in range(n1):
        if i in used_rows:
            continue  # skip if already matched in primary
        for j in range(n2):
            if j in used_cols:
                continue  # skip if map2's j was used in primary
            sc = Score[i, j]
            # If we exceed the smaller threshold, we can match
            if sc >= tau_secondary:
                # In many-to-many, we do NOT automatically exclude i or j
                # from further matches unless desired. If you want only one
                # match per structure, you can add 'used_rows.add(i)' etc.
                secondary_pairs.append([i, j])

    # Collect final unmatched sets
    # If you allow many matches in the secondary pass (no set updates),
    # then a structure can remain "unmatched" if it never gets any secondary match.
    matched_rows = set([p[0] for p in primary_pairs]) | set([s[0] for s in secondary_pairs])
    matched_cols = set([p[1] for p in primary_pairs]) | set([s[1] for s in secondary_pairs])
    unpaired_1 = [i for i in range(n1) if i not in matched_rows]
    unpaired_2 = [j for j in range(n2) if j not in matched_cols]

    # Convert results to arrays, sorted by the first dimension
    if len(primary_pairs) > 0:
        primary_pairs = np.array(primary_pairs, dtype=int)
        primary_pairs = primary_pairs[np.argsort(primary_pairs[:,0])]
    else:
        primary_pairs = np.empty((0,2), dtype=int)

    if len(secondary_pairs) > 0:
        secondary_pairs = np.array(secondary_pairs, dtype=int)
        secondary_pairs = secondary_pairs[np.argsort(secondary_pairs[:,0])]
    else:
        secondary_pairs = np.empty((0,2), dtype=int)

    return primary_pairs, secondary_pairs, unpaired_1, unpaired_2


def improved_match_spatial_overlap(lh1, lh2, pers_thr=0.03, tau_primary=0.1, tau_multi=0.05):
    """
    Full pipeline: 
      1) Extract 0D features from each likelihood map (pd,bcp,dcp),
      2) Rebuild their pixel masks,
      3) Weighted IoU + Hungarian for a global 1‑to‑1,
      4) Many‑to‑many second pass if needed.

    Returns
    -------
    primary_pairs, secondary, unpaired_1, unpaired_2
      where each pair is (idx_in_map1, idx_in_map2, Score).
    """
    # 1) diagrams
    pd1, bcp1, dcp1, pd1_noisy, bcp1_noisy, dcp1_noisy = get_persistence_diagram(lh1, threshold=pers_thr)
    pd2, bcp2, dcp2, pd2_noisy, bcp2_noisy, dcp2_noisy = get_persistence_diagram(lh2, threshold=pers_thr)

    # gather per-component persistence (or use others if needed)
    pers1 = np.abs(pd1[:,1] - pd1[:,0]) if len(pd1)>0 else []
    pers2 = np.abs(pd2[:,1] - pd2[:,0]) if len(pd2)>0 else []

    # 2) build masks for each persistent feature
    masks1 = []
    for i in range(len(pd1)):
        mask_i = _grow_component_mask(lh1, tuple(bcp1[i].astype(int)),
                                      death_val=pd1[i,1])
        masks1.append(mask_i)

    masks2 = []
    for j in range(len(pd2)):
        mask_j = _grow_component_mask(lh2, tuple(bcp2[j].astype(int)),
                                      death_val=pd2[j,1])
        masks2.append(mask_j)

    primary, secondary, unpaired_1, unpaired_2 = multi_phase_overlap_matching(
        masks1, pers1, bcp1,
        masks2, pers2, bcp2,
        tau_primary=0.10,     # stricter
        tau_secondary=0.02,   # more lenient
        #relative_dist=0.3,    # optional fraction, e.g. 30% of max distance
        relative_dist=None,
        use_distance_penalty=True
    )


    return primary, secondary, unpaired_1, unpaired_2

