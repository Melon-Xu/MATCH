import torch
import numpy as np
from match_pair import (
    improved_match_spatial_overlap,
    get_persistence_diagram,
    _grow_component_mask
)
from match_global import GlobalMatchGraph


def consecutive_pred_global_penalty_graph(
    mc_probs: torch.Tensor,
    device: torch.device = torch.device("cuda"),
    pers_thr: float = 0.03,
    iou_thr: float = 0.1
) -> torch.Tensor:
    """
    A 'global' version of MC-dropout consistency, using a graph-based matching approach.

    1) For each sample in the batch:
       - We parse T MC passes.
       - For each pass, we extract topological structures (valid + noisy).
       - We also build component masks for each valid structure (so we can overlap-match).
       - We add each pass's structures to a GlobalMatchGraph (some passes might be None if all 0.5).
       - Then we add edges between consecutive passes using multi_phase_overlap_matching.
       - We find connected components in that graph.

    2) A connected component is "globally matched" if it contains exactly one valid structure
       from each of the *non-None* passes.
       -> Those structures get birth => 0, death => 1 in all passes.
    3) All other valid structures => push to diagonal; noisy => diagonal.

    Returns a single scalar `total_loss` on the specified device.
    """
    T, B, H, W = mc_probs.shape
    if T < 1:
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=device)
    n_total = 0

    mc_np = mc_probs.detach().cpu().numpy()  # (T,B,H,W)

    for b_idx in range(B):
        # Build a GlobalMatchGraph for the T passes of this sample
        graph = GlobalMatchGraph(num_passes=T)

        # -------------------- 1) Build the graph (skip all-0.5 passes) --------------------
        for t in range(T):
            lh_t = mc_np[t, b_idx]  # shape (H,W)
            # get valid/noisy
            pd_v, bcp_v, dcp_v, pd_n, bcp_n, dcp_n = get_persistence_diagram(
                lh_t, threshold=pers_thr
            )
            # build masks & pers for valid structures
            masks = []
            pers_list = []
            for i in range(len(pd_v)):
                b_pix = tuple(bcp_v[i].astype(int))
                death_val = pd_v[i,1]
                mask_i = _grow_component_mask(lh_t, b_pix, death_val=death_val)
                masks.append(mask_i)
                pers_list.append(abs(pd_v[i,1] - pd_v[i,0]))

            # add to graph; if pass is all 0.5 or if you do a check inside add_pass_features,
            # graph.pass_features[t] might become None
            graph.add_pass_features(
                t,
                prob_map=lh_t,      # so we can skip if it's all 0.5
                pd_valid=pd_v,
                bcp_valid=bcp_v,
                dcp_valid=dcp_v,
                pd_noisy=pd_n,
                bcp_noisy=bcp_n,
                dcp_noisy=dcp_n,
                masks=masks,
                pers=pers_list
            )

        # -------------------- 2) Add edges for consecutive passes --------------------
        for t in range(T - 1):
            graph.add_edges_for_pass_pair(
                t, t+1,
                tau_primary=iou_thr,
                tau_secondary=(iou_thr*0.5),
                relative_dist=0.3
            )

        # -------------------- 3) Find connected components ---------------------------
        comps = graph.find_connected_components()

        # -------------------- 4) Identify fully matched components -------------------
        # pass_matched[t] = set of structure indices that are in a "fully matched" chain
        pass_matched = [set() for _ in range(T)]

        # Because some passes might be None, gather which pass indices are "active"
        active_passes = [ t for t in range(T) if graph.pass_features[t] is not None ]

        # For each connected component => see if it has exactly one structure from each active pass
        local_loss = torch.tensor(0.0, device=device)

        for comp in comps:
            # comp is a set of (pass_idx, structure_idx)
            pass_to_idxs = {}
            for (p_i, idx_p) in comp:
                pass_to_idxs.setdefault(p_i, [])
                pass_to_idxs[p_i].append(idx_p)

            # Check if *all* active passes are in comp, each exactly once
            if len(pass_to_idxs) == len(active_passes) and all(len(pass_to_idxs[p]) == 1 for p in pass_to_idxs):
                # => fully matched across the active passes
                for p_i in active_passes:
                    idx_p = pass_to_idxs[p_i][0]
                    pass_matched[p_i].add(idx_p)

                    b_pt = graph.pass_features[p_i]["bcp_valid"][idx_p]
                    d_pt = graph.pass_features[p_i]["dcp_valid"][idx_p]
                    map_torch = mc_probs[p_i, b_idx]  # shape(H,W) on device
                    local_loss += _push_to_01(map_torch, b_pt, d_pt)

        # -------------------- 5) For all other valid structures => diagonal -----------
        # Also noisy => diagonal
        for t in range(T):
            f = graph.pass_features[t]
            if f is None:
                # pass is skipped => do nothing
                continue

            pd_valid   = f["pd_valid"]
            bcp_valid  = f["bcp_valid"]
            dcp_valid  = f["dcp_valid"]

            pd_noisy   = f["pd_noisy"]
            bcp_noisy  = f["bcp_noisy"]
            dcp_noisy  = f["dcp_noisy"]

            map_torch = mc_probs[t, b_idx]

            # valid unmatched
            for i_v in range(len(pd_valid)):
                if i_v not in pass_matched[t]:
                    b_pt = bcp_valid[i_v]
                    d_pt = dcp_valid[i_v]
                    local_loss += _push_diagonal(map_torch, b_pt, d_pt)

            # noisy => diagonal
            for i_n in range(len(pd_noisy)):
                b_pt = bcp_noisy[i_n]
                d_pt = dcp_noisy[i_n]
                local_loss += _push_diagonal(map_torch, b_pt, d_pt)

        total_loss += local_loss
        n_total += 1

    if n_total > 0:
        total_loss /= n_total

    return total_loss


def _push_to_01(
    map_torch: torch.Tensor,
    b_pt: np.ndarray,
    d_pt: np.ndarray,
) -> torch.Tensor:
    """
    MSE penalty that 'birth pixel' => 0, 'death pixel' => 1.
    """
    H, W = map_torch.shape
    by, bx = int(b_pt[0]), int(b_pt[1])
    dy, dx = int(d_pt[0]), int(d_pt[1])
    loss = torch.tensor(0.0, device=map_torch.device)
    if 0 <= by < H and 0 <= bx < W:
        loss += (map_torch[by,bx] - 0.0)**2
    if 0 <= dy < H and 0 <= dx < W:
        loss += (map_torch[dy,dx] - 1.0)**2
    return loss

def _birthdeath_mse(
    map_torch: torch.Tensor,
    b_pt: np.ndarray, 
    d_pt: np.ndarray,
    b_ref_val: float,
    d_ref_val: float
) -> torch.Tensor:
    """
    MSE that pushes the pass's birth pixel => b_ref_val,
                   pushes the pass's death pixel => d_ref_val.
    """
    H, W = map_torch.shape
    by, bx = int(b_pt[0]), int(b_pt[1]) 
    dy, dx = int(d_pt[0]), int(d_pt[1])

    mse_local = torch.tensor(0.0, device=map_torch.device)
    if 0<=by<H and 0<=bx<W:
        mse_local += (map_torch[by,bx] - b_ref_val)**2
    if 0<=dy<H and 0<=dx<W:
        mse_local += (map_torch[dy,dx] - d_ref_val)**2
    return mse_local

def _push_diagonal(
    map_torch: torch.Tensor,
    b_pt: np.ndarray,
    d_pt: np.ndarray
) -> torch.Tensor:
    """
    MSE that sets 'birth' = 'death'.  i.e. birth_val => death_val, death_val => birth_val
    in a symmetrical sense.  We'll do (birth-death)^2 for the two coords.
    """
    H, W = map_torch.shape
    by, bx = int(b_pt[0]), int(b_pt[1])
    dy, dx = int(d_pt[0]), int(d_pt[1])
    if (by<0 or by>=H or bx<0 or bx>=W or dy<0 or dy>=H or dx<0 or dx>=W):
        return torch.tensor(0.0, device=map_torch.device)

    val_b = map_torch[by,bx]
    val_d = map_torch[dy,dx]
    return (val_b - val_d)**2