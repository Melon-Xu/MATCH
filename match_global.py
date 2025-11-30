from collections import defaultdict, deque
import numpy as np

def is_allclose_05(prob_map: np.ndarray, tol=1e-6) -> bool:
    """
    Returns True if all values in `prob_map` are approximately 0.5 within `tol`.
    """
    # Simple approach: check np.allclose(prob_map, 0.5) with an absolute tolerance
    return np.allclose(prob_map, 0.5, atol=tol)

class GlobalMatchGraph:
    """
    Each node is (t, i) = pass index and structure index in that pass.
    adjacency[(t,i)] = list of (t', j) nodes matched to (t,i).
    """
    def __init__(self, num_passes):
        self.num_passes = num_passes
        self.adjacency = defaultdict(list)
        self.pass_features = [{} for _ in range(num_passes)]

    def add_pass_features(self, t, prob_map, pd_valid, bcp_valid, dcp_valid,
                          pd_noisy, bcp_noisy, dcp_noisy,
                          masks, pers):

        # 1) Check if pass is effectively all 0.5
        if is_allclose_05(prob_map):
            # => skip this pass
            self.pass_features[t] = None
            return

        self.pass_features[t] = {
            "prob_map":   prob_map,
            "pd_valid":   np.array(pd_valid),
            "bcp_valid":  np.array(bcp_valid),
            "dcp_valid":  np.array(dcp_valid),
            "pd_noisy":   np.array(pd_noisy),
            "bcp_noisy":  np.array(bcp_noisy),
            "dcp_noisy":  np.array(dcp_noisy),
            "masks":      masks,
            "pers":       np.array(pers)
        }

    def add_edges_for_pass_pair(self, t1, t2,
                                tau_primary=0.1,
                                tau_secondary=0.02,
                                relative_dist=0.3):
        # We call 'multi_phase_overlap_matching' from matching_by_overlap or similar
        from matching_by_overlap import multi_phase_overlap_matching

        f1 = self.pass_features[t1]
        f2 = self.pass_features[t2]

        if f1 is None or f2 is None:
            return 

        primary, secondary, unpaired_1, unpaired_2 = multi_phase_overlap_matching(
            masks1=f1["masks"], pers1=f1["pers"], bcp1=f1["bcp_valid"],
            masks2=f2["masks"], pers2=f2["pers"], bcp2=f2["bcp_valid"],
            tau_primary=tau_primary,
            tau_secondary=tau_secondary,
            relative_dist=relative_dist,
            use_distance_penalty=False
        )
        all_matches = []
        if len(primary)>0:
            all_matches.extend(primary.tolist())
        if len(secondary)>0:
            all_matches.extend(secondary.tolist())

        # Add edges in adjacency
        for (i, j) in all_matches:
            node1 = (t1, i)
            node2 = (t2, j)
            self.adjacency[node1].append(node2)
            self.adjacency[node2].append(node1)

    def find_connected_components(self):
        visited = set()
        components = []
        all_nodes = list(self.adjacency.keys())
        for node in all_nodes:
            if node not in visited:
                queue = deque([node])
                comp = set([node])
                visited.add(node)
                while queue:
                    curr = queue.popleft()
                    for neigh in self.adjacency[curr]:
                        if neigh not in visited:
                            visited.add(neigh)
                            comp.add(neigh)
                            queue.append(neigh)
                components.append(comp)
        return components


def global_multi_pass_matching(mc_probs, pers_thr=0.03,
                               tau_primary=0.1,
                               tau_secondary=0.02):
    """
    Example multi-pass approach:
      1) For each pass, extract topological features + masks.
      2) Build a GlobalMatchGraph.
      3) Add edges between consecutive passes (0->1,1->2,...).
      4) Find connected components.

    mc_probs: (T,H,W) or (T,B,H,W).
              For simplicity, assume (T,H,W) if B=1.
    Returns: 
      comps = list of sets, each set is {(t, i), (t2,i2), ...}.
    """
    import torch
    from matching_by_overlap import get_persistence_diagram, _grow_component_mask

    # If shape is (T,B,H,W), pick B=0 for demonstration
    if len(mc_probs.shape) == 4:
        T, B, H, W = mc_probs.shape
        if B > 1:
            print("Warning: More than one batch dimension. Using b=0 only for demo.")
        mc_probs = mc_probs[:,0]  # shape (T,H,W)

    T, H, W = mc_probs.shape
    graph = GlobalMatchGraph(num_passes=T)

    # (1) Extract pass features
    for t in range(T):
        lh_t = mc_probs[t]  # shape (H,W)
        # get valid/noisy
        pd_v, bcp_v, dcp_v, pd_n, bcp_n, dcp_n = get_persistence_diagram(lh_t, threshold=pers_thr)

        # build masks & pers
        masks = []
        pers_list = []
        for i in range(len(pd_v)):
            mask_i = _grow_component_mask(lh_t, tuple(bcp_v[i].astype(int)),
                                          death_val=pd_v[i,1])
            masks.append(mask_i)
            # A typical measure of significance
            pers_list.append(abs(pd_v[i,1] - pd_v[i,0]))

        graph.add_pass_features(t, pd_v, bcp_v, dcp_v,
                                pd_n, bcp_n, dcp_n,
                                masks, pers_list)

    # (2) Add edges pass by pass
    for t in range(T-1):
        graph.add_edges_for_pass_pair(t, t+1,
                                      tau_primary=tau_primary,
                                      tau_secondary=tau_secondary,
                                      relative_dist=0.3)

    # (3) Connected components
    comps = graph.find_connected_components()
    return graph, comps