import numpy as np
import time

class NearestCentroidNeighbor:
    def __init__(
        self,
        k: int=5,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.k = k
        self.verbose = verbose
        self.time = time
        self.random_state = random_state

    def compute_euclidean_distances(self, point, candidates):
        diff = candidates - point  # broadcast: (n_candidates, n_features)
        return np.sqrt((diff ** 2).sum(axis=1))


    def neighbors(self, query_idx, X, k, exclude_self = True):
        query_point = X[query_idx]
        n_samples_total = X.shape[0]

        # Build pool of candidate indices (exclude self if requested)
        if exclude_self:
            candidates = np.array([i for i in range(n_samples_total) if i != query_idx])
        else:
            candidates = np.arange(n_samples_total)

        # Safety: cannot select more neighbours than available candidates
        k_actual = min(k, len(candidates))

        selected_indices = []          # NCN set (grows at each step)
        selected_points = []           # Corresponding feature vectors

        remaining = list(candidates)   # Candidates not yet selected

        for step in range(k_actual):
            if step == 0:
                # Step 1: First neighbour = simple nearest neighbour
                remaining_array = np.array(remaining)
                dists = self.compute_euclidean_distances(query_point, X[remaining_array])
                best_local_idx = np.argmin(dists)           # index into remaining_array
                best_global_idx = remaining_array[best_local_idx]
            else:
                # Step 2+: Find candidate that minimises centroid distance to query
                # current_centroid = np.mean(selected_points, axis=0)  # shape (n_features,)
                remaining_array = np.array(remaining)

                best_centroid_dist = np.inf
                best_global_idx = None

                for cand_global_idx in remaining_array:
                    # Tentative centroid if we add this candidate
                    tentative_points = selected_points + [X[cand_global_idx]]
                    tentative_centroid = np.mean(tentative_points, axis=0)
                    centroid_dist = np.linalg.norm(tentative_centroid - query_point)

                    if centroid_dist < best_centroid_dist:
                        best_centroid_dist = centroid_dist
                        best_global_idx = cand_global_idx

            # Add best candidate to NCN set
            selected_indices.append(best_global_idx)
            selected_points.append(X[best_global_idx].copy())
            remaining.remove(best_global_idx)

        selected_indices = np.array(selected_indices)

        # Compute final Euclidean distances from query to each selected neighbour
        final_distances = self.compute_euclidean_distances(query_point, X[selected_indices])

        # Order by increasing Euclidean distance (nearest first)
        order = np.argsort(final_distances)
        return selected_indices[order], final_distances[order]


    def build_neighborhood_cache(self,X, k):
        n = X.shape[0]
        cache = {}
        if self.verbose:
            print(f"  Building NCN cache (n={n}, k={k}) ...", end="", flush=True)
        t0 = self.time.time()
        for i in range(n):
            cache[i] = self.neighbors(i, X, k)
        elapsed = self.time.time() - t0
        if self.verbose:
            print(f" done in {elapsed:.2f}s")
        return cache