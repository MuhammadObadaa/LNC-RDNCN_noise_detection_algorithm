from typing import List
import random
import numpy as np
import time
from collections import defaultdict, Counter

from data_classes.sample_decision import SampleDecision
from data_classes.iteration_record import IterationRecord
import NCN
import importlib
importlib.reload(NCN)
NearestCentroidNeighbor = NCN.NearestCentroidNeighbor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold


class NoiseClassifier:
    def __init__(
        self,
        k_ncn: int=5,
        voting_threshold: float = 0.5,
        classifiers: List[object] = None,
        n_filters: int = 5,
        cv_folds: int = 5,
        max_iterations: int = 20,
        termination_delta: float = 0.0,
        adaptive_filter_threshold:float = 0.2,
        # rns_correct_threshold: str = 'mean',
        random_state: int = 42,
        verbose: bool = False
    ):

        # TODO: make customizable nearest samples algorithm (NCN, KNN)
        self.k_ncn = k_ncn
        self.ncn = NearestCentroidNeighbor(k=self.k_ncn)
        self.voting_threshold = voting_threshold
        self.cv_folds = cv_folds
        self.max_iterations = max_iterations
        self.termination_delta = termination_delta
        self.adaptive_filter_threshold = adaptive_filter_threshold
        self.verbose = verbose
        self.time = time
        self.random_state = random_state

        if classifiers != None:
            self.classifiers = classifiers
        else:
            self.n_filters = n_filters
            self.classifiers = random.sample([
                KNeighborsClassifier(n_neighbors=1),
                KNeighborsClassifier(n_neighbors=3),
                DecisionTreeClassifier(
                    min_samples_leaf=2,
                    random_state=random_state
                ),
                RandomForestClassifier(
                    n_estimators=25,
                    random_state=random_state
                ),
                SVC(kernel='rbf',C=100,gamma=0.3,
                    probability=False,
                    random_state=random_state    
                )
            ],self.n_filters)
            
            if n_filters > len(self.classifiers):
                #TODO: need to be implemented
                n_filters = len(self.classifiers)

    def apply_single_filter( self, X, y, classifier, cv_folds):
        n = len(y)
        # Use cross_val_predict: predictions for each sample when it's in test fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        try:
            y_pred = cross_val_predict(classifier, X, y, cv=skf)
        except Exception:
            # Fallback: leave-one-out approximation via simple prediction
            y_pred = y.copy()

        noise_set = set()
        recommended_labels = {}

        for i in range(n):
            if y_pred[i] != y[i]:
                noise_set.add(i)
                # recommended_labels[i] = int(y_pred[i])
                recommended_labels[i] = y_pred[i]

        return noise_set, recommended_labels


    def run_ensemble_filters(self, X, y):

        filter_configs = [(f"model_{i+1}_{self.classifiers[i].__class__.__name__}",self.classifiers[i]) for i in range(len(self.classifiers))]

        all_noise_sets = []
        all_recommended_labels = []

        if self.verbose:
            print("\n  Running ensemble filters:")

        for name, clf in filter_configs:
            ns, rl = self.apply_single_filter(X, y, clf, self.cv_folds)
            all_noise_sets.append(ns)
            all_recommended_labels.append(rl)
            if self.verbose:
                print(f"\t{name:12s}: {len(ns):4d} suspected noisy samples")

        return all_noise_sets, all_recommended_labels


    def construct_final_noise_set(self, all_noise_sets, all_recommended_labels, n_samples):
        # Count how many filters flag each sample
        vote_counts = np.zeros(n_samples, dtype=int)
        for ns in all_noise_sets:
            for idx in ns:
                vote_counts[idx] += 1

        # Majority threshold: strictly more than half (paper: > n/2 for n=5 → ≥3)
        threshold = self.n_filters * self.voting_threshold  # 5 * 0.5 = 2.5 → ≥3

        NF = set(int(i) for i in np.where(vote_counts > threshold)[0])

        # Collect recommended labels from all filters that flagged each sample
        C_combined = defaultdict(list)
        for i, (ns, rl) in enumerate(zip(all_noise_sets, all_recommended_labels)):
            for idx in NF:
                if idx in ns and idx in rl:
                    C_combined[idx].append(rl[idx])

        return NF, dict(C_combined)

    def compute_noise_cluster_membership(self, X, NF, ncn_cache):
        NF_list = sorted(NF)
        cluster_counts = {idx: 0 for idx in NF_list}

        for ej in NF_list:
            # Radius of ej's noise cluster = distance to farthest NCN neighbour
            _, ej_distances = ncn_cache[ej]
            if len(ej_distances) == 0:
                continue
            cluster_radius = ej_distances[-1]  # farthest neighbour distance

            # Check which other noisy samples ei fall within ej's cluster
            for ei in NF_list:
                if ei == ej:
                    continue
                dist_ei_to_ej = np.linalg.norm(X[ei] - X[ej])
                if dist_ei_to_ej <= cluster_radius:
                    cluster_counts[ei] += 1  # ei is inside ej's cluster

        return cluster_counts


    def compute_confidence(self,cluster_counts):
        return {idx: 1.0 / (1.0 + c) for idx, c in cluster_counts.items()}


    def compute_distance_rankings(self,distances):
        k = len(distances)
        # Nearest (index 0) gets rank k, farthest (index k-1) gets rank 1
        ranks = np.arange(k, 0, -1, dtype=float)  # [k, k-1, ..., 1]
        return ranks


    def compute_relative_density(self, neighbor_indices, neighbor_distances, target_set, confidence_values):
        k = len(neighbor_indices)
        if k == 0:
            return 0.0

        ranks = self.compute_distance_rankings(neighbor_distances)  # shape (k,)

        numerator = 0.0
        denominator = 0.0

        for _, (nb_idx, rank) in enumerate(zip(neighbor_indices, ranks)):
            # Credibility weight: default to 1.0 for samples not in confidence dict
            conf = confidence_values.get(int(nb_idx), 1.0)
            weighted_rank = conf * rank

            denominator += weighted_rank
            if int(nb_idx) in target_set:
                numerator += weighted_rank

        if denominator == 0.0:
            return 0.0

        return numerator / denominator


    def compute_clean_value(self, ej_idx, NF, ncn_cache, confidence_values):
        nb_indices, nb_distances = ncn_cache[ej_idx]

        # Compute credibility-weighted relative density of NF samples
        rd_noise = self.compute_relative_density(nb_indices, nb_distances, NF, confidence_values)

        if ej_idx not in NF:
            # ej is a clean sample: clean ∈ [0.5, 1]
            clean_val = 1.0 - 0.5 * rd_noise
        else:
            # ej is a noisy sample: clean ∈ [0, 0.5]
            clean_val = 0.5 * (1.0 - rd_noise)

        return float(np.clip(clean_val, 0.0, 1.0))

    def compute_different_class(self, label_ei, label_ej):
        return 1.0 if label_ei != label_ej else -1.0


    def compute_neighborhood_score(self, ei_idx, y_current, NF, ncn_cache, confidence_values, clean_values):
        nb_indices,_ = ncn_cache[ei_idx]
        k = len(nb_indices)

        if k == 0:
            return 0.0

        total = 0.0
        for ej_idx in nb_indices:
            ej_idx = int(ej_idx)
            conf_ej = confidence_values.get(ej_idx, 1.0)
            diff_class = self.compute_different_class(y_current[ei_idx], y_current[ej_idx])
            clean_ej = clean_values.get(ej_idx, 0.5)  # default: ambiguous

            total += conf_ej * diff_class * clean_ej

        return total / k


    def compute_noise_scores(self, NF, y_current, ncn_cache, confidence_values, clean_values):
        rns_scores = {}

        for ei_idx in NF:
            conf_ei = confidence_values.get(ei_idx, 1.0)
            nbhood = self.compute_neighborhood_score(
                ei_idx, y_current, NF, ncn_cache, confidence_values, clean_values
            )
            rns_ei = conf_ei + nbhood
            rns_scores[ei_idx] = float(rns_ei)

        return rns_scores

    def decide_noise_sample_fate(self, ei_idx, rns_score, rns_mean, confidence_ei, y_current, recommended_labels) :
        d = SampleDecision(
            sample_idx=ei_idx,
            rns_score=rns_score,
            rns_mean=rns_mean,
            confidence=confidence_ei,
            current_label=y_current[ei_idx],
            recommended_labels=recommended_labels
        )

        # --- Rule 1: Negative noise score → false positive → RETAIN ---
        if rns_score < 0:
            d.decision = "RETAIN"
            d.rule_applied = 1
            return d

        # Determine majority label from filter recommendations
        if len(recommended_labels) > 0:
            label_counts = Counter(recommended_labels)
            majority_label, majority_count = label_counts.most_common(1)[0]
            unanimous = (majority_count == len(recommended_labels))
            # Majority means more than half of the filters that flagged it agree
            majority_vote = (majority_count > len(recommended_labels) / 2)
        else:
            majority_label = None
            unanimous = False
            majority_vote = False

        # --- Rule 2: Unanimous recommendation from all filters → CORRECT ---
        if unanimous and majority_label is not None and majority_label != y_current[ei_idx]:
            d.decision = "CORRECT"
            d.corrected_label = majority_label
            d.rule_applied = 2
            return d

        # --- Rule 3: rNS ≥ rNS̄ AND majority label → CORRECT ---
        if rns_score >= rns_mean and majority_vote and majority_label is not None:
            d.decision = "CORRECT"
            d.corrected_label = majority_label
            d.rule_applied = 3
            return d

        # --- Rule 4: All other cases → DELETE ---
        d.decision = "DELETE"
        d.rule_applied = 4
        return d


    def process_all_noisy_samples(self, NF, rns_scores, y_current, C_combined, confidence_values):
        if len(rns_scores) == 0:
            return []

        rns_mean = float(np.mean(list(rns_scores.values())))

        decisions = []
        for ei_idx in sorted(NF):
            rns_ei = rns_scores[ei_idx]
            conf_ei = confidence_values.get(ei_idx, 1.0)
            rec_labels = C_combined.get(ei_idx, [])

            dec = self.decide_noise_sample_fate( ei_idx, rns_ei, rns_mean, conf_ei, y_current, rec_labels)
            decisions.append(dec)

        return decisions

    
    def compute_initial_noise_rate(self,NF_initial, n_samples):
        return len(NF_initial) / n_samples


    def adaptive_filter(self, X, y, active_mask, initial_noise_rate):
        adaptive_remove_mask = np.zeros(len(y), dtype=bool)

        if initial_noise_rate <= self.adaptive_filter_threshold:
            if self.verbose:
                print(f"\tAdaptive filtering skipped: NR_j0 = {initial_noise_rate:.1%} "
                      f"≤ {self.adaptive_filter_threshold:.0%}")
            return adaptive_remove_mask

        if self.verbose:
            print(f"\tAdaptive filtering triggered: NR_j0 = {initial_noise_rate:.1%} "
                  f"> {self.adaptive_filter_threshold:.0%}")

        active_indices = np.where(active_mask)[0]
        X_active = X[active_indices]
        y_active = y[active_indices]

        n_active = len(active_indices)
        if n_active < 10:  # too few samples to train reliably
            return adaptive_remove_mask

        # Train each surrogate classifier on active data and predict
        #TODO: need to be customized
        filter_classifiers = [
            KNeighborsClassifier(n_neighbors=1),
            KNeighborsClassifier(n_neighbors=3),
            DecisionTreeClassifier(min_samples_leaf=2, random_state=self.random_state),
            RandomForestClassifier(n_estimators=50, random_state=self.random_state),
            SVC(kernel='rbf', C=100, gamma=0.3, random_state=self.random_state),
        ]

        # Vote count: how many filters misclassify each active sample
        vote_counts = np.zeros(n_active, dtype=int)

        skf = StratifiedKFold(n_splits=min(self.cv_folds, n_active // len(np.unique(y))),
                              shuffle=True, random_state=self.random_state)
        if skf.n_splits < 2:
            return adaptive_remove_mask

        for clf in filter_classifiers:
            try:
                y_pred = cross_val_predict(clf, X_active, y_active, cv=skf)
                vote_counts += (y_pred != y_active).astype(int)
            except Exception:
                continue  # Skip filter if it fails on this data

        # Mark samples where ALL 5 filters misclassify as adaptive noise
        all_misclassified_local = (vote_counts == self.n_filters)
        adaptive_noisy_global = active_indices[all_misclassified_local]
        adaptive_remove_mask[adaptive_noisy_global] = True

        if self.verbose:
            print(f"    Adaptive filtering: {adaptive_remove_mask.sum()} additional samples flagged.")

        return adaptive_remove_mask

    def fit(self, X, y_noisy):
        n_samples = len(y_noisy)

        # Working copies (we simulate — originals untouched)
        active_mask = np.ones(n_samples, dtype=bool)   # True = sample active
        y_working = y_noisy.copy()                     # Labels with simulated corrections

        iteration_records = []
        prev_rns_mean = None
        initial_noise_rate = None
        NF_initial = None

        for iteration in range(self.max_iterations):
            # --- Determine active samples ---
            active_indices = np.where(active_mask)[0]
            n_active = len(active_indices)

            if self.verbose:
                print(f"\n[Iteration {iteration + 1}] Active samples: {n_active}")

            X_active = X[active_indices]
            y_active = y_working[active_indices]

            if n_active < self.k_ncn + 1:
                print(f"  Too few active samples ({n_active}) to continue. Stopping.")
                break

            # ============================================================
            # STEP 1: Noise Identification — run ensemble filters
            # ============================================================
            if self.verbose:
                print("  Step 1: Running ensemble filters...")

            all_noise_sets_local, all_recommended_labels_local = self.run_ensemble_filters( X_active, y_active)

            # Translate local indices → global indices
            all_noise_sets_global = []
            all_recommended_labels_global = []
            for ns_local, rl_local in zip(all_noise_sets_local, all_recommended_labels_local):
                ns_global = {int(active_indices[i]) for i in ns_local}
                rl_global = {int(active_indices[i]): v for i, v in rl_local.items()}
                all_noise_sets_global.append(ns_global)
                all_recommended_labels_global.append(rl_global)

            # ============================================================
            # STEP 2: Final Noise Set Construction — majority voting
            # ============================================================
            NF, C_combined = self.construct_final_noise_set( all_noise_sets_global, all_recommended_labels_global, n_samples)

            if self.verbose:
                print(f"  Step 2: Final noise set |NF| = {len(NF)}")

            # Record initial noise rate (first iteration only)
            if iteration == 0:
                NF_initial = NF.copy()
                initial_noise_rate = self.compute_initial_noise_rate(NF_initial, n_samples)
                if self.verbose:
                    print(f"\t\tInitial noise rate NR_j0 = {initial_noise_rate:.2%}")

            if len(NF) == 0:
                if self.verbose:
                    print("  No noisy samples detected. Terminating.")
                break

            # ============================================================
            # STEP 3: NCN Cache + Noise Score Calculation
            # ============================================================
            if self.verbose:
                print("  Step 3: Building NCN cache and computing noise scores...")

            # Build NCN cache on active features only (then map back to global)
            ncn_cache_local = self.ncn.build_neighborhood_cache(X_active, self.k_ncn)
            # Map local cache to global indices
            ncn_cache_global = {
                int(active_indices[local_i]): (
                    np.array([int(active_indices[nb]) for nb in nb_idx]),
                    nb_dist
                )
                for local_i, (nb_idx, nb_dist) in ncn_cache_local.items()
            }

            # Compute noise cluster membership and confidence
            cluster_counts = self.compute_noise_cluster_membership(X, NF, ncn_cache_global)
            confidence_values = self.compute_confidence(cluster_counts)

            # Add default confidence for active non-NF samples (used in neighbourhood)
            for idx in active_indices:
                if int(idx) not in confidence_values:
                    confidence_values[int(idx)] = 1.0  # not in any noise cluster

            # Compute clean values for all active samples
            clean_values = {}
            for idx in active_indices:
                clean_values[int(idx)] = self.compute_clean_value( int(idx), NF, ncn_cache_global, confidence_values)

            # Compute noise scores for NF samples
            rns_scores = self.compute_noise_scores( NF, y_working, ncn_cache_global, confidence_values, clean_values)

            rns_mean = float(np.mean(list(rns_scores.values()))) if rns_scores else 0.0

            if self.verbose:
                print(f"\t\trNS̄ (mean noise score) = {rns_mean:.4f}")
                score_vals = list(rns_scores.values())
                print(f"\t\trNS range: [{min(score_vals):.4f}, {max(score_vals):.4f}]")

            # ============================================================
            # TERMINATION CHECK: rNS̄ increment ≤ 0
            # ============================================================
            if prev_rns_mean is not None:
                increment = rns_mean - prev_rns_mean
                if self.verbose:
                    print(f"         rNS̄ increment = {increment:.4f}")
                if increment <= self.termination_delta:
                    if self.verbose:
                        print(f"  Termination criterion met (increment={increment:.4f} ≤ "
                            f"{self.termination_delta}). Stopping iteration.")
                    break

            prev_rns_mean = rns_mean

            # ============================================================
            # STEP 4: Noise Sample Processing
            # ============================================================
            if self.verbose:
                print("  Step 4: Applying decision rules...")

            decisions = self.process_all_noisy_samples(
                NF, rns_scores, y_working,
                # y_clean_reference,
                C_combined, confidence_values
            )

            # Apply decisions to working state
            n_retained = n_corrected = n_deleted = 0
            for dec in decisions:
                if dec.decision == "RETAIN":
                    n_retained += 1
                    # Keep original label — no change
                elif dec.decision == "CORRECT":
                    n_corrected += 1
                    # Simulate label correction
                    y_working[dec.sample_idx] = dec.corrected_label
                elif dec.decision == "DELETE":
                    n_deleted += 1
                    # Simulate deletion (deactivate)
                    active_mask[dec.sample_idx] = False

            if self.verbose:
                print(f"\t\tRetain: {n_retained}, Correct: {n_corrected}, "
                    f"Delete: {n_deleted}")

            record = IterationRecord(
                iteration=iteration,
                n_active=n_active,
                NF_size=len(NF),
                rns_mean=rns_mean,
                n_retained=n_retained,
                n_corrected=n_corrected,
                n_deleted=n_deleted,
                decisions=decisions
            )
            iteration_records.append(record)

            if n_corrected + n_deleted == 0:
                if self.verbose:
                    print("  No changes made this iteration. Terminating.")
                break

        # ============================================================
        # STEP 5: Adaptive Filtering
        # ============================================================
        if self.verbose:
            print(f"\n  Step 5: Adaptive Filtering check...")

        adaptive_remove_mask = self.adaptive_filter( X, y_working, active_mask, initial_noise_rate or 0.0)

        return active_mask, y_working, iteration_records, adaptive_remove_mask
