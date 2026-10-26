from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

RMSD_MATRIX_PATH = Path("outputs/my_protein/full_length_ca_pairwise_rmsd_angstrom.npy")
OUTPUT_DIRECTORY = Path("outputs/my_protein/hac_average")

# ONLY ONE TO CUT DENDOGRAM. NONE to INSPECT
N_CLUSTERS = None
DISTANCE_THRESHOLD_ANGSTROM = None

if N_CLUSTERS is not None and DISTANCE_THRESHOLD_ANGSTROM is not None:
    raise ValueError("Set N_CLUSTERS or DISTANCE_THRESHOLD_ANGSTROM, not both")

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

rmsd_matrix = np.asarray(np.load(RMSD_MATRIX_PATH), dtype=np.float64)

if rmsd_matrix.ndim != 2 or rmsd_matrix.shape[0] != rmsd_matrix.shape[1]:
    raise ValueError(f"RMSD matrix must be square. Received shape: {rmsd_matrix.shape}")
if not np.isfinite(rmsd_matrix).all():
    raise ValueError("RMSD matrix contains NaN or infinite values")
if np.any(rmsd_matrix < 0.0):
    raise ValueError("RMSD matrix contains negative distances")
if not np.allclose(rmsd_matrix, rmsd_matrix.T, atol=1e-6):
    raise ValueError("RMSD matrix is not symmetric")
if not np.allclose(np.diag(rmsd_matrix), 0.0, atol=1e-6):
    raise ValueError("RMSD matrix diagonal is not zero")

n_conformations = rmsd_matrix.shape[0]
condensed_rmsd = squareform(rmsd_matrix, checks=False)
linkage_matrix = linkage(condensed_rmsd, method="average")

np.save(OUTPUT_DIRECTORY / "average_linkage_matrix.npy", linkage_matrix)

merge_heights = linkage_matrix[:, 2]
clusters_after_merge = n_conformations - np.arange(1, n_conformations)
gap_to_next_merge = np.append(np.diff(merge_heights), np.nan)

merge_table = pd.DataFrame(
    {"merge_index": np.arange(n_conformations - 1),
        "number_of_clusters_after_merge": clusters_after_merge,
        "merge_height_angstrom": merge_heights,
        "gap_to_next_merge_angstrom": gap_to_next_merge,
        "merged_cluster_size": linkage_matrix[:, 3].astype(int),})

merge_table.to_csv(OUTPUT_DIRECTORY / "merge_table.csv", index=False)

fig, ax = plt.subplots(figsize=(18, 8))
dendrogram(linkage_matrix, no_labels=True, color_threshold=None, ax=ax)
ax.set_xlabel("Conformations")
ax.set_ylabel("Average-linkage distance (Å)")
ax.set_title("Average-linkage HAC dendrogram: full-length Cα RMSD")
fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY / "average_linkage_dendrogram.png", dpi=300)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
axes[0].plot(clusters_after_merge, merge_heights, linewidth=1.5)
axes[0].set_ylabel("Merge height (Å)")
axes[0].set_title("Average-linkage merge heights")
axes[0].grid(alpha=0.25)
axes[1].plot(clusters_after_merge[:-1], gap_to_next_merge[:-1], linewidth=1.5)
axes[1].set_xlabel("Number of clusters after merge")
axes[1].set_ylabel("Gap to next merge (Å)")
axes[1].set_title("Increase required for the next merge")
axes[1].grid(alpha=0.25)
axes[1].invert_xaxis()

fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY / "average_linkage_merge_heights.png", dpi=300)
plt.close(fig)

distance_thresholds = np.unique(merge_heights)
dominant_membership = np.zeros((len(distance_thresholds), n_conformations), dtype=bool)
threshold_summary_records = []
dominant_metric_cache = {}
previous_dominant_members = None

for threshold_index, distance_threshold in enumerate(distance_thresholds):
    threshold_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance") - 1
    threshold_cluster_labels, threshold_cluster_sizes = np.unique(threshold_labels, return_counts=True)
    dominant_position = int(np.argmax(threshold_cluster_sizes))
    dominant_label = int(threshold_cluster_labels[dominant_position])
    dominant_members = np.flatnonzero(threshold_labels == dominant_label)
    dominant_size = len(dominant_members)
    dominant_membership[threshold_index, dominant_members] = True

    member_key = tuple(dominant_members)
    if member_key not in dominant_metric_cache:
        dominant_rmsd_matrix = rmsd_matrix[np.ix_(dominant_members, dominant_members)]
        medoid_index = int(dominant_members[np.argmin(dominant_rmsd_matrix.sum(axis=1))])
        dominant_pairwise_rmsd = dominant_rmsd_matrix[np.triu_indices(dominant_size, k=1)]

        if dominant_pairwise_rmsd.size:
            percentile_25_rmsd, median_rmsd, percentile_75_rmsd, percentile_90_rmsd = np.percentile(
                dominant_pairwise_rmsd,
                [25, 50, 75, 90],
            )
            dominant_metrics = {
                "number_of_within_cluster_rmsd_pairs": int(dominant_pairwise_rmsd.size),
                "mean_within_cluster_rmsd_angstrom": float(np.mean(dominant_pairwise_rmsd)),
                "percentile_25_within_cluster_rmsd_angstrom": float(percentile_25_rmsd),
                "median_within_cluster_rmsd_angstrom": float(median_rmsd),
                "percentile_75_within_cluster_rmsd_angstrom": float(percentile_75_rmsd),
                "percentile_90_within_cluster_rmsd_angstrom": float(percentile_90_rmsd),
                "maximum_within_cluster_rmsd_angstrom": float(np.max(dominant_pairwise_rmsd)),
                "dominant_cluster_medoid_index": medoid_index,
            }
        else:
            dominant_metrics = {
                "number_of_within_cluster_rmsd_pairs": 0,
                "mean_within_cluster_rmsd_angstrom": np.nan,
                "percentile_25_within_cluster_rmsd_angstrom": np.nan,
                "median_within_cluster_rmsd_angstrom": np.nan,
                "percentile_75_within_cluster_rmsd_angstrom": np.nan,
                "percentile_90_within_cluster_rmsd_angstrom": np.nan,
                "maximum_within_cluster_rmsd_angstrom": np.nan,
                "dominant_cluster_medoid_index": medoid_index,}

        dominant_metric_cache[member_key] = dominant_metrics

    if previous_dominant_members is None:
        intersection_size = np.nan
        union_size = np.nan
        jaccard_overlap_with_previous_threshold = np.nan
        fraction_of_previous_dominant_retained = np.nan
        fraction_of_current_dominant_from_previous = np.nan
    else:
        intersection_size = int(np.intersect1d(previous_dominant_members, dominant_members, assume_unique=True).size)
        union_size = int(np.union1d(previous_dominant_members, dominant_members).size)
        jaccard_overlap_with_previous_threshold = intersection_size / union_size
        fraction_of_previous_dominant_retained = intersection_size / len(previous_dominant_members)
        fraction_of_current_dominant_from_previous = intersection_size / dominant_size

    threshold_summary_records.append(
        {
            "threshold_index": threshold_index,
            "distance_threshold_angstrom": float(distance_threshold),
            "number_of_clusters": len(threshold_cluster_labels),
            "dominant_cluster_label": dominant_label,
            "dominant_cluster_size": dominant_size,
            "dominant_cluster_population": dominant_size / n_conformations,
            **dominant_metric_cache[member_key],
            "intersection_with_previous_dominant": intersection_size,
            "union_with_previous_dominant": union_size,
            "jaccard_overlap_with_previous_threshold": jaccard_overlap_with_previous_threshold,
            "fraction_of_previous_dominant_retained": fraction_of_previous_dominant_retained,
            "fraction_of_current_dominant_from_previous": fraction_of_current_dominant_from_previous,})

    previous_dominant_members = dominant_members

threshold_summary = pd.DataFrame(threshold_summary_records)
threshold_summary.to_csv(OUTPUT_DIRECTORY / "distance_threshold_evaluation.csv", index=False)

membership_table = pd.DataFrame(
    dominant_membership.astype(np.uint8),
    columns=[f"conformation_{index}" for index in range(n_conformations)],)
membership_table.insert(0, "distance_threshold_angstrom", distance_thresholds)
membership_table.insert(0, "threshold_index", np.arange(len(distance_thresholds)))
membership_table.to_csv(OUTPUT_DIRECTORY / "dominant_membership_by_threshold.csv", index=False)

np.savez_compressed(
    OUTPUT_DIRECTORY / "distance_threshold_evaluation.npz",
    distance_threshold_angstrom=distance_thresholds,
    dominant_membership=dominant_membership,
    dominant_cluster_medoid_index=threshold_summary["dominant_cluster_medoid_index"].to_numpy(dtype=int),
    dominant_cluster_population=threshold_summary["dominant_cluster_population"].to_numpy(),)

fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
axes[0].plot(threshold_summary["distance_threshold_angstrom"], threshold_summary["number_of_clusters"])
axes[0].set_ylabel("Number of clusters")
axes[0].set_title("Average-linkage HAC distance-threshold evaluation")
axes[0].grid(alpha=0.25)

axes[1].plot(threshold_summary["distance_threshold_angstrom"], threshold_summary["dominant_cluster_population"])
axes[1].set_ylabel("Dominant population")
axes[1].set_ylim(0.0, 1.02)
axes[1].grid(alpha=0.25)

axes[2].fill_between(
    threshold_summary["distance_threshold_angstrom"],
    threshold_summary["percentile_25_within_cluster_rmsd_angstrom"],
    threshold_summary["percentile_75_within_cluster_rmsd_angstrom"],
    alpha=0.25,
    label="25th–75th percentile",)
axes[2].plot(
    threshold_summary["distance_threshold_angstrom"],
    threshold_summary["median_within_cluster_rmsd_angstrom"],
    label="Median",)
axes[2].plot(
    threshold_summary["distance_threshold_angstrom"],
    threshold_summary["percentile_90_within_cluster_rmsd_angstrom"],
    label="90th percentile",)
axes[2].plot(
    threshold_summary["distance_threshold_angstrom"],
    threshold_summary["maximum_within_cluster_rmsd_angstrom"],
    linewidth=1.0,
    label="Maximum",)
axes[2].set_ylabel("Dominant-cluster RMSD (Å)")
axes[2].legend()
axes[2].grid(alpha=0.25)

axes[3].plot(
    threshold_summary["distance_threshold_angstrom"],
    threshold_summary["jaccard_overlap_with_previous_threshold"],)
axes[3].set_xlabel("Distance threshold (Å)")
axes[3].set_ylabel("Dominant-membership\nJaccard overlap")
axes[3].set_ylim(0.0, 1.02)
axes[3].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY / "distance_threshold_evaluation.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(16, 10))
image = ax.imshow(dominant_membership, aspect="auto", interpolation="nearest", cmap="binary")
ax.set_xlabel("Conformation index")
ax.set_ylabel("Threshold index: increasing distance")
ax.set_title("Dominant-cluster membership across distance thresholds")
fig.colorbar(image, ax=ax, label="Dominant-cluster member")
fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY / "dominant_membership_by_threshold.png", dpi=300)
plt.close(fig)

print("\nAverage-linkage HAC")
print(f"Number of conformations: {n_conformations}")
print(f"Final merge height: {merge_heights[-1]:.6f} Å")
print(f"Distance thresholds evaluated: {len(distance_thresholds)}")
print(f"Linkage matrix: {OUTPUT_DIRECTORY / 'average_linkage_matrix.npy'}")
print(f"Merge table: {OUTPUT_DIRECTORY / 'merge_table.csv'}")
print(f"Dendrogram: {OUTPUT_DIRECTORY / 'average_linkage_dendrogram.png'}")
print(f"Merge-height plot: {OUTPUT_DIRECTORY / 'average_linkage_merge_heights.png'}")
print(f"Threshold summary: {OUTPUT_DIRECTORY / 'distance_threshold_evaluation.csv'}")
print(f"Dominant membership: {OUTPUT_DIRECTORY / 'dominant_membership_by_threshold.csv'}")
print(f"Threshold plot: {OUTPUT_DIRECTORY / 'distance_threshold_evaluation.png'}")
print(f"Membership plot: {OUTPUT_DIRECTORY / 'dominant_membership_by_threshold.png'}")

if N_CLUSTERS is None and DISTANCE_THRESHOLD_ANGSTROM is None:
    print("No final dendrogram cut was applied")
    raise SystemExit(0)

if N_CLUSTERS is not None:
    if not isinstance(N_CLUSTERS, int) or not 2 <= N_CLUSTERS < n_conformations:
        raise ValueError(f"N_CLUSTERS must be an integer from 2 to {n_conformations - 1}")
    labels = cut_tree(linkage_matrix, n_clusters=N_CLUSTERS).reshape(-1)
    cut_description = f"number_of_clusters_{N_CLUSTERS}"
else:
    if DISTANCE_THRESHOLD_ANGSTROM < 0.0:
        raise ValueError("DISTANCE_THRESHOLD_ANGSTROM must be nonnegative")
    labels = fcluster(linkage_matrix, t=DISTANCE_THRESHOLD_ANGSTROM, criterion="distance") - 1
    cut_description = f"distance_threshold_{DISTANCE_THRESHOLD_ANGSTROM:g}_angstrom"

cluster_labels, cluster_sizes = np.unique(labels, return_counts=True)
number_of_clusters = len(cluster_labels)

np.save(OUTPUT_DIRECTORY / f"labels_{cut_description}.npy", labels)
pd.DataFrame(
    {
        "conformation_index": np.arange(n_conformations),
        "cluster_label": labels,
    }
).to_csv(OUTPUT_DIRECTORY / f"assignments_{cut_description}.csv", index=False)

cluster_summary_records = []

for cluster_label, cluster_size in zip(cluster_labels, cluster_sizes):
    members = np.flatnonzero(labels == cluster_label)
    within_cluster_matrix = rmsd_matrix[np.ix_(members, members)]
    medoid_index = int(members[np.argmin(within_cluster_matrix.sum(axis=1))])
    within_cluster_rmsd = within_cluster_matrix[np.triu_indices(cluster_size, k=1)]

    if within_cluster_rmsd.size:
        mean_rmsd = float(np.mean(within_cluster_rmsd))
        median_rmsd = float(np.median(within_cluster_rmsd))
        percentile_90_rmsd = float(np.percentile(within_cluster_rmsd, 90))
        maximum_rmsd = float(np.max(within_cluster_rmsd))
    else:
        mean_rmsd = np.nan
        median_rmsd = np.nan
        percentile_90_rmsd = np.nan
        maximum_rmsd = np.nan

    cluster_summary_records.append(
        {
            "cluster_label": int(cluster_label),
            "cluster_size": int(cluster_size),
            "population_of_total_ensemble": int(cluster_size) / n_conformations,
            "medoid_conformation_index": medoid_index,
            "mean_within_cluster_rmsd_angstrom": mean_rmsd,
            "median_within_cluster_rmsd_angstrom": median_rmsd,
            "percentile_90_within_cluster_rmsd_angstrom": percentile_90_rmsd,
            "maximum_within_cluster_rmsd_angstrom": maximum_rmsd,})

cluster_summary = pd.DataFrame(cluster_summary_records).sort_values("cluster_size", ascending=False)
cluster_summary.to_csv(OUTPUT_DIRECTORY / f"cluster_summary_{cut_description}.csv", index=False)

if 1 < number_of_clusters < n_conformations:
    silhouette = silhouette_score(rmsd_matrix, labels, metric="precomputed")
else:
    silhouette = np.nan

fig, ax = plt.subplots(figsize=(12, 6))
sorted_sizes = np.sort(cluster_sizes)[::-1]
ax.bar(np.arange(1, number_of_clusters + 1), sorted_sizes / n_conformations)
ax.set_xlabel("Cluster rank by population")
ax.set_ylabel("Population fraction")
ax.set_title(f"Average-linkage HAC cluster populations: {cut_description}")
fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY / f"cluster_populations_{cut_description}.png", dpi=300)
plt.close(fig)

print(f"\nApplied cut: {cut_description}")
print(f"Number of clusters: {number_of_clusters}")
print(f"Silhouette score: {silhouette:.6f}")
print("\nCluster summary")
print(cluster_summary.to_string(index=False))
