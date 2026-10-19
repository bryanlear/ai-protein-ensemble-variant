import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score

RMSD_MATRIX_PATH = Path(
    "/outputs/my_protein/"
    "full_length_ca_pairwise_rmsd_angstrom.npy")
OUTPUT_DIRECTORY = Path(
    "/outputs/my_protein/k_medoids")


def initialize_medoids(distance_matrix, number_of_clusters):
    number_of_conformations = distance_matrix.shape[0]
    medoids = [int(np.argmin(distance_matrix.sum(axis=1)))]
    distance_to_nearest_medoid = distance_matrix[:, medoids[0]].copy()

    while len(medoids) < number_of_clusters:
        candidates = np.setdiff1d(
            np.arange(number_of_conformations),
            np.asarray(medoids),
            assume_unique=True,)
        candidate_costs = np.minimum(
            distance_to_nearest_medoid[:, None],
            distance_matrix[:, candidates],).sum(axis=0)
        new_medoid = int(candidates[np.argmin(candidate_costs)])
        medoids.append(new_medoid)
        distance_to_nearest_medoid = np.minimum(
            distance_to_nearest_medoid,
            distance_matrix[:, new_medoid],)

    return np.asarray(medoids, dtype=int)

def assign_to_medoids(distance_matrix, medoids):
    distances_to_medoids = distance_matrix[:, medoids]
    labels = np.argmin(distances_to_medoids, axis=1)
    labels[medoids] = np.arange(len(medoids))
    distances_to_assigned_medoids = distances_to_medoids[
        np.arange(distance_matrix.shape[0]),
        labels,]
    return labels, distances_to_assigned_medoids

def fit_k_medoids(distance_matrix, number_of_clusters, maximum_iterations):
    number_of_conformations = distance_matrix.shape[0]
    medoids = initialize_medoids(distance_matrix, number_of_clusters)

    for iteration in range(1, maximum_iterations + 1):
        labels, nearest_distances = assign_to_medoids(distance_matrix, medoids)
        distances_to_medoids = distance_matrix[:, medoids]

        if number_of_clusters == 1:
            second_nearest_distances = np.full(number_of_conformations, np.inf)
        else:
            second_nearest_distances = np.partition(distances_to_medoids, 1, axis=1)[:, 1]

        nonmedoids = np.setdiff1d(
            np.arange(number_of_conformations),
            medoids,
            assume_unique=True,)
        current_cost = float(nearest_distances.sum())
        best_cost = current_cost
        best_medoid_position = None
        best_replacement = None

        for medoid_position in range(number_of_clusters):
            distance_without_medoid = np.where(
                labels == medoid_position,
                second_nearest_distances,
                nearest_distances,)
            swap_costs = np.minimum(
                distance_without_medoid[:, None],
                distance_matrix[:, nonmedoids],).sum(axis=0)
            candidate_position = int(np.argmin(swap_costs))
            candidate_cost = float(swap_costs[candidate_position])

            if candidate_cost < best_cost - 1e-12:
                best_cost = candidate_cost
                best_medoid_position = medoid_position
                best_replacement = int(nonmedoids[candidate_position])

        if best_medoid_position is None:
            break

        medoids[best_medoid_position] = best_replacement

    labels, distances_to_assigned_medoids = assign_to_medoids(distance_matrix, medoids)
    return medoids, labels, distances_to_assigned_medoids, iteration

def save_silhouette_plot(silhouette_values, labels, average_silhouette, number_of_clusters):
    fig, ax = plt.subplots(figsize=(9, 7))
    y_lower = 10
    colors = plt.get_cmap("tab20")

    for cluster_label in range(number_of_clusters):
        cluster_values = np.sort(silhouette_values[labels == cluster_label])
        cluster_size = len(cluster_values)
        y_upper = y_lower + cluster_size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            color=colors(cluster_label % 20),
            alpha=0.8,)
        ax.text(-0.95, y_lower + 0.5 * cluster_size, str(cluster_label))
        y_lower = y_upper + 10

    ax.axvline(average_silhouette, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {average_silhouette:.4f}")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Conformations grouped by cluster")
    ax.set_yticks([])
    ax.set_title(f"K-medoids silhouette distribution: k={number_of_clusters}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / f"silhouette_distribution_k_{number_of_clusters}.png", dpi=300)
    plt.close(fig)


parser = argparse.ArgumentParser(description="Run PAM k-medoids on a precomputed C-alpha RMSD matrix.")
parser.add_argument(
    "k_values",
    nargs="+",
    type=int,
    help="Cluster counts to test, for example: 2 3 4 5 6",)
parser.add_argument(
    "--maximum-iterations",
    type=int,
    default=100,
    help="Maximum PAM swap iterations for each k. Default: 100.",)
arguments = parser.parse_args()

k_values = tuple(dict.fromkeys(arguments.k_values))
if len(k_values) < 2:
    raise ValueError("Provide at least two different k values for silhouette comparison")
if arguments.maximum_iterations < 1:
    raise ValueError("--maximum-iterations must be at least 1")

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

number_of_conformations = rmsd_matrix.shape[0]
if any(k < 2 or k >= number_of_conformations for k in k_values):
    raise ValueError(f"Each k must be from 2 to {number_of_conformations - 1}")

run_summary_records = []

for number_of_clusters in k_values:
    medoids, labels, distances_to_medoids, iterations = fit_k_medoids(
        rmsd_matrix,
        number_of_clusters,
        arguments.maximum_iterations,)

    silhouette_values = silhouette_samples(rmsd_matrix, labels, metric="precomputed")
    average_silhouette = float(silhouette_score(rmsd_matrix, labels, metric="precomputed"))
    cluster_labels, cluster_sizes = np.unique(labels, return_counts=True)
    dominant_position = int(np.argmax(cluster_sizes))
    dominant_label = int(cluster_labels[dominant_position])
    dominant_size = int(cluster_sizes[dominant_position])

    np.save(OUTPUT_DIRECTORY / f"labels_k_{number_of_clusters}.npy", labels)
    np.save(OUTPUT_DIRECTORY / f"medoids_k_{number_of_clusters}.npy", medoids)

    pd.DataFrame(
        {
            "conformation_index": np.arange(number_of_conformations),
            "cluster_label": labels,
            "cluster_medoid_index": medoids[labels],
            "distance_to_medoid_angstrom": distances_to_medoids,
            "silhouette_coefficient": silhouette_values,
        }
    ).to_csv(OUTPUT_DIRECTORY / f"assignments_k_{number_of_clusters}.csv", index=False)

    save_silhouette_plot(silhouette_values, labels, average_silhouette, number_of_clusters,)

    run_summary_records.append(
        {
            "number_of_clusters": number_of_clusters,
            "silhouette_score": average_silhouette,
            "total_distance_to_medoids_angstrom": float(distances_to_medoids.sum()),
            "mean_distance_to_medoid_angstrom": float(distances_to_medoids.mean()),
            "dominant_cluster_label": dominant_label,
            "dominant_cluster_size": dominant_size,
            "dominant_cluster_population": dominant_size / number_of_conformations,
            "dominant_cluster_medoid_index": int(medoids[dominant_label]),
            "pam_iterations": iterations,})

run_summary = pd.DataFrame(run_summary_records).sort_values("number_of_clusters")
run_summary.to_csv(OUTPUT_DIRECTORY / "silhouette_analysis_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(run_summary["number_of_clusters"], run_summary["silhouette_score"], marker="o", linewidth=1.8,)
ax.set_xlabel("Number of clusters, k")
ax.set_ylabel("Mean silhouette score")
ax.set_xticks(run_summary["number_of_clusters"])
ax.grid(alpha=0.25)
ax.set_title("K-medoids silhouette analysis")
fig.tight_layout()
fig.savefig(OUTPUT_DIRECTORY / "silhouette_score_by_k.png", dpi=300)
plt.close(fig)

print("\nK-medoids silhouette analysis")
print(run_summary.to_string(index=False))
print(f"\nOutputs: {OUTPUT_DIRECTORY}")
