# pointcloud_resampler.py

import numpy as np
import time

# ── open3d opzionale ────────────────────────────────────────────────────────
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

# ── scipy opzionale (KDTree veloce) ────────────────────────────────────────
try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================  
# KDTree helper
# ============================================================================  

class _NumpyKDTree:
    def __init__(self, pts):
        self._pts = pts

    def query(self, qpts, k=1):
        dists, idxs = [], []
        for q in np.atleast_2d(qpts):
            d2 = np.sum((self._pts - q) ** 2, axis=1)
            idx = np.argsort(d2)[:k]
            dists.append(np.sqrt(d2[idx]))
            idxs.append(idx)
        return np.array(dists), np.array(idxs)


def build_kdtree(pts):
    return KDTree(pts) if HAS_SCIPY else _NumpyKDTree(pts)


# ============================================================================  
# CLASSE PRINCIPALE
# ============================================================================  

class PointCloudResampler:

    # ----------------------------------------------------------------------
    # DOWNSAMPLING
    # ----------------------------------------------------------------------

    @staticmethod
    def voxel_downsample(points, voxel_size):
        if HAS_O3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            return np.asarray(pcd.voxel_down_sample(voxel_size).points)
        else:
            voxel_idx = np.floor(points / voxel_size).astype(np.int64)
            voxels = {}
            for i, key in enumerate(map(tuple, voxel_idx)):
                voxels.setdefault(key, []).append(points[i])
            return np.array([np.mean(pts, axis=0) for pts in voxels.values()])

    @staticmethod
    def poisson_disk(points, min_dist, seed=42):
        rng = np.random.default_rng(seed)
        pts = points[rng.permutation(len(points))]
        cell_size = min_dist
        grid = {}

        def _neighbors(ci, cj, ck):
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    for dk in range(-2, 3):
                        yield (ci + di, cj + dj, ck + dk)

        accepted = []
        for p in pts:
            ci, cj, ck = (int(np.floor(p[ax] / cell_size)) for ax in range(3))
            too_close = False
            for nb in _neighbors(ci, cj, ck):
                for q in grid.get(nb, []):
                    if np.linalg.norm(p - q) < min_dist:
                        too_close = True
                        break
                if too_close:
                    break
            if not too_close:
                grid.setdefault((ci, cj, ck), []).append(p)
                accepted.append(p)

        return np.array(accepted)

    @staticmethod
    def fps(points, n_samples, seed=42):
        N = len(points)
        if n_samples >= N:
            return points.copy()
        selected = np.zeros(n_samples, dtype=np.int64)
        distances = np.full(N, np.inf)
        rng = np.random.default_rng(seed)
        selected[0] = rng.integers(0, N)
        for i in range(1, n_samples):
            last = points[selected[i - 1]]
            distances = np.minimum(distances, np.sum((points - last) ** 2, axis=1))
            selected[i] = np.argmax(distances)
        return points[selected]

    # ----------------------------------------------------------------------
    # UPSAMPLING
    # ----------------------------------------------------------------------

    @staticmethod
    def upsample_jitter(points, n_new, voxel_size=None, seed=42):
        rng = np.random.default_rng(seed)

        # stima voxel_size
        if voxel_size is None:
            tree = build_kdtree(points)
            dists, _ = tree.query(points[:min(2000, len(points))], k=2)
            voxel_size = float(np.median(dists[:, 1])) * 1.5
            voxel_size = max(voxel_size, 1e-9)

        voxel_idx = np.floor(points / voxel_size).astype(np.int64)
        occupied = set(map(tuple, voxel_idx))

        offsets = [(i, j, k) for i in [-1, 0, 1]
                            for j in [-1, 0, 1]
                            for k in [-1, 0, 1]
                            if (i, j, k) != (0, 0, 0)]

        candidates = set()
        for occ in occupied:
            for off in offsets:
                nb = (occ[0]+off[0], occ[1]+off[1], occ[2]+off[2])
                if nb not in occupied:
                    candidates.add(nb)

        if not candidates:
            idxs = rng.integers(0, len(points), size=n_new)
            noise = rng.normal(0, voxel_size * 0.3, size=(n_new, 3))
            return points[idxs] + noise

        candidates = list(candidates)
        tree = build_kdtree(points)
        new_pts = []

        while len(new_pts) < n_new:
            batch = min(n_new - len(new_pts), len(candidates))
            chosen = [candidates[i] for i in rng.integers(0, len(candidates), batch)]
            centers = np.array(chosen, dtype=float) * voxel_size + voxel_size * 0.5

            k = min(4, len(points))
            dists, idxs = tree.query(centers, k=k)
            dists = np.atleast_2d(dists)
            idxs = np.atleast_2d(idxs)

            for j in range(len(centers)):
                d = dists[j]
                idx = idxs[j]
                w = 1.0 / (d + 1e-9)
                w /= w.sum()
                interp = (points[idx] * w[:, None]).sum(axis=0)
                jitter = rng.normal(0, voxel_size * 0.25, size=3)
                new_pts.append(interp + jitter)

        return np.array(new_pts[:n_new])

    @staticmethod
    def upsample_knn(points, n_new, k=6, density_percentile=30.0, seed=42):
        rng = np.random.default_rng(seed)
        N = len(points)
        k = min(k, N - 1)

        tree = build_kdtree(points)
        dists, idxs = tree.query(points, k=k+1)
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]
        local_density = dists.mean(axis=1)

        threshold = np.percentile(local_density, 100 - density_percentile)
        sparse_mask = local_density >= threshold
        sparse_pts = np.where(sparse_mask)[0]
        if len(sparse_pts) == 0:
            sparse_pts = np.arange(N)

        new_pts = []
        while len(new_pts) < n_new:
            batch = min(n_new - len(new_pts), len(sparse_pts))
            chosen = rng.choice(sparse_pts, size=batch, replace=True)

            for ci in chosen:
                p = points[ci]
                nb_idx = idxs[ci, rng.integers(0, k)]
                q = points[nb_idx]

                t = rng.uniform(0.2, 0.8)
                mid = p + t * (q - p)

                seg = q - p
                seg_len = np.linalg.norm(seg)
                if seg_len > 1e-9:
                    rand_v = rng.normal(size=3)
                    rand_v -= rand_v.dot(seg / seg_len) * (seg / seg_len)
                    rand_v /= (np.linalg.norm(rand_v) + 1e-9)
                    jitter = rand_v * rng.normal(0, seg_len * 0.15)
                else:
                    jitter = rng.normal(0, local_density[ci] * 0.2, size=3)

                new_pts.append(mid + jitter)

        return np.array(new_pts[:n_new])

    # ----------------------------------------------------------------------
    # MERGE
    # ----------------------------------------------------------------------

    @staticmethod
    def upsample_merge(points, n_new, method="jitter",
                       voxel_size=None, k=6, density_percentile=30.0,
                       final_voxel_clean=True, seed=42):

        if n_new <= 0:
            return points.copy()

        if method == "jitter":
            new_pts = PointCloudResampler.upsample_jitter(points, n_new,
                                                          voxel_size=voxel_size,
                                                          seed=seed)
        elif method == "knn":
            new_pts = PointCloudResampler.upsample_knn(points, n_new,
                                                       k=k,
                                                       density_percentile=density_percentile,
                                                       seed=seed)
        else:
            raise ValueError("Metodo upsampling sconosciuto")

        merged = np.vstack([points, new_pts])

        if final_voxel_clean and voxel_size is not None:
            merged = PointCloudResampler.voxel_downsample(merged, voxel_size * 0.8)

        return merged
