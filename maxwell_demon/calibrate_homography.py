import numpy as np
import cv2

# -------------------------
# 1. Homography estimation
# -------------------------

def compute_homography(image_pts, world_pts):
    """
    image_pts: Nx2 array of pixel coordinates (u,v)
    world_pts: Nx2 array of real coordinates (X,Y)
    returns: homography H (3x3 matrix)
    """
    H, mask = cv2.findHomography(image_pts, world_pts, cv2.RANSAC)
    return H


# -------------------------
# 2. Map image point -> world
# -------------------------

def map_point(H, pt):
    """
    H: homography mapping image->world
    pt: (u, v)
    returns: (X, Y)
    """
    # Homogeneous multiplication
    p = np.array([pt[0], pt[1], 1.0], dtype=float)
    Pw = H @ p
    return Pw[0]/Pw[2], Pw[1]/Pw[2]


def map_points(H, pts):
    return np.array([map_point(H, p) for p in pts])


# -------------------------
# 3. Distance in world coordinates
# -------------------------

def world_distance(H, p1, p2):
    P1 = np.array(map_point(H, p1))
    P2 = np.array(map_point(H, p2))
    return np.linalg.norm(P1 - P2)


# -------------------------
# 4. Monte-Carlo uncertainty estimation
# -------------------------

def measure_distance_monte_carlo(
        image_pts, world_pts,
        p1, p2,
        sigma_px=1.0,
        sigma_gcp=0.01,
        N=2000):

    image_pts = np.array(image_pts)
    world_pts = np.array(world_pts)

    distances = []

    for _ in range(N):

        # 1. Perturb image control points
        img_pert = image_pts + np.random.normal(0, sigma_px, image_pts.shape)

        # 2. Perturb ground control points
        world_pert = world_pts + np.random.normal(0, sigma_gcp, world_pts.shape)

        # 3. Compute homography
        H = compute_homography(img_pert, world_pert)

        # 4. Perturb measurement points (pixel localization noise)
        p1p = p1 + np.random.normal(0, sigma_px, (2,))
        p2p = p2 + np.random.normal(0, sigma_px, (2,))

        # 5. Compute world distance
        d = world_distance(H, p1p, p2p)
        distances.append(d)

    distances = np.array(distances)
    return distances.mean(), distances.std(), np.percentile(distances, [2.5, 97.5])


# -------------------------
# Example usage
# -------------------------


# Example control points
image_pts = np.array([
    [8, 32],
    [793, 12],
    [24, 783],
    [788, 798]
])


world_pts = np.array([
    [0.0, 0.0],
    [295.0, 0.0],
    [0.0, 295.0],
    [295.0, 295.0]
])  # meters

# Points to measure
p1 = np.array([595, 322])
p2 = np.array([618, 322])

mean, std, ci = measure_distance_monte_carlo(
    image_pts, world_pts,
    p1, p2,
    sigma_px=1.0,
    sigma_gcp=0.01,
    N=2000
)

print(f"Estimated distance: {mean:.3f} mm")
print(f"Std dev: {std:.3f} mm")
print(f"95% CI: {ci}")
