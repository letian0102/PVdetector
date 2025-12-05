import numpy as np

from peak_valley.alignment import align_distributions


def test_target_landmarks_enforce_positive_gap_when_flat():
    counts = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    landmark_matrix = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ])

    _, aligned_landmarks, _ = align_distributions(
        counts,
        [],
        [],
        landmark_matrix=landmark_matrix,
        target_landmark=[0.0, 1.0, 1.0],
    )

    assert np.all(aligned_landmarks[:, 2] > aligned_landmarks[:, 1])
