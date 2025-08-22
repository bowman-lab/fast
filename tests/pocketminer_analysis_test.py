import numpy as np
import pytest
import mdtraj as md
import os

from fast.analysis.pocketminer import physically_capped_voronoi_score, get_backbone_residue_centers
from scipy.spatial import QhullError

@pytest.fixture
def traj():
    pdb_path = os.path.join(os.path.dirname(__file__), "data", "1EXM.pdb")
    return md.load(pdb_path)

np.random.seed(42)

def test_basic_tetrahedron():
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    probs = np.array([1.0, 1.0, 1.0, 1.0])
    score = physically_capped_voronoi_score(points, probs)
    assert abs(score-1/6) < 1e-5, f"Score is out of spec {score}"

def test_second_basic_tetrahedron():
    points = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    probs = np.array([1.0, 1.0, 1.0, 1.0])
    score = physically_capped_voronoi_score(points, probs)
    assert abs(score-8/3) < 1e-5, f"Score is out of spec {score}"

def test_zero_probabilities():
    points = np.random.rand(10, 3)
    probs = np.zeros(10)
    score = physically_capped_voronoi_score(points, probs)
    assert score == 0.0

def test_uniform_probabilities_volume_decreases_with_power():
    points = np.random.rand(20, 3)
    probs = np.ones(20)
    score12 = physically_capped_voronoi_score(points, probs, power=0.5)
    score1 = physically_capped_voronoi_score(points, probs, power=1)
    score2 = physically_capped_voronoi_score(points, probs, power=2)
    assert score2 > 0
    assert score1 > score2
    assert score12 > score1

def test_non_uniform_probabilities_and_power():
    points = np.random.rand(15, 3)
    probs = np.linspace(0, 1, 15)
    score12 = physically_capped_voronoi_score(points, probs, power=0.5)
    score1 = physically_capped_voronoi_score(points, probs, power=1)
    score2 = physically_capped_voronoi_score(points, probs, power=2)
    assert score2 > 0
    assert score1 > score2
    assert score12 > score1

def test_degenerate_coplanar_points():
    points = np.array([[x, y, 0] for x in range(3) for y in range(3)])
    probs = np.ones(len(points))
    with pytest.raises(QhullError):
        score = physically_capped_voronoi_score(points, probs)

def test_degenerate_coplanar_points_non_nitpicky():
    points = np.array([[x, y, 0] for x in range(3) for y in range(3)])
    probs = np.ones(len(points))
    score = physically_capped_voronoi_score(points, probs, nitpicky=False)
    assert isinstance(score, float)
    assert score >= 0

def test_overlapping_points():
    points = np.random.rand(10, 3)
    points[1] = points[0]  # force overlap
    probs = np.ones(10)
    score = physically_capped_voronoi_score(points, probs)
    assert isinstance(score, float)
    assert score >= 0

def test_empty_input():
    points = np.empty((0, 3))
    probs = np.array([])
    score = physically_capped_voronoi_score(points, probs)
    assert score == 0.0

def test_residue_centers_shape(traj):
    centers = get_backbone_residue_centers(traj)

    n_frames = traj.n_frames
    prot_iis = traj.top.select("protein and (name N or name CA or name C or name O)")
    prot_bb = traj.atom_slice(prot_iis)
    n_residues = prot_bb.n_residues
    assert isinstance(centers, np.ndarray), "Output should be a NumPy array"
    assert centers.shape == (n_frames, n_residues, 3), (
        f"Expected shape {(n_frames, n_residues, 3)}, got {centers.shape}"
    )
    assert np.isfinite(centers).all(), "Centers contain NaN or Inf"


def test_residue_centers_shape_singleframe(traj):
    centers = get_backbone_residue_centers(traj[0])

    prot_iis = traj.top.select("protein and (name N or name CA or name C or name O)")
    prot_bb = traj.atom_slice(prot_iis)
    n_residues = prot_bb.n_residues
    assert isinstance(centers, np.ndarray), "Output should be a NumPy array"
    assert centers.shape == (1, n_residues, 3), (
        f"Expected shape {(1, n_residues, 3)}, got {centers.shape}"
    )
    assert np.isfinite(centers).all(), "Centers contain NaN or Inf"
