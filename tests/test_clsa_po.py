import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.clsa_po import CLSA_PO
from algorithms.utils import sphere, rastrigin


def test_clsa_po_sphere():
    clsa = CLSA_PO(pop_size=30, max_iter=200)
    best_pos, best_fit, conv = clsa.optimize(sphere, dim=10, lb=-100, ub=100)
    assert best_fit < 1.0, f"CLSA-PO未收敛，最优值={best_fit}"
    assert len(conv) == 201


def test_clsa_po_convergence_decreasing():
    clsa = CLSA_PO(pop_size=30, max_iter=100)
    _, _, conv = clsa.optimize(sphere, dim=10, lb=-100, ub=100)
    for i in range(1, len(conv)):
        assert conv[i] <= conv[i - 1] + 1e-10


def test_clsa_po_output_shape():
    clsa = CLSA_PO(pop_size=20, max_iter=50)
    best_pos, best_fit, conv = clsa.optimize(sphere, dim=5, lb=-10, ub=10)
    assert best_pos.shape == (5,)
