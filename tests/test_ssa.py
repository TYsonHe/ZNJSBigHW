import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.ssa import SSA
from algorithms.utils import sphere


def test_ssa_sphere():
    ssa = SSA(pop_size=30, max_iter=200)
    best_pos, best_fit, conv = ssa.optimize(sphere, dim=10, lb=-100, ub=100)
    assert best_fit < 1.0, f"SSA未收敛，最优值={best_fit}"
    assert len(conv) == 201


def test_ssa_convergence_decreasing():
    ssa = SSA(pop_size=30, max_iter=100)
    _, _, conv = ssa.optimize(sphere, dim=10, lb=-100, ub=100)
    for i in range(1, len(conv)):
        assert conv[i] <= conv[i - 1] + 1e-10


def test_ssa_output_shape():
    ssa = SSA(pop_size=20, max_iter=50)
    best_pos, best_fit, conv = ssa.optimize(sphere, dim=5, lb=-10, ub=10)
    assert best_pos.shape == (5,)
