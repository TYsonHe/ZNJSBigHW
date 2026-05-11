import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.utils import (
    levy_flight, tent_chaos_init, adaptive_levy_scale,
    nonlinear_inertia_weight, sphere, rosenbrock, rastrigin, ackley, griewank,
    BENCHMARK_FUNCTIONS
)


def test_levy_flight_shape():
    step = levy_flight(30)
    assert step.shape == (30,)
    assert not np.any(np.isnan(step))


def test_tent_chaos_range():
    pos = tent_chaos_init(50, 30)
    assert pos.shape == (50, 30)
    assert np.all(pos >= 0) and np.all(pos <= 1)


def test_tent_chaos_diversity():
    """混沌初始化的种群应比随机初始化更均匀"""
    pos = tent_chaos_init(1000, 10)
    std = np.std(pos)
    assert std > 0.1  # 不应过度聚集


def test_adaptive_levy_scale_decreasing():
    s0 = adaptive_levy_scale(0, 500)
    s1 = adaptive_levy_scale(250, 500)
    s2 = adaptive_levy_scale(500, 500)
    assert s0 > s1 > s2
    assert abs(s0 - 0.3) < 1e-10


def test_nonlinear_inertia_weight():
    w0 = nonlinear_inertia_weight(0, 500)
    w_end = nonlinear_inertia_weight(500, 500)
    assert abs(w0 - 0.9) < 1e-10
    assert w_end < 0.9
    assert w_end > 0.4


def test_benchmark_functions_at_optimum():
    x_opt = np.zeros(30)
    assert abs(sphere(x_opt)) < 1e-10
    assert abs(rastrigin(x_opt)) < 1e-10
    assert abs(ackley(x_opt)) < 1e-10
    assert abs(griewank(x_opt)) < 1e-10
    x_rb = np.ones(30)
    assert abs(rosenbrock(x_rb)) < 1e-10


def test_benchmark_config():
    assert len(BENCHMARK_FUNCTIONS) == 5
    for name, cfg in BENCHMARK_FUNCTIONS.items():
        assert 'func' in cfg
        assert 'lb' in cfg
        assert 'ub' in cfg
        assert cfg['lb'] < cfg['ub']
