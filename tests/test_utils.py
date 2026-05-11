import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.utils import (
    levy_flight, tent_chaos_init, adaptive_levy_scale,
    nonlinear_inertia_weight, sphere, rosenbrock, rastrigin, ackley, griewank,
    schwefel, levy as levy_func, alpine,
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
    s0 = adaptive_levy_scale(0, 500, alpha_max=1.0, alpha_min=0.1)
    s1 = adaptive_levy_scale(250, 500, alpha_max=1.0, alpha_min=0.1)
    s2 = adaptive_levy_scale(500, 500, alpha_max=1.0, alpha_min=0.1)
    assert s0 > s1 > s2
    assert abs(s0 - 1.0) < 1e-10  # 初始值=alpha_max
    assert abs(s2 - 0.1) < 1e-10  # 终止值=alpha_min


def test_nonlinear_inertia_weight():
    w0 = nonlinear_inertia_weight(0, 500, w_start=0.9, w_end=0.4, k=2)
    w_end = nonlinear_inertia_weight(500, 500, w_start=0.9, w_end=0.4, k=2)
    assert abs(w0 - 0.9) < 1e-10  # 初始值=w_start
    assert abs(w_end - 0.4) < 1e-10  # 终止值=w_end


def test_benchmark_functions_at_optimum():
    x_opt = np.zeros(30)
    assert abs(sphere(x_opt)) < 1e-10
    assert abs(rastrigin(x_opt)) < 1e-10
    assert abs(ackley(x_opt)) < 1e-10
    assert abs(griewank(x_opt)) < 1e-10
    assert abs(alpine(x_opt)) < 1e-10
    x_rb = np.ones(30)
    assert abs(rosenbrock(x_rb)) < 1e-10
    assert abs(levy_func(x_rb)) < 1e-10
    # Schwefel最优点近似420.9687
    x_sw = np.full(30, 420.9687)
    assert schwefel(x_sw) < 1.0  # 近似最优，允许小误差


def test_benchmark_config():
    assert len(BENCHMARK_FUNCTIONS) == 8
    for name, cfg in BENCHMARK_FUNCTIONS.items():
        assert 'func' in cfg
        assert 'lb' in cfg
        assert 'ub' in cfg
        assert cfg['lb'] < cfg['ub']
