import numpy as np
from math import gamma, sin, pi


def levy_flight(dim, beta=1.5):
    """Mantegna算法生成Lévy飞行步长"""
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v) ** (1 / beta))
    return step


def tent_chaos_init(pop_size, dim, alpha=0.5):
    """Tent混沌映射生成初始种群位置（映射到[0,1]）"""
    positions = np.zeros((pop_size, dim))
    for i in range(pop_size):
        z = np.random.rand()
        for j in range(dim):
            if z < alpha:
                z = z / alpha
            else:
                z = (1 - z) / (1 - alpha)
            # 防止停滞在0
            if z == 0:
                z = np.random.rand() * 0.1
            positions[i, j] = z
    return positions


def adaptive_levy_scale(t, max_iter, alpha0=0.3):
    """自适应Lévy飞行缩放因子：alpha(t) = alpha0 * exp(-t/max_iter)"""
    return alpha0 * np.exp(-t / max_iter)


def nonlinear_inertia_weight(t, max_iter, w_start=0.9, w_end=0.4, k=3):
    """非线性递减惯性权重：w(t) = w_end + (w_start - w_end) * exp(-k * (t/max_iter)^2)"""
    return w_end + (w_start - w_end) * np.exp(-k * (t / max_iter) ** 2)


# ========== 基准测试函数 ==========
def sphere(x):
    """Sphere函数，单峰，全局最优0"""
    return np.sum(x ** 2)


def rosenbrock(x):
    """Rosenbrock函数，单峰，全局最优0"""
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def rastrigin(x):
    """Rastrigin函数，多峰，全局最优0"""
    return np.sum(x ** 2 - 10 * np.cos(2 * pi * x) + 10)


def ackley(x):
    """Ackley函数，多峰，全局最优0"""
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def griewank(x):
    """Griewank函数，多峰，全局最优0"""
    i = np.arange(1, len(x) + 1)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1


# 基准函数配置表
BENCHMARK_FUNCTIONS = {
    'Sphere': {'func': sphere, 'lb': -100, 'ub': 100},
    'Rosenbrock': {'func': rosenbrock, 'lb': -30, 'ub': 30},
    'Rastrigin': {'func': rastrigin, 'lb': -5.12, 'ub': 5.12},
    'Ackley': {'func': ackley, 'lb': -32, 'ub': 32},
    'Griewank': {'func': griewank, 'lb': -600, 'ub': 600},
}
