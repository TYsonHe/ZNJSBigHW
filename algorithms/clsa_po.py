import numpy as np
from .utils import levy_flight, tent_chaos_init, adaptive_levy_scale, nonlinear_inertia_weight


class CLSA_PO:
    """改进鹦鹉优化算法（Tent混沌初始化 + 自适应Lévy飞行 + 非线性惯性权重）"""

    def __init__(self, pop_size=50, max_iter=500, alpha0=0.3, w_start=0.9, w_end=0.4, k=3):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha0 = alpha0      # Lévy飞行初始步长因子
        self.w_start = w_start    # 初始惯性权重
        self.w_end = w_end        # 最终惯性权重
        self.k = k                # 惯性权重控制参数

    def optimize(self, func, dim, lb, ub):
        """执行CLSA-PO优化"""
        n = self.pop_size

        # 改进1：Tent混沌映射初始化
        chaos_pos = tent_chaos_init(n, dim)
        positions = lb + chaos_pos * (ub - lb)
        fitness = np.array([func(pos) for pos in positions])

        best_idx = np.argmin(fitness)
        best_pos = positions[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # 改进3：非线性递减惯性权重
            w = nonlinear_inertia_weight(t, self.max_iter, self.w_start, self.w_end, self.k)
            # 改进2：自适应Lévy缩放因子
            alpha_t = adaptive_levy_scale(t, self.max_iter, self.alpha0)

            mean_pos = np.mean(positions, axis=0)

            for i in range(n):
                behavior = np.random.rand()

                if behavior < 0.25:
                    # 觅食行为（加入自适应Lévy缩放 + 惯性权重）
                    new_pos = (w * (positions[i] - best_pos) * alpha_t * levy_flight(dim)
                               + np.random.rand() * ((1 - t / self.max_iter) ** (2 * t / self.max_iter)) * mean_pos)

                elif behavior < 0.5:
                    # 停留行为（加入惯性权重）
                    new_pos = w * positions[i] + best_pos * levy_flight(dim) + np.random.rand() * np.ones(dim)

                elif behavior < 0.75:
                    # 交流行为（保持原公式，已有自适应特性）
                    P = np.random.rand()
                    if P <= 0.5:
                        new_pos = 2 * np.random.rand() * (1 - t / self.max_iter) * (positions[i] - mean_pos)
                    else:
                        new_pos = 2 * np.random.rand() * np.exp(-t / (np.random.rand() * self.max_iter + 1e-10)) * positions[i]

                else:
                    # 恐惧陌生人行为（加入惯性权重）
                    new_pos = (w * positions[i]
                               + np.random.rand() * np.cos(0.5 * np.pi * t / self.max_iter) * (best_pos - positions[i])
                               - np.cos(np.random.rand() * np.pi) * (t / self.max_iter) ** (2 / self.max_iter) * (positions[i] - best_pos))

                new_pos = np.clip(new_pos, lb, ub)

                new_fit = func(new_pos)
                if new_fit < fitness[i]:
                    positions[i] = new_pos
                    fitness[i] = new_fit
                    if new_fit < best_fit:
                        best_fit = new_fit
                        best_pos = new_pos.copy()

            convergence.append(best_fit)

        return best_pos, best_fit, convergence
