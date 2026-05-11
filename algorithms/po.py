import numpy as np
from .utils import levy_flight


class PO:
    """鹦鹉优化算法"""

    def __init__(self, pop_size=50, max_iter=500):
        self.pop_size = pop_size
        self.max_iter = max_iter

    def optimize(self, func, dim, lb, ub):
        """执行PO优化"""
        n = self.pop_size

        # 初始化种群
        positions = np.random.uniform(lb, ub, (n, dim))
        fitness = np.array([func(pos) for pos in positions])

        best_idx = np.argmin(fitness)
        best_pos = positions[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            mean_pos = np.mean(positions, axis=0)

            for i in range(n):
                # 随机选择一种行为
                behavior = np.random.rand()

                if behavior < 0.25:
                    # 觅食行为
                    new_pos = ((positions[i] - best_pos) * levy_flight(dim)
                               + np.random.rand() * ((1 - t / self.max_iter) ** (2 * t / self.max_iter)) * mean_pos)

                elif behavior < 0.5:
                    # 停留行为
                    new_pos = positions[i] + best_pos * levy_flight(dim) + np.random.rand() * np.ones(dim)

                elif behavior < 0.75:
                    # 交流行为
                    P = np.random.rand()
                    if P <= 0.5:
                        new_pos = 2 * np.random.rand() * (1 - t / self.max_iter) * (positions[i] - mean_pos)
                    else:
                        new_pos = 2 * np.random.rand() * np.exp(-t / (np.random.rand() * self.max_iter + 1e-10)) * positions[i]

                else:
                    # 恐惧陌生人行为
                    new_pos = (positions[i]
                               + np.random.rand() * np.cos(0.5 * np.pi * t / self.max_iter) * (best_pos - positions[i])
                               - np.cos(np.random.rand() * np.pi) * (t / self.max_iter) ** (2 / self.max_iter) * (positions[i] - best_pos))

                # 越界处理
                new_pos = np.clip(new_pos, lb, ub)

                # 贪心更新
                new_fit = func(new_pos)
                if new_fit < fitness[i]:
                    positions[i] = new_pos
                    fitness[i] = new_fit
                    if new_fit < best_fit:
                        best_fit = new_fit
                        best_pos = new_pos.copy()

            convergence.append(best_fit)

        return best_pos, best_fit, convergence
