import numpy as np


class PSO:
    """粒子群优化算法"""

    def __init__(self, pop_size=50, max_iter=500, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def optimize(self, func, dim, lb, ub):
        """执行PSO优化

        Args:
            func: 目标函数
            dim: 问题维度
            lb: 搜索空间下界
            ub: 搜索空间上界

        Returns:
            best_pos: 最优位置
            best_fit: 最优适应度值
            convergence: 收敛曲线（每代最优值）
        """
        # 初始化
        positions = np.random.uniform(lb, ub, (self.pop_size, dim))
        velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.pop_size, dim))
        fitness = np.array([func(pos) for pos in positions])

        p_best_pos = positions.copy()
        p_best_fit = fitness.copy()

        g_best_idx = np.argmin(fitness)
        g_best_pos = positions[g_best_idx].copy()
        g_best_fit = fitness[g_best_idx]

        convergence = [g_best_fit]

        for t in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * t / self.max_iter

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i]
                                 + self.c1 * r1 * (p_best_pos[i] - positions[i])
                                 + self.c2 * r2 * (g_best_pos - positions[i]))

                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                fit = func(positions[i])
                if fit < p_best_fit[i]:
                    p_best_fit[i] = fit
                    p_best_pos[i] = positions[i].copy()
                if fit < g_best_fit:
                    g_best_fit = fit
                    g_best_pos = positions[i].copy()

            convergence.append(g_best_fit)

        return g_best_pos, g_best_fit, convergence
