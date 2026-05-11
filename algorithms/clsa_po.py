import numpy as np
from .utils import levy_flight, tent_chaos_init, adaptive_levy_scale, nonlinear_inertia_weight


class CLSA_PO:
    """改进鹦鹉优化算法（Tent混沌初始化 + 自适应Lévy飞行 + 非线性惯性权重）

    核心改进逻辑：
    - 自适应Lévy缩放alpha_t：初期大步长探索，后期小步长精细搜索
    - 反向惯性权重(1-w_t)：初期弱收敛保持探索，后期强收敛加速开发
    两者协同实现探索→开发的自适应过渡
    """

    def __init__(self, pop_size=50, max_iter=500, alpha_max=1.5, alpha_min=0.1, w_start=0.9, w_end=0.4, k=2):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha_max = alpha_max  # Lévy飞行初始步长因子（探索阶段）
        self.alpha_min = alpha_min  # Lévy飞行最终步长因子（开发阶段）
        self.w_start = w_start      # 初始惯性权重
        self.w_end = w_end          # 最终惯性权重
        self.k = k                  # 惯性权重非线性控制参数

    def optimize(self, func, dim, lb, ub):
        """执行CLSA-PO优化"""
        n = self.pop_size

        # 改进1：Tent混沌映射初始化（覆盖更均匀）
        chaos_pos = tent_chaos_init(n, dim)
        positions = lb + chaos_pos * (ub - lb)
        fitness = np.array([func(pos) for pos in positions])

        best_idx = np.argmin(fitness)
        best_pos = positions[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence = [best_fit]

        for t in range(self.max_iter):
            # 改进3：非线性递减惯性权重（前期大→保留探索，后期小→加强开发）
            w = nonlinear_inertia_weight(t, self.max_iter, self.w_start, self.w_end, self.k)
            # 反向惯性权重：初期小（弱收敛→探索），后期大（强收敛→开发）
            w_exploit = self.w_start + self.w_end - w
            # 改进2：自适应Lévy缩放因子（初期大步长探索，后期小步长精细搜索）
            alpha_t = adaptive_levy_scale(t, self.max_iter, self.alpha_max, self.alpha_min)

            mean_pos = np.mean(positions, axis=0)

            for i in range(n):
                behavior = np.random.rand()

                if behavior < 0.25:
                    # 觅食行为：自适应Lévy步长控制探索-开发平衡
                    # 初期alpha_t大→大步长全局搜索，后期alpha_t小→精细局部搜索
                    new_pos = (alpha_t * (positions[i] - best_pos) * levy_flight(dim)
                               + np.random.rand() * ((1 - t / self.max_iter) ** (2 * t / self.max_iter)) * mean_pos)

                elif behavior < 0.5:
                    # 停留行为：自适应Lévy缩放控制向最优位置搜索的步长
                    # 初期大步长在最优位置附近广泛搜索，后期小步长精细逼近
                    new_pos = positions[i] + alpha_t * best_pos * levy_flight(dim) + np.random.rand() * np.ones(dim)

                elif behavior < 0.75:
                    # 交流行为（保持原公式，已有自适应特性）
                    P = np.random.rand()
                    if P <= 0.5:
                        new_pos = 2 * np.random.rand() * (1 - t / self.max_iter) * (positions[i] - mean_pos)
                    else:
                        new_pos = 2 * np.random.rand() * np.exp(-t / (np.random.rand() * self.max_iter + 1e-10)) * positions[i]

                else:
                    # 恐惧陌生人行为：反向惯性权重调节向最优位置靠拢的力度
                    # 初期w_exploit小→弱吸引→保持探索多样性
                    # 后期w_exploit大→强吸引→加速收敛至最优
                    new_pos = (positions[i]
                               + w_exploit * np.random.rand() * np.cos(0.5 * np.pi * t / self.max_iter) * (best_pos - positions[i])
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
