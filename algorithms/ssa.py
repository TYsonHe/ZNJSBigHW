import numpy as np


class SSA:
    """麻雀搜索算法"""

    def __init__(self, pop_size=50, max_iter=500, pd_ratio=0.2, sd_ratio=0.1, st=0.8):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pd_ratio = pd_ratio   # 发现者比例
        self.sd_ratio = sd_ratio   # 警戒者比例
        self.st = st               # 安全阈值

    def optimize(self, func, dim, lb, ub):
        """执行SSA优化"""
        n = self.pop_size
        pd_num = int(n * self.pd_ratio)
        sd_num = int(n * self.sd_ratio)

        # 初始化种群
        positions = np.random.uniform(lb, ub, (n, dim))
        fitness = np.array([func(pos) for pos in positions])

        best_pos = positions[np.argmin(fitness)].copy()
        best_fit = np.min(fitness)

        convergence = [best_fit]

        for t in range(self.max_iter):
            # 按适应度排序
            sorted_idx = np.argsort(fitness)
            positions = positions[sorted_idx]
            fitness = fitness[sorted_idx]

            best_pos = positions[0].copy()
            best_fit = fitness[0]
            worst_pos = positions[-1].copy()
            worst_fit = fitness[-1]

            R2 = np.random.rand()
            new_positions = positions.copy()

            # 发现者更新
            for i in range(pd_num):
                alpha = np.random.rand()
                if R2 < self.st:
                    # 无捕食者，广泛搜索
                    new_positions[i] = positions[i] * np.exp(-i / (alpha * self.max_iter + 1e-10))
                else:
                    # 发现捕食者，飞向安全区域
                    Q = np.random.normal(0, 1, dim)
                    new_positions[i] = positions[i] + Q

            # 跟随者更新
            for i in range(pd_num, n):
                if i > n / 2:
                    # 适应度较差的跟随者
                    Q = np.random.normal(0, 1, dim)
                    new_positions[i] = Q * np.exp((worst_pos - positions[i]) / (i ** 2 + 1e-10))
                else:
                    # 在发现者附近搜索
                    A = np.random.choice([-1, 1], dim)
                    A_plus = np.linalg.pinv(A.reshape(1, -1))
                    L = np.ones(dim)
                    new_positions[i] = best_pos + np.abs(positions[i] - best_pos) * A_plus.flatten() * L

            # 警戒者更新（从种群中随机选取sd_num个）
            sd_indices = np.random.choice(n, sd_num, replace=False)
            for idx in sd_indices:
                beta = np.random.normal(0, 1)
                K = np.random.uniform(-1, 1)
                epsilon = 1e-10
                f_i = fitness[idx]
                f_g = best_fit
                f_w = worst_fit

                if f_i > f_g:
                    # 种群边缘，向最优位置靠近
                    new_positions[idx] = best_pos + beta * np.abs(positions[idx] - best_pos)
                else:
                    # 种群中间，意识到危险
                    new_positions[idx] = positions[idx] + K * (
                        np.abs(positions[idx] - worst_pos) / (f_i - f_w + epsilon)
                    )

            # 越界处理与适应度更新
            new_positions = np.clip(new_positions, lb, ub)
            for i in range(n):
                new_fit = func(new_positions[i])
                if new_fit < fitness[i]:
                    positions[i] = new_positions[i]
                    fitness[i] = new_fit
                    if new_fit < best_fit:
                        best_fit = new_fit
                        best_pos = new_positions[i].copy()

            convergence.append(best_fit)

        return best_pos, best_fit, convergence
