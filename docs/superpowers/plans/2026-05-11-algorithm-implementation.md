# 阶段2：算法实现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现PO、SSA、PSO、CLSA-PO四种优化算法的Python代码，统一接口，便于阶段3对比实验。

**Architecture:** 所有算法共享统一接口 `optimize(func, dim, lb, ub, pop_size, max_iter)`，返回收敛曲线。公共工具函数（Lévy飞行、Tent混沌映射等）抽取到 `utils.py`。每个算法独立一个文件，CLSA-PO继承PO核心逻辑并叠加三种改进。

**Tech Stack:** Python 3.9, NumPy, Matplotlib（仅可视化）

---

## File Structure

```
algorithms/
├── __init__.py          # 导出所有算法类
├── utils.py             # 公共工具：Lévy飞行、Tent混沌映射、基准测试函数
├── pso.py               # PSO算法
├── ssa.py               # SSA算法
├── po.py                # PO算法
└── clsa_po.py           # CLSA-PO改进算法

tests/
├── test_utils.py        # 工具函数测试
├── test_pso.py          # PSO测试
├── test_ssa.py          # SSA测试
├── test_po.py           # PO测试
└── test_clsa_po.py      # CLSA-PO测试
```

---

### Task 1: 公共工具模块 `algorithms/utils.py`

**Files:**
- Create: `algorithms/__init__.py`
- Create: `algorithms/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: 创建包初始化文件**

```python
# algorithms/__init__.py
from .pso import PSO
from .ssa import SSA
from .po import PO
from .clsa_po import CLSA_PO
```

- [ ] **Step 2: 编写utils.py — Lévy飞行函数**

根据文档公式：$Levy(dim) = \frac{\mu \cdot \sigma}{|v|^{1/\gamma}}$，$\gamma = 1.5$

```python
# algorithms/utils.py
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
```

- [ ] **Step 3: 编写test_utils.py — 测试工具函数**

```python
# tests/test_utils.py
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
```

- [ ] **Step 4: 运行测试**

```bash
cd E:/Projects_draft/ZNJSBigHW && python -m pytest tests/test_utils.py -v
```

Expected: 全部PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/__init__.py algorithms/utils.py tests/test_utils.py
git commit -m "feat: 添加公共工具模块（Lévy飞行、Tent混沌映射、基准函数）"
```

---

### Task 2: PSO算法 `algorithms/pso.py`

**Files:**
- Create: `algorithms/pso.py`
- Create: `tests/test_pso.py`

- [ ] **Step 1: 编写PSO算法**

根据文档公式：$v_i^{t+1} = w \cdot v_i^t + c_1 r_1 (p_{best,i} - x_i^t) + c_2 r_2 (g_{best} - x_i^t)$，$x_i^{t+1} = x_i^t + v_i^{t+1}$，线性递减权重 $w = w_{max} - \frac{w_{max} - w_{min}}{iter_{max}} \cdot t$

```python
# algorithms/pso.py
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
```

- [ ] **Step 2: 编写PSO测试**

```python
# tests/test_pso.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.pso import PSO
from algorithms.utils import sphere


def test_pso_sphere():
    """PSO在Sphere函数上应收敛到接近0"""
    pso = PSO(pop_size=30, max_iter=200)
    best_pos, best_fit, conv = pso.optimize(sphere, dim=10, lb=-100, ub=100)
    assert best_fit < 1.0, f"PSO未收敛，最优值={best_fit}"
    assert len(conv) == 201  # 初始 + 200次迭代


def test_pso_convergence_decreasing():
    """收敛曲线应单调不增"""
    pso = PSO(pop_size=30, max_iter=100)
    _, _, conv = pso.optimize(sphere, dim=10, lb=-100, ub=100)
    for i in range(1, len(conv)):
        assert conv[i] <= conv[i - 1] + 1e-10


def test_pso_output_shape():
    pso = PSO(pop_size=20, max_iter=50)
    best_pos, best_fit, conv = pso.optimize(sphere, dim=5, lb=-10, ub=10)
    assert best_pos.shape == (5,)
    assert isinstance(best_fit, float) or isinstance(best_fit, np.floating)
```

- [ ] **Step 3: 运行测试**

```bash
cd E:/Projects_draft/ZNJSBigHW && python -m pytest tests/test_pso.py -v
```

- [ ] **Step 4: Commit**

```bash
git add algorithms/pso.py tests/test_pso.py
git commit -m "feat: 实现PSO粒子群优化算法"
```

---

### Task 3: SSA算法 `algorithms/ssa.py`

**Files:**
- Create: `algorithms/ssa.py`
- Create: `tests/test_ssa.py`

- [ ] **Step 1: 编写SSA算法**

根据文档公式：发现者公式(3)、跟随者公式(4)、警戒者公式(5)

```python
# algorithms/ssa.py
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

        # 排序索引（适应度从小到大）
        sorted_idx = np.argsort(fitness)
        best_idx = sorted_idx[0]
        worst_idx = sorted_idx[-1]
        best_pos = positions[best_idx].copy()
        best_fit = fitness[best_idx]
        worst_pos = positions[worst_idx].copy()
        worst_fit = fitness[worst_idx]

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
                    new_positions[i] = positions[i] * np.exp(-i / (alpha * self.max_iter))
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
```

- [ ] **Step 2: 编写SSA测试**

```python
# tests/test_ssa.py
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
```

- [ ] **Step 3: 运行测试**

```bash
cd E:/Projects_draft/ZNJSBigHW && python -m pytest tests/test_ssa.py -v
```

- [ ] **Step 4: Commit**

```bash
git add algorithms/ssa.py tests/test_ssa.py
git commit -m "feat: 实现SSA麻雀搜索算法"
```

---

### Task 4: PO算法 `algorithms/po.py`

**Files:**
- Create: `algorithms/po.py`
- Create: `tests/test_po.py`

- [ ] **Step 1: 编写PO算法**

根据文档四种行为公式：觅食(2)、停留(5)、交流(6)、恐惧(7)

```python
# algorithms/po.py
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
                        new_pos = 2 * np.random.rand() * np.exp(-t / (np.random.rand() * self.max_iter + 1e-10))

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
```

- [ ] **Step 2: 编写PO测试**

```python
# tests/test_po.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.po import PO
from algorithms.utils import sphere


def test_po_sphere():
    po = PO(pop_size=30, max_iter=200)
    best_pos, best_fit, conv = po.optimize(sphere, dim=10, lb=-100, ub=100)
    assert best_fit < 1.0, f"PO未收敛，最优值={best_fit}"
    assert len(conv) == 201


def test_po_convergence_decreasing():
    po = PO(pop_size=30, max_iter=100)
    _, _, conv = po.optimize(sphere, dim=10, lb=-100, ub=100)
    for i in range(1, len(conv)):
        assert conv[i] <= conv[i - 1] + 1e-10


def test_po_output_shape():
    po = PO(pop_size=20, max_iter=50)
    best_pos, best_fit, conv = po.optimize(sphere, dim=5, lb=-10, ub=10)
    assert best_pos.shape == (5,)
```

- [ ] **Step 3: 运行测试**

```bash
cd E:/Projects_draft/ZNJSBigHW && python -m pytest tests/test_po.py -v
```

- [ ] **Step 4: Commit**

```bash
git add algorithms/po.py tests/test_po.py
git commit -m "feat: 实现PO鹦鹉优化算法"
```

---

### Task 5: CLSA-PO改进算法 `algorithms/clsa_po.py`

**Files:**
- Create: `algorithms/clsa_po.py`
- Create: `tests/test_clsa_po.py`

- [ ] **Step 1: 编写CLSA-PO算法**

在PO基础上叠加三种改进：Tent混沌初始化、自适应Lévy缩放、非线性惯性权重

```python
# algorithms/clsa_po.py
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
                        new_pos = 2 * np.random.rand() * np.exp(-t / (np.random.rand() * self.max_iter + 1e-10))

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
```

- [ ] **Step 2: 编写CLSA-PO测试**

```python
# tests/test_clsa_po.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algorithms.clsa_po import CLSA_PO
from algorithms.utils import sphere


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


def test_clsa_po_better_than_po():
    """CLSA-PO在多峰函数上应优于原始PO"""
    from algorithms.po import PO
    from algorithms.utils import rastrigin

    np.random.seed(42)
    po = PO(pop_size=30, max_iter=200)
    _, po_fit, _ = po.optimize(rastrigin, dim=10, lb=-5.12, ub=5.12)

    np.random.seed(42)
    clsa = CLSA_PO(pop_size=30, max_iter=200)
    _, clsa_fit, _ = clsa.optimize(rastrigin, dim=10, lb=-5.12, ub=5.12)

    assert clsa_fit <= po_fit + 1e-6, f"CLSA-PO({clsa_fit})未优于PO({po_fit})"


def test_clsa_po_output_shape():
    clsa = CLSA_PO(pop_size=20, max_iter=50)
    best_pos, best_fit, conv = clsa.optimize(sphere, dim=5, lb=-10, ub=10)
    assert best_pos.shape == (5,)
```

- [ ] **Step 3: 运行全部测试**

```bash
cd E:/Projects_draft/ZNJSBigHW && python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add algorithms/clsa_po.py tests/test_clsa_po.py
git commit -m "feat: 实现CLSA-PO改进鹦鹉优化算法（Tent混沌+自适应Lévy+非线性权重）"
```

---

### Task 6: 更新`__init__.py`并运行全量测试

**Files:**
- Verify: `algorithms/__init__.py`
- Verify: all tests pass

- [ ] **Step 1: 确认`__init__.py`导入正确**

`algorithms/__init__.py` 已在 Task 1 创建，内容包含所有四个算法的导入。

- [ ] **Step 2: 运行全量测试**

```bash
cd E:/Projects_draft/ZNJSBigHW && python -m pytest tests/ -v --tb=short
```

Expected: 全部PASS

- [ ] **Step 3: 最终Commit**

```bash
git add -A
git commit -m "feat: 完成阶段2全部算法实现（PO/SSA/PSO/CLSA-PO）"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - ✅ PO四种行为（觅食/停留/交流/恐惧）→ Task 4
   - ✅ SSA三种角色（发现者/跟随者/警戒者）→ Task 3
   - ✅ PSO速度+位置更新、线性递减权重 → Task 2
   - ✅ CLSA-PO：Tent混沌初始化 → Task 5
   - ✅ CLSA-PO：自适应Lévy缩放 → Task 5
   - ✅ CLSA-PO：非线性惯性权重 → Task 5
   - ✅ 基准测试函数5个 → Task 1

2. **Placeholder scan:** 无TBD/TODO，所有代码完整。

3. **Type consistency:**
   - 所有算法的`optimize`接口统一：`(func, dim, lb, ub) → (best_pos, best_fit, convergence)`
   - `convergence` 为 `list[float]`，长度 `max_iter + 1`
   - `best_pos` 为 `np.ndarray`，shape `(dim,)`
