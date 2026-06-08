# 路径规划算法对比报告 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份围绕物流配送路径规划中 ACO 与 GA 对比的实验代码、实验结果、中文学术报告草稿和最终导出文件。

**Architecture:** 所有产物保存在 `final_report`。实验代码使用 Python 标准库实现 ACO 与 GA，并用手写 SVG 输出图表，避免引入额外依赖；报告使用 Markdown 撰写，最后根据可用工具导出 PDF。

**Tech Stack:** Python 3 标准库、Markdown、CSV、SVG、PowerShell/命令行导出工具。

---

## File Structure

- Create: `final_report/experiment.py`  
  负责生成配送点、实现 ACO 与 GA、运行多次实验、输出 CSV 与 SVG。
- Create: `final_report/results/summary.csv`  
  保存算法、实例、最优值、平均值、标准差、运行时间等汇总结果。
- Create: `final_report/results/convergence.svg`  
  保存 ACO 与 GA 在主实验上的平均收敛曲线。
- Create: `final_report/results/routes.svg`  
  保存主实验上两种算法最优路径对比图。
- Create: `final_report/report.md`  
  保存 5000-6000 字中文报告正文。
- Create: `final_report/references.md`  
  保存参考文献条目与引用说明。
- Modify: `final_report/task_plan.md`  
  更新阶段状态。
- Modify: `final_report/findings.md`  
  记录实验观察和可写入报告的发现。
- Modify: `final_report/progress.md`  
  记录执行日志。

---

### Task 1: 准备实验代码骨架

**Files:**
- Create: `final_report/experiment.py`

- [ ] **Step 1: 写入可执行骨架**

```python
from __future__ import annotations

import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

@dataclass
class RunResult:
    algorithm: str
    instance: str
    best_length: float
    mean_length: float
    std_length: float
    mean_time: float
    best_route: list[int]
    mean_curve: list[float]


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[euclidean(p, q) for q in points] for p in points]


def route_length(route: list[int], dist: list[list[float]]) -> float:
    return sum(dist[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("experiment skeleton ready")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行骨架**

Run: `python final_report/experiment.py`  
Expected: 输出 `experiment skeleton ready`，并创建 `final_report/results`。

---

### Task 2: 实现数据实例与基础测试函数

**Files:**
- Modify: `final_report/experiment.py`

- [ ] **Step 1: 添加实例生成函数**

```python
def random_instance(n: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]


def fixed_instance() -> list[tuple[float, float]]:
    return [
        (10, 10), (18, 22), (25, 5), (31, 35), (42, 18),
        (48, 42), (55, 8), (63, 28), (70, 12), (78, 38),
        (82, 20), (90, 45), (12, 55), (20, 72), (36, 60),
        (44, 80), (58, 66), (69, 82), (75, 58), (88, 75),
    ]
```

- [ ] **Step 2: 添加基础自检函数**

```python
def self_check() -> None:
    points = [(0, 0), (3, 4), (6, 0)]
    dist = build_distance_matrix(points)
    assert round(dist[0][1], 6) == 5
    assert round(route_length([0, 1, 2], dist), 6) == round(5 + 5 + 6, 6)
```

- [ ] **Step 3: 在 main 中调用自检**

```python
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    self_check()
    print("self check passed")
```

- [ ] **Step 4: 运行自检**

Run: `python final_report/experiment.py`  
Expected: 输出 `self check passed`。

---

### Task 3: 实现遗传算法 GA

**Files:**
- Modify: `final_report/experiment.py`

- [ ] **Step 1: 添加 GA 所需函数**

```python
def ordered_crossover(a: list[int], b: list[int], rng: random.Random) -> list[int]:
    n = len(a)
    i, j = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[i:j + 1] = a[i:j + 1]
    fill = [x for x in b if x not in child]
    k = 0
    for idx in range(n):
        if child[idx] == -1:
            child[idx] = fill[k]
            k += 1
    return child


def mutate_swap(route: list[int], rng: random.Random, rate: float) -> None:
    if rng.random() < rate:
        i, j = rng.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]


def tournament(population: list[list[int]], dist: list[list[float]], rng: random.Random) -> list[int]:
    candidates = rng.sample(population, 3)
    return min(candidates, key=lambda r: route_length(r, dist))[:]
```

- [ ] **Step 2: 添加 GA 主函数**

```python
def run_ga(points: list[tuple[float, float]], seed: int, generations: int = 120, pop_size: int = 80) -> tuple[float, list[int], list[float]]:
    rng = random.Random(seed)
    dist = build_distance_matrix(points)
    n = len(points)
    base = list(range(n))
    population = []
    for _ in range(pop_size):
        route = base[:]
        rng.shuffle(route)
        population.append(route)

    best_route = min(population, key=lambda r: route_length(r, dist))[:]
    best = route_length(best_route, dist)
    curve = []

    for _ in range(generations):
        population.sort(key=lambda r: route_length(r, dist))
        next_population = [population[0][:], population[1][:]]
        while len(next_population) < pop_size:
            p1 = tournament(population, dist, rng)
            p2 = tournament(population, dist, rng)
            child = ordered_crossover(p1, p2, rng)
            mutate_swap(child, rng, 0.18)
            next_population.append(child)
        population = next_population
        current = min(population, key=lambda r: route_length(r, dist))
        current_len = route_length(current, dist)
        if current_len < best:
            best = current_len
            best_route = current[:]
        curve.append(best)
    return best, best_route, curve
```

- [ ] **Step 3: 验证 GA 可运行**

Run: `python -c "import final_report.experiment as e; p=e.random_instance(12, 1); print(round(e.run_ga(p, 1, 5, 20)[0], 2))"`  
Expected: 输出一个正数路径长度。

---

### Task 4: 实现蚁群算法 ACO

**Files:**
- Modify: `final_report/experiment.py`

- [ ] **Step 1: 添加 ACO 路径构造函数**

```python
def choose_next(current: int, unvisited: set[int], pheromone: list[list[float]], dist: list[list[float]], rng: random.Random, alpha: float, beta: float) -> int:
    weights = []
    for city in unvisited:
        tau = pheromone[current][city] ** alpha
        eta = (1.0 / (dist[current][city] + 1e-9)) ** beta
        weights.append((city, tau * eta))
    total = sum(w for _, w in weights)
    pick = rng.random() * total
    acc = 0.0
    for city, weight in weights:
        acc += weight
        if acc >= pick:
            return city
    return weights[-1][0]
```

- [ ] **Step 2: 添加 ACO 主函数**

```python
def run_aco(points: list[tuple[float, float]], seed: int, iterations: int = 120, ants: int = 50) -> tuple[float, list[int], list[float]]:
    rng = random.Random(seed)
    dist = build_distance_matrix(points)
    n = len(points)
    pheromone = [[1.0 for _ in range(n)] for _ in range(n)]
    alpha = 1.0
    beta = 4.0
    evaporation = 0.45
    q = 100.0
    best = float("inf")
    best_route: list[int] = []
    curve = []

    for _ in range(iterations):
        routes = []
        for _ant in range(ants):
            start = rng.randrange(n)
            route = [start]
            unvisited = set(range(n))
            unvisited.remove(start)
            while unvisited:
                nxt = choose_next(route[-1], unvisited, pheromone, dist, rng, alpha, beta)
                route.append(nxt)
                unvisited.remove(nxt)
            length = route_length(route, dist)
            routes.append((route, length))
            if length < best:
                best = length
                best_route = route[:]

        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= 1.0 - evaporation
        for route, length in routes:
            deposit = q / length
            for i in range(n):
                a = route[i]
                b = route[(i + 1) % n]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit
        curve.append(best)
    return best, best_route, curve
```

- [ ] **Step 3: 验证 ACO 可运行**

Run: `python -c "import final_report.experiment as e; p=e.random_instance(12, 1); print(round(e.run_aco(p, 1, 5, 10)[0], 2))"`  
Expected: 输出一个正数路径长度。

---

### Task 5: 汇总多次实验并输出 CSV

**Files:**
- Modify: `final_report/experiment.py`
- Create: `final_report/results/summary.csv`

- [ ] **Step 1: 添加多次运行函数**

```python
def average_curves(curves: list[list[float]]) -> list[float]:
    length = min(len(c) for c in curves)
    return [statistics.mean(c[i] for c in curves) for i in range(length)]


def evaluate_algorithm(name: str, runner, instance_name: str, points: list[tuple[float, float]], seeds: list[int]) -> RunResult:
    lengths = []
    times = []
    curves = []
    best_route: list[int] = []
    best_length = float("inf")
    for seed in seeds:
        start = time.perf_counter()
        length, route, curve = runner(points, seed)
        elapsed = time.perf_counter() - start
        lengths.append(length)
        times.append(elapsed)
        curves.append(curve)
        if length < best_length:
            best_length = length
            best_route = route[:]
    return RunResult(
        algorithm=name,
        instance=instance_name,
        best_length=best_length,
        mean_length=statistics.mean(lengths),
        std_length=statistics.pstdev(lengths),
        mean_time=statistics.mean(times),
        best_route=best_route,
        mean_curve=average_curves(curves),
    )


def write_summary(results: list[RunResult]) -> None:
    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "algorithm", "best_length", "mean_length", "std_length", "mean_time_seconds"])
        for r in results:
            writer.writerow([r.instance, r.algorithm, f"{r.best_length:.4f}", f"{r.mean_length:.4f}", f"{r.std_length:.4f}", f"{r.mean_time:.4f}"])
```

- [ ] **Step 2: 修改 main 运行实验**

```python
def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    self_check()
    seeds = list(range(10))
    instances = [
        ("random_30", random_instance(30, 20260608)),
        ("fixed_20", fixed_instance()),
    ]
    results: list[RunResult] = []
    for instance_name, points in instances:
        results.append(evaluate_algorithm("ACO", run_aco, instance_name, points, seeds))
        results.append(evaluate_algorithm("GA", run_ga, instance_name, points, seeds))
    write_summary(results)
    print("wrote results/summary.csv")
```

- [ ] **Step 3: 运行完整实验**

Run: `python final_report/experiment.py`  
Expected: 输出 `wrote results/summary.csv`，生成 `final_report/results/summary.csv`。

---

### Task 6: 输出 SVG 图表

**Files:**
- Modify: `final_report/experiment.py`
- Create: `final_report/results/convergence.svg`
- Create: `final_report/results/routes.svg`

- [ ] **Step 1: 添加 SVG 工具函数**

```python
def scale_points(points: list[tuple[float, float]], width: int, height: int, margin: int) -> list[tuple[float, float]]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    sx = (width - 2 * margin) / (max_x - min_x + 1e-9)
    sy = (height - 2 * margin) / (max_y - min_y + 1e-9)
    return [(margin + (x - min_x) * sx, height - margin - (y - min_y) * sy) for x, y in points]


def polyline(points: list[tuple[float, float]], color: str) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" opacity="0.85" />'
```

- [ ] **Step 2: 添加收敛曲线输出函数**

```python
def write_convergence_svg(results: list[RunResult]) -> None:
    selected = [r for r in results if r.instance == "random_30"]
    width, height, margin = 900, 520, 60
    all_values = [v for r in selected for v in r.mean_curve]
    min_v, max_v = min(all_values), max(all_values)
    colors = {"ACO": "#1f77b4", "GA": "#d62728"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20">Random-30 平均收敛曲线</text>')
    lines.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>')
    lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>')
    for r in selected:
        pts = []
        for i, value in enumerate(r.mean_curve):
            x = margin + i * (width - 2 * margin) / (len(r.mean_curve) - 1)
            y = height - margin - (value - min_v) * (height - 2 * margin) / (max_v - min_v + 1e-9)
            pts.append((x, y))
        lines.append(polyline(pts, colors[r.algorithm]))
        lines.append(f'<text x="{width-margin-120}" y="{80 + 25 * len(lines) % 80}" fill="{colors[r.algorithm]}" font-size="16">{r.algorithm}</text>')
    lines.append('</svg>')
    (RESULTS_DIR / "convergence.svg").write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 3: 添加路径图输出函数**

```python
def write_routes_svg(results: list[RunResult], points: list[tuple[float, float]]) -> None:
    selected = [r for r in results if r.instance == "random_30"]
    width, height, margin = 1000, 520, 50
    scaled = scale_points(points, width // 2 - 40, height, margin)
    colors = {"ACO": "#1f77b4", "GA": "#d62728"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    for panel, r in enumerate(selected):
        offset = panel * (width // 2)
        route_points = [(scaled[i][0] + offset, scaled[i][1]) for i in r.best_route]
        route_points.append(route_points[0])
        lines.append(f'<text x="{offset + width/4}" y="30" text-anchor="middle" font-size="20">{r.algorithm} 最优路径</text>')
        lines.append(polyline(route_points, colors[r.algorithm]))
        for idx, (x, y) in enumerate(scaled):
            lines.append(f'<circle cx="{x + offset:.1f}" cy="{y:.1f}" r="4" fill="black"/>')
            lines.append(f'<text x="{x + offset + 5:.1f}" y="{y - 5:.1f}" font-size="10">{idx}</text>')
    lines.append('</svg>')
    (RESULTS_DIR / "routes.svg").write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 在 main 中调用图表输出**

```python
    write_convergence_svg(results)
    write_routes_svg(results, instances[0][1])
    print("wrote summary.csv, convergence.svg and routes.svg")
```

- [ ] **Step 5: 重新运行实验**

Run: `python final_report/experiment.py`  
Expected: 生成 `summary.csv`、`convergence.svg`、`routes.svg`。

---

### Task 7: 撰写报告正文

**Files:**
- Create: `final_report/report.md`
- Create: `final_report/references.md`
- Modify: `final_report/findings.md`

- [ ] **Step 1: 写入参考文献文件**

```markdown
# 参考文献记录

[1] Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 26(1), 29-41.

[2] Holland, J. H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.

[3] Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley.

[4] Applegate, D. L., Bixby, R. E., Chvatal, V., & Cook, W. J. (2006). The Traveling Salesman Problem: A Computational Study. Princeton University Press.
```

- [ ] **Step 2: 根据实验结果撰写 `report.md`**

报告必须包含这些一级标题：

```markdown
# 面向物流配送路径规划的蚁群算法与遗传算法比较研究

## 摘要

## 1 引言

## 2 问题建模与评价框架

## 3 蚁群算法与遗传算法的机制比较

## 4 实验设计

## 5 实验结果与分析

## 6 讨论：优势、适用边界与局限

## 7 个人贡献总结

## 8 结论

## 参考文献

## 附录：AI 使用说明与反思
```

- [ ] **Step 3: 检查报告长度和关键要求**

Run: `python -c "from pathlib import Path; t=Path('final_report/report.md').read_text(encoding='utf-8'); print(len(t)); print(all(x in t for x in ['个人贡献', 'AI 使用说明', '优势', '局限', '蚁群算法', '遗传算法']))"`  
Expected: 字符数足以支撑 5000-6000 字正文，第二行输出 `True`。

---

### Task 8: 导出与最终检查

**Files:**
- Create: `final_report/学号-姓名-面向物流配送路径规划的蚁群算法与遗传算法比较研究.pdf`
- Modify: `final_report/task_plan.md`
- Modify: `final_report/progress.md`

- [ ] **Step 1: 确认学号和姓名**

向用户询问最终 PDF 文件名所需的学号和姓名。没有这两个信息时，不生成最终命名 PDF。

- [ ] **Step 2: 检查可用导出工具**

Run: `pandoc --version`  
Expected: 若可用，使用 pandoc 导出 PDF；若不可用，检查是否可以使用 Word、浏览器打印或其它本机工具。

- [ ] **Step 3: 导出 PDF**

首选命令：

```powershell
pandoc "final_report/report.md" -o "final_report/学号-姓名-面向物流配送路径规划的蚁群算法与遗传算法比较研究.pdf"
```

- [ ] **Step 4: 最终验证**

Run: `python -c "from pathlib import Path; print(Path('final_report/report.md').exists()); print(Path('final_report/results/summary.csv').exists()); print(Path('final_report/results/convergence.svg').exists()); print(Path('final_report/results/routes.svg').exists())"`  
Expected: 四行均为 `True`。

---

## Self-Review

- Spec coverage: 计划覆盖选题、实验代码、结果输出、报告正文、参考文献、AI 使用反思和最终导出。
- Placeholder scan: 计划没有使用 TBD、TODO 或未定义任务；唯一外部输入是用户学号与姓名，用于最终文件命名。
- Type consistency: `RunResult`、`run_aco`、`run_ga`、`write_summary`、`write_convergence_svg`、`write_routes_svg` 在各任务中保持一致。
