from __future__ import annotations

import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


def self_check() -> None:
    points = [(0, 0), (3, 4), (6, 0)]
    dist = build_distance_matrix(points)
    assert round(dist[0][1], 6) == 5
    assert round(route_length([0, 1, 2], dist), 6) == 16


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


def run_ga(
    points: list[tuple[float, float]],
    seed: int,
    generations: int = 120,
    pop_size: int = 80,
) -> tuple[float, list[int], list[float]]:
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


def choose_next(
    current: int,
    unvisited: set[int],
    pheromone: list[list[float]],
    dist: list[list[float]],
    rng: random.Random,
    alpha: float,
    beta: float,
) -> int:
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


def run_aco(
    points: list[tuple[float, float]],
    seed: int,
    iterations: int = 120,
    ants: int = 50,
) -> tuple[float, list[int], list[float]]:
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


def average_curves(curves: list[list[float]]) -> list[float]:
    length = min(len(c) for c in curves)
    return [statistics.mean(c[i] for c in curves) for i in range(length)]


def evaluate_algorithm(
    name: str,
    runner: Callable[[list[tuple[float, float]], int], tuple[float, list[int], list[float]]],
    instance_name: str,
    points: list[tuple[float, float]],
    seeds: list[int],
) -> RunResult:
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
            writer.writerow([
                r.instance,
                r.algorithm,
                f"{r.best_length:.4f}",
                f"{r.mean_length:.4f}",
                f"{r.std_length:.4f}",
                f"{r.mean_time:.4f}",
            ])


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


def write_convergence_svg(results: list[RunResult]) -> None:
    selected = [r for r in results if r.instance == "random_30"]
    width, height, margin = 900, 520, 60
    all_values = [v for r in selected for v in r.mean_curve]
    min_v, max_v = min(all_values), max(all_values)
    colors = {"ACO": "#1f77b4", "GA": "#d62728"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20">Random-30 平均收敛曲线</text>')
    lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
    lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>')
    for index, r in enumerate(selected):
        pts = []
        for i, value in enumerate(r.mean_curve):
            x = margin + i * (width - 2 * margin) / (len(r.mean_curve) - 1)
            y = height - margin - (value - min_v) * (height - 2 * margin) / (max_v - min_v + 1e-9)
            pts.append((x, y))
        lines.append(polyline(pts, colors[r.algorithm]))
        lines.append(f'<text x="{width - margin - 120}" y="{80 + 25 * index}" fill="{colors[r.algorithm]}" font-size="16">{r.algorithm}</text>')
    lines.append('</svg>')
    (RESULTS_DIR / "convergence.svg").write_text("\n".join(lines), encoding="utf-8")


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
        lines.append(f'<text x="{offset + width / 4}" y="30" text-anchor="middle" font-size="20">{r.algorithm} 最优路径</text>')
        lines.append(polyline(route_points, colors[r.algorithm]))
        for idx, (x, y) in enumerate(scaled):
            lines.append(f'<circle cx="{x + offset:.1f}" cy="{y:.1f}" r="4" fill="black"/>')
            lines.append(f'<text x="{x + offset + 5:.1f}" y="{y - 5:.1f}" font-size="10">{idx}</text>')
    lines.append('</svg>')
    (RESULTS_DIR / "routes.svg").write_text("\n".join(lines), encoding="utf-8")


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
    write_convergence_svg(results)
    write_routes_svg(results, instances[0][1])
    print("wrote summary.csv, convergence.svg and routes.svg")


if __name__ == "__main__":
    main()
