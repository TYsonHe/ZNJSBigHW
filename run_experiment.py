"""
阶段3：对比实验运行脚本
在5个基准测试函数上运行4种算法（PO、CLSA-PO、SSA、PSO）各30次，
输出统计表格（最优值、均值、标准差）和收敛曲线对比图。
"""
import numpy as np
import json
import os

from algorithms.pso import PSO
from algorithms.ssa import SSA
from algorithms.po import PO
from algorithms.clsa_po import CLSA_PO
from algorithms.utils import BENCHMARK_FUNCTIONS

# ========== 实验参数 ==========
POP_SIZE = 50
MAX_ITER = 500
DIM = 30
RUNS = 30

ALGORITHMS = {
    'PSO': PSO(pop_size=POP_SIZE, max_iter=MAX_ITER),
    'SSA': SSA(pop_size=POP_SIZE, max_iter=MAX_ITER),
    'PO': PO(pop_size=POP_SIZE, max_iter=MAX_ITER),
    'CLSA-PO': CLSA_PO(pop_size=POP_SIZE, max_iter=MAX_ITER),
}

OUTPUT_DIR = 'output'


def run_single(algorithm, func, dim, lb, ub):
    """运行单次实验"""
    best_pos, best_fit, convergence = algorithm.optimize(func, dim, lb, ub)
    return best_fit, convergence


def run_experiment():
    """运行全部对比实验"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 存储所有结果
    results = {}

    for func_name, cfg in BENCHMARK_FUNCTIONS.items():
        func = cfg['func']
        lb, ub = cfg['lb'], cfg['ub']
        print(f"\n{'='*60}")
        print(f"测试函数: {func_name} (dim={DIM}, lb={lb}, ub={ub})")
        print(f"{'='*60}")

        results[func_name] = {}

        for algo_name, algo in ALGORITHMS.items():
            print(f"\n  运行 {algo_name} ...", end=" ", flush=True)
            all_fits = []
            all_convergences = []

            for run in range(RUNS):
                best_fit, conv = run_single(algo, func, DIM, lb, ub)
                all_fits.append(best_fit)
                all_convergences.append(conv)

            all_fits = np.array(all_fits)

            # 统计指标
            best_val = float(np.min(all_fits))
            mean_val = float(np.mean(all_fits))
            std_val = float(np.std(all_fits))
            worst_val = float(np.max(all_fits))
            median_val = float(np.median(all_fits))

            # 平均收敛曲线
            mean_conv = np.mean(all_convergences, axis=0).tolist()

            results[func_name][algo_name] = {
                'best': best_val,
                'worst': worst_val,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'convergence': mean_conv,
                'all_fits': all_fits.tolist(),
            }

            print(f"完成 (best={best_val:.4e}, mean={mean_val:.4e}, std={std_val:.4e})")

    # 保存结果到JSON
    result_path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {result_path}")

    # 打印汇总表格
    print_summary_table(results)

    return results


def print_summary_table(results):
    """打印汇总统计表格"""
    for func_name in results:
        print(f"\n{'='*80}")
        print(f"函数: {func_name}")
        print(f"{'='*80}")
        print(f"{'算法':<10} {'最优值':<15} {'最差值':<15} {'均值':<15} {'中位数':<15} {'标准差':<15}")
        print('-' * 80)
        for algo_name in ['PSO', 'SSA', 'PO', 'CLSA-PO']:
            r = results[func_name][algo_name]
            print(f"{algo_name:<10} {r['best']:<15.4e} {r['worst']:<15.4e} {r['mean']:<15.4e} {r['median']:<15.4e} {r['std']:<15.4e}")


def plot_convergence(results):
    """生成收敛曲线对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    colors = {'PSO': '#1f77b4', 'SSA': '#ff7f0e', 'PO': '#2ca02c', 'CLSA-PO': '#d62728'}
    linestyles = {'PSO': '--', 'SSA': '-.', 'PO': ':', 'CLSA-PO': '-'}

    for func_name in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        for algo_name in ['PSO', 'SSA', 'PO', 'CLSA-PO']:
            conv = results[func_name][algo_name]['convergence']
            ax.plot(conv, label=algo_name, color=colors[algo_name],
                    linestyle=linestyles[algo_name], linewidth=1.5)

        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('适应度值', fontsize=12)
        ax.set_title(f'{func_name} 函数收敛曲线对比', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'convergence_{func_name}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"已保存: {path}")

    # 全部函数的合并图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, func_name in enumerate(results):
        ax = axes[idx]
        for algo_name in ['PSO', 'SSA', 'PO', 'CLSA-PO']:
            conv = results[func_name][algo_name]['convergence']
            ax.plot(conv, label=algo_name, color=colors[algo_name],
                    linestyle=linestyles[algo_name], linewidth=1.2)

        ax.set_title(f'{func_name}', fontsize=12)
        ax.set_xlabel('迭代次数', fontsize=9)
        ax.set_ylabel('适应度值', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # 隐藏多余的子图
    axes[5].set_visible(False)

    plt.suptitle('基准测试函数收敛曲线对比', fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'convergence_all.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {path}")


if __name__ == '__main__':
    print("开始运行对比实验...")
    print(f"参数: pop_size={POP_SIZE}, max_iter={MAX_ITER}, dim={DIM}, runs={RUNS}")
    print(f"算法: {list(ALGORITHMS.keys())}")
    print(f"函数: {list(BENCHMARK_FUNCTIONS.keys())}")

    results = run_experiment()
    plot_convergence(results)

    print("\n实验完成！")
