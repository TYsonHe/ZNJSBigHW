"""
阶段3：对比实验运行脚本
在8个基准测试函数上运行4种算法（PO、CLSA-PO、SSA、PSO）各30次，
输出统计表格（最优值、均值、标准差）、收敛曲线对比图和Wilcoxon秩和检验结果。
"""
import numpy as np
import json
import os
from datetime import datetime

from algorithms.pso import PSO
from algorithms.ssa import SSA
from algorithms.po import PO
from algorithms.clsa_po import CLSA_PO
from algorithms.utils import BENCHMARK_FUNCTIONS

# ========== 实验参数 ==========
POP_SIZE = 50
MAX_ITER = 500
DIM = 50
RUNS = 30

ALGORITHMS = {
    'PSO': PSO(pop_size=POP_SIZE, max_iter=MAX_ITER),
    'SSA': SSA(pop_size=POP_SIZE, max_iter=MAX_ITER),
    'PO': PO(pop_size=POP_SIZE, max_iter=MAX_ITER),
    'CLSA-PO': CLSA_PO(pop_size=POP_SIZE, max_iter=MAX_ITER),
}

OUTPUT_DIR = 'output/' + datetime.now().strftime('%Y%m%d_%H%M%S')


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


def wilcoxon_rank_sum_test(results):
    """Wilcoxon秩和检验：CLSA-PO vs 其他算法"""
    from scipy.stats import mannwhitneyu

    print(f"\n{'='*80}")
    print("Wilcoxon秩和检验结果（CLSA-PO vs 其他算法，α=0.05）")
    print(f"{'='*80}")
    print(f"{'函数':<12} {'对比算法':<10} {'p值':<15} {'符号':<6} {'结论'}")
    print('-' * 80)

    test_results = {}
    for func_name in results:
        test_results[func_name] = {}
        clsa_fits = np.array(results[func_name]['CLSA-PO']['all_fits'])

        for algo_name in ['PSO', 'SSA', 'PO']:
            algo_fits = np.array(results[func_name][algo_name]['all_fits'])

            # 当所有值完全相同时无法进行检验
            if np.all(clsa_fits == algo_fits):
                symbol = '≈'
                conclusion = '无显著差异'
                p_val = 1.0
            else:
                try:
                    stat, p_val = mannwhitneyu(clsa_fits, algo_fits, alternative='two-sided')
                    if p_val < 0.05:
                        clsa_mean = np.mean(clsa_fits)
                        algo_mean = np.mean(algo_fits)
                        if clsa_mean < algo_mean:
                            symbol = '+'
                            conclusion = 'CLSA-PO显著优于'
                        else:
                            symbol = '-'
                            conclusion = 'CLSA-PO显著劣于'
                    else:
                        symbol = '≈'
                        conclusion = '无显著差异'
                except ValueError:
                    symbol = '≈'
                    conclusion = '无法检验'
                    p_val = float('nan')

            test_results[func_name][algo_name] = {
                'p_value': float(p_val),
                'symbol': symbol,
                'conclusion': conclusion,
            }
            print(f"{func_name:<12} {algo_name:<10} {p_val:<15.4e} {symbol:<6} {conclusion} {algo_name}")

    # 保存检验结果
    test_path = os.path.join(OUTPUT_DIR, 'wilcoxon_test.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"\n检验结果已保存到 {test_path}")

    return test_results


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
    n_funcs = len(results)
    n_cols = 4
    n_rows = (n_funcs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
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
    for idx in range(n_funcs, len(axes)):
        axes[idx].set_visible(False)

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
    wilcoxon_rank_sum_test(results)
    plot_convergence(results)

    print("\n实验完成！")
