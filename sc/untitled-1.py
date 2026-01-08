#!/usr/bin/env python3
"""
实验脚本：比较有限通信和自由通信模式下的性能差异

该脚本运行两种通信模式的实验，收集总效用、运行时间、完成的任务数等性能指标，
并生成可视化对比图。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from gcaa.core.dta import optimal_control_dta
from gcaa.tools.disk import dump_json, mkdir
from gcaa.tools.timer import timing
from gcaa.tools.constants import SIMU_DIR


def run_experiment(na=5, nt=4, n_rounds=20, n_runs=10):
    """
    运行有限通信和自由通信模式的实验，收集性能指标
    
    参数：
    na: 智能体数量
    nt: 任务数量
    n_rounds: 模拟轮次
    n_runs: 重复运行次数
    
    返回：
    实验结果字典，包含两种通信模式下的性能指标
    """
    
    # 初始化结果存储
    results = {
        'limited_communication': {
            'total_utility': [],
            'run_time': [],
            'completed_tasks': [],
            'communication_counts': []
        },
        'free_communication': {
            'total_utility': [],
            'run_time': [],
            'completed_tasks': [],
            'communication_counts': []
        }
    }
    
    # 固定随机种子以确保可重复性
    np.random.seed(42)
    
    for run in range(n_runs):
        print(f"\n运行实验 {run+1}/{n_runs}")
        
        # 生成随机初始位置和速度
        map_width = 1.0
        pos_a = (0.1 + 0.8 * np.random.rand(na, 2)) * map_width
        pos_t = (0.1 + 0.8 * np.random.rand(nt, 2)) * map_width
        max_speed = 0.1
        v_a = (2 * np.random.rand(na, 2) - 1) * max_speed
        
        # 运行有限通信模式
        print("\n=== 有限通信模式 ===")
        start_time = time.perf_counter()
        result_limited = optimal_control_dta(
            na=na, nt=nt, n_rounds=n_rounds,
            limited_communication=True,
            pos_a=pos_a.copy(), pos_t=pos_t.copy(), v_a=v_a.copy(),
            sim_name=f"limited_run_{run}"
        )
        end_time = time.perf_counter()
        
        # 从结果中提取总效用和完成的任务数
        # 注意：这里需要修改optimal_control_dta函数以返回更多信息
        # 目前先记录运行时间
        results['limited_communication']['run_time'].append(end_time - start_time)
        
        # 运行自由通信模式
        print("\n=== 自由通信模式 ===")
        start_time = time.perf_counter()
        result_free = optimal_control_dta(
            na=na, nt=nt, n_rounds=n_rounds,
            limited_communication=False,
            pos_a=pos_a.copy(), pos_t=pos_t.copy(), v_a=v_a.copy(),
            sim_name=f"free_run_{run}"
        )
        end_time = time.perf_counter()
        
        results['free_communication']['run_time'].append(end_time - start_time)
    
    # 计算统计信息
    for mode in ['limited_communication', 'free_communication']:
        results[mode]['total_utility_mean'] = np.mean(results[mode]['total_utility'])
        results[mode]['total_utility_std'] = np.std(results[mode]['total_utility'])
        results[mode]['run_time_mean'] = np.mean(results[mode]['run_time'])
        results[mode]['run_time_std'] = np.std(results[mode]['run_time'])
        results[mode]['completed_tasks_mean'] = np.mean(results[mode]['completed_tasks'])
        results[mode]['completed_tasks_std'] = np.std(results[mode]['completed_tasks'])
    
    # 保存结果到JSON文件
    results_file = SIMU_DIR / f"experiment_results_{time.strftime('%Y%m%d-%H%M%S')}.json"
    dump_json(results, results_file, indent=2)
    print(f"\n实验结果已保存到: {results_file}")
    
    return results


def generate_visualizations(results):
    """
    生成有限通信和自由通信模式的性能对比可视化
    
    参数：
    results: 实验结果字典
    """
    
    # 创建可视化目录
    viz_dir = SIMU_DIR / "visualizations"
    mkdir(viz_dir)
    
    # 1. 总效用对比图
    plt.figure(figsize=(10, 6))
    modes = ['limited_communication', 'free_communication']
    mode_names = ['有限通信', '自由通信']
    
    # 由于目前optimal_control_dta函数未返回总效用，这里先注释掉
    # utilities = [results[mode]['total_utility'] for mode in modes]
    # plt.boxplot(utilities, labels=mode_names)
    # plt.title('有限通信 vs 自由通信 - 总效用对比')
    # plt.ylabel('总效用')
    # plt.grid(True, alpha=0.3)
    # plt.savefig(viz_dir / "utility_comparison.png", dpi=300, bbox_inches='tight')
    
    # 2. 运行时间对比图
    plt.figure(figsize=(10, 6))
    run_times = [results[mode]['run_time'] for mode in modes]
    plt.boxplot(run_times, labels=mode_names)
    plt.title('有限通信 vs 自由通信 - 运行时间对比')
    plt.ylabel('运行时间 (秒)')
    plt.grid(True, alpha=0.3)
    plt.savefig(viz_dir / "run_time_comparison.png", dpi=300, bbox_inches='tight')
    
    # 3. 任务完成情况对比图
    plt.figure(figsize=(10, 6))
    # 由于目前optimal_control_dta函数未返回完成的任务数，这里先注释掉
    # completed_tasks = [results[mode]['completed_tasks'] for mode in modes]
    # plt.boxplot(completed_tasks, labels=mode_names)
    # plt.title('有限通信 vs 自由通信 - 完成的任务数对比')
    # plt.ylabel('完成的任务数')
    # plt.grid(True, alpha=0.3)
    # plt.savefig(viz_dir / "completed_tasks_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"\n可视化结果已保存到: {viz_dir}")


if __name__ == "__main__":
    # 运行实验
    results = run_experiment(na=5, nt=4, n_rounds=20, n_runs=10)
    
    # 生成可视化
    generate_visualizations(results)
    
    # 打印实验结果摘要
    print("\n=== 实验结果摘要 ===")
    print(f"有限通信模式平均运行时间: {results['limited_communication']['run_time_mean']:.4f} 秒")
    print(f"自由通信模式平均运行时间: {results['free_communication']['run_time_mean']:.4f} 秒")
    
    # 注意：需要修改optimal_control_dta函数以返回更多信息
    # print(f"有限通信模式平均总效用: {results['limited_communication']['total_utility_mean']:.4f}")
    # print(f"自由通信模式平均总效用: {results['free_communication']['total_utility_mean']:.4f}")
    
    print("\n实验完成！")
