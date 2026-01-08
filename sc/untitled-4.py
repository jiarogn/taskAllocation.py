import numpy as np
import matplotlib.pyplot as plt
import os
import time
from untitled_3 import optimal_control_dta

# 实验参数设置
na = 5  # 智能体数
nt = 4  # 任务数
n_rounds = 20  # 迭代轮数
n_seeds = 10  # 随机种子数
map_width = 10  # 地图宽度
comm_distance = 3.0  # 通信距离阈值

# 创建结果保存目录
save_dir = "communication_comparison_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 存储所有实验结果
results = {
    "free": [],
    "limited": []
}

# 运行实验
for seed in range(n_seeds):
    print(f"\nRunning seed {seed+1}/{n_seeds}")
    
    # 自由通信模式
    print("  Free communication mode...")
    free_result = optimal_control_dta(
        na=na, nt=nt, n_rounds=n_rounds,
        map_width=map_width, CommLimit=False,
        seed=seed, animate=False, verbose=False
    )
    results["free"].append(free_result)
    
    # 有限通信模式
    print("  Limited communication mode...")
    limited_result = optimal_control_dta(
        na=na, nt=nt, n_rounds=n_rounds,
        map_width=map_width, CommLimit=True, comm_distance=comm_distance,
        seed=seed, animate=False, verbose=False
    )
    results["limited"].append(limited_result)

# 分析结果
def analyze_results(results, mode):
    total_utilities = [r["total_utility"] for r in results]
    total_completed_tasks = [r["total_completed_tasks"] for r in results]
    total_comm_overhead = [r["total_communication_overhead"] for r in results]
    
    return {
        "mean_utility": np.mean(total_utilities),
        "std_utility": np.std(total_utilities),
        "mean_completed_tasks": np.mean(total_completed_tasks),
        "std_completed_tasks": np.std(total_completed_tasks),
        "mean_comm_overhead": np.mean(total_comm_overhead),
        "std_comm_overhead": np.std(total_comm_overhead)
    }

free_analysis = analyze_results(results["free"], "free")
limited_analysis = analyze_results(results["limited"], "limited")

# 打印分析结果
print("\n" + "="*50)
print("Experimental Results Analysis")
print("="*50)
print(f"Free Communication:")
print(f"  Mean Total Utility: {free_analysis['mean_utility']:.2f} ± {free_analysis['std_utility']:.2f}")
print(f"  Mean Completed Tasks: {free_analysis['mean_completed_tasks']:.2f} ± {free_analysis['std_completed_tasks']:.2f}")
print(f"  Mean Communication Overhead: {free_analysis['mean_comm_overhead']:.2f} ± {free_analysis['std_comm_overhead']:.2f}")

print(f"\nLimited Communication:")
print(f"  Mean Total Utility: {limited_analysis['mean_utility']:.2f} ± {limited_analysis['std_utility']:.2f}")
print(f"  Mean Completed Tasks: {limited_analysis['mean_completed_tasks']:.2f} ± {limited_analysis['std_completed_tasks']:.2f}")
print(f"  Mean Communication Overhead: {limited_analysis['mean_comm_overhead']:.2f} ± {limited_analysis['std_comm_overhead']:.2f}")

print(f"\nPerformance Difference:")
print(f"  Utility Degradation: {(limited_analysis['mean_utility'] - free_analysis['mean_utility'])/free_analysis['mean_utility']*100:.2f}%")
print(f"  Communication Overhead Reduction: {(free_analysis['mean_comm_overhead'] - limited_analysis['mean_comm_overhead'])/free_analysis['mean_comm_overhead']*100:.2f}%")

# 可视化两种通信模式的对比
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. 总效用对比
axs[0, 0].bar(0, free_analysis['mean_utility'], yerr=free_analysis['std_utility'], capsize=5, label='Free Communication')
axs[0, 0].bar(1, limited_analysis['mean_utility'], yerr=limited_analysis['std_utility'], capsize=5, label='Limited Communication')
axs[0, 0].set_xticks([0, 1])
axs[0, 0].set_xticklabels(['Free', 'Limited'])
axs[0, 0].set_title('Total Utility Comparison')
axs[0, 0].set_ylabel('Utility')
axs[0, 0].legend()

# 2. 完成任务数对比
axs[0, 1].bar(0, free_analysis['mean_completed_tasks'], yerr=free_analysis['std_completed_tasks'], capsize=5, label='Free Communication')
axs[0, 1].bar(1, limited_analysis['mean_completed_tasks'], yerr=limited_analysis['std_completed_tasks'], capsize=5, label='Limited Communication')
axs[0, 1].set_xticks([0, 1])
axs[0, 1].set_xticklabels(['Free', 'Limited'])
axs[0, 1].set_title('Completed Tasks Comparison')
axs[0, 1].set_ylabel('Number of Tasks')
axs[0, 1].legend()

# 3. 通信开销对比
axs[1, 0].bar(0, free_analysis['mean_comm_overhead'], yerr=free_analysis['std_comm_overhead'], capsize=5, label='Free Communication')
axs[1, 0].bar(1, limited_analysis['mean_comm_overhead'], yerr=limited_analysis['std_comm_overhead'], capsize=5, label='Limited Communication')
axs[1, 0].set_xticks([0, 1])
axs[1, 0].set_xticklabels(['Free', 'Limited'])
axs[1, 0].set_title('Communication Overhead Comparison')
axs[1, 0].set_ylabel('Overhead')
axs[1, 0].legend()

# 4. 效用随轮次变化（使用第一个种子的数据）
rounds = np.arange(n_rounds)
axs[1, 1].plot(rounds, results["free"][0]["U_tot"], label='Free Communication')
axs[1, 1].plot(rounds, results["limited"][0]["U_tot"], label='Limited Communication')
axs[1, 1].set_title('Utility Evolution Over Rounds')
axs[1, 1].set_xlabel('Round')
axs[1, 1].set_ylabel('Total Utility')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "communication_comparison_summary.png"))
plt.show()

# 可视化特定种子的轨迹（使用第一个种子的数据）
seed_to_visualize = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 自由通信模式轨迹
ax1.set_title(f"Free Communication (Seed {seed_to_visualize+1})")
ax1.set_xlim(0, map_width)
ax1.set_ylim(0, map_width)
ax1.set_aspect('equal')

# 绘制智能体轨迹
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
while len(colors) < na:
    colors.extend(colors)
colors = colors[:na]

for i in range(na):
    # 提取轨迹
    agent_path = []
    for round_idx in range(n_rounds):
        pos = results["free"][seed_to_visualize]["historical_path"][round_idx][i]
        agent_path.append(pos)
    agent_path = np.array(agent_path)
    ax1.plot(agent_path[:, 0], agent_path[:, 1], '-o', color=colors[i], label=f'Agent {i+1}')

# 绘制任务位置（初始位置）
initial_tasks = results["free"][seed_to_visualize]["X_full_simu"][0][:, na:, 0]  # 假设任务位置存储在X的后半部分
ax1.plot(initial_tasks[0, :], initial_tasks[1, :], 'rs', markersize=8, label='Tasks')

ax1.legend()
ax1.grid(True, alpha=0.3)

# 有限通信模式轨迹
ax2.set_title(f"Limited Communication (Seed {seed_to_visualize+1})")
ax2.set_xlim(0, map_width)
ax2.set_ylim(0, map_width)
ax2.set_aspect('equal')

# 绘制智能体轨迹
for i in range(na):
    # 提取轨迹
    agent_path = []
    for round_idx in range(n_rounds):
        pos = results["limited"][seed_to_visualize]["historical_path"][round_idx][i]
        agent_path.append(pos)
    agent_path = np.array(agent_path)
    ax2.plot(agent_path[:, 0], agent_path[:, 1], '-o', color=colors[i], label=f'Agent {i+1}')

# 绘制任务位置（初始位置）
initial_tasks = results["limited"][seed_to_visualize]["X_full_simu"][0][:, na:, 0]  # 假设任务位置存储在X的后半部分
ax2.plot(initial_tasks[0, :], initial_tasks[1, :], 'rs', markersize=8, label='Tasks')

ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"trajectory_comparison_seed_{seed_to_visualize+1}.png"))
plt.show()

print(f"\n实验完成！结果已保存到 {save_dir} 目录。")
