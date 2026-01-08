## 项目简介

群智感知任务分配拍卖系统（Group Crowd-sensing Auction Allocation, GCAA）是一个用于动态任务与智能体分配的拍卖算法框架。该系统主要应用于多智能体系统中，通过拍卖机制实现任务与智能体的高效匹配与动态分配。

## 功能特性

- **动态任务分配**：支持任务和智能体的动态加入与离开
- **拍卖机制**：基于拍卖理论实现高效的资源分配
- **通信限制支持**：可模拟智能体间通信受限的真实场景
- **仿真可视化**：提供任务分配过程的可视化模拟
- **多种算法**：内置贪心算法等多种分配策略
- **可扩展性**：支持自定义算法和仿真参数

## 项目结构

```
task-allocation-auctions/
├── gcaa/                 # 主要功能模块
│   ├── algorithms/       # 分配算法实现
│   ├── core/             # 核心功能组件
│   ├── simulations/      # 仿真结果文件
│   └── tests/            # 单元测试
├── exp-results/          # 实验结果
├── img/                  # 可视化图像
├── mov/                  # 动画演示
├── sc/                   # 脚本文件
├── scripts/              # 工具脚本
├── pyproject.toml        # 项目配置
├── pytest.ini            # 测试配置
└── requirements.txt      # 依赖列表
```

## 核心模块

### gcaa.algorithms
实现了多种任务分配算法，目前包含贪心算法。

### gcaa.core
- **allocation.py**：分配核心逻辑
- **control.py**：系统控制模块
- **dta.py**：动态任务分配实现
- **utility.py**：实用工具函数

### gcaa.tools
提供了多种辅助工具，包括数组处理、日期处理、绘图工具等。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
pytest -v
```

### 运行仿真

```python
# 导入必要的模块
from gcaa.core.dta import DynamicTaskAllocation
from gcaa.algorithms.greedy import GreedyAlgorithm

# 创建任务分配实例
dta = DynamicTaskAllocation(na=5, nt=5, algorithm=GreedyAlgorithm())

# 运行仿真
dta.run_simulation()

# 可视化结果
dta.visualize()
```

## 算法介绍

### 贪心算法
贪心算法是一种基于局部最优选择的启发式算法，在每一步选择当前看起来最优的解，以期望通过局部最优选择得到全局最优解。在任务分配场景中，贪心算法会优先将任务分配给当前最合适的智能体。

## 仿真结果

系统提供了丰富的仿真结果可视化，包括：

- 任务分配热力图
- 智能体运动轨迹
- 通信限制下的分配效果
- 不同参数配置的对比分析

## 应用场景

- **无人机协同任务**：多无人机任务分配与路径规划
- **传感器网络**：分布式传感器资源优化分配
- **机器人协作**：多机器人系统任务调度
- **交通管理**：智能交通系统中的车辆调度

## 开发与扩展

### 添加新算法

```python
from gcaa.algorithms import BaseAlgorithm

class NewAlgorithm(BaseAlgorithm):
    def allocate(self, agents, tasks):
        # 实现自定义分配逻辑
        pass
```

### 自定义仿真参数

```python
from gcaa.core.dta import DynamicTaskAllocation

# 配置仿真参数
params = {
    'na': 10,          # 智能体数量
    'nt': 10,          # 任务数量
    'communication_limit': 5,  # 通信限制
    'speed': 2.0       # 智能体速度
}

dta = DynamicTaskAllocation(**params)
```

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request


## 更新日志

### v1.0.0
- 初始版本发布
- 实现基本的动态任务分配功能
- 支持贪心算法
- 提供可视化工具

---

**备注**：本项目基于群智感知理论和拍卖机制，旨在为多智能体系统中的任务分配问题提供高效的解决方案。
        