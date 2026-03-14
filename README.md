# 太空垃圾高速捕获机械臂仿真系统

基于LSTM运动预测与MPC轨迹规划的太空碎片捕获仿真研究

## 项目结构

```
├── sim/                     # 仿真核心
│   ├── space_env.py         # 太空环境（微重力、碰撞检测）
│   ├── space_env_wave.py    # 多碎片波次模式
│   ├── robot_arm.py         # 6-DOF机械臂（DH参数、DLS逆运动学）
│   ├── debris.py            # 碎片动力学（四元数旋转、欧拉方程）
│   └── sensors.py           # 视觉传感器模拟
├── prediction/              # 运动预测
│   ├── lstm_predictor.py    # LSTM网络 + 物理基线预测器
│   ├── trajectory_data.py   # 训练数据生成
│   └── train.py             # 模型训练脚本
├── planning/                # 轨迹规划
│   ├── mpc_planner.py       # MPC规划器
│   └── grasp_strategy.py    # 自适应抓取策略
├── evaluation/              # 评估与报告
│   ├── metrics.py           # 性能指标
│   └── report_generator.py  # 图表生成
├── visualization/           # 可视化
│   ├── renderer.py          # 2D Pygame渲染
│   └── renderer_3d.py       # 3D OpenGL渲染
├── config/default_config.yaml
├── demo_auto.py             # 自动捕获演示（2D）
├── demo_auto_3d.py          # 自动捕获演示（3D）
├── demo_manual.py           # 手动游戏模式（2D）
├── demo_manual_3d.py        # 手动游戏模式（3D，10关）
├── run_experiments.py       # 批量对比实验
├── record_demo.py           # 录制演示视频
└── requirements.txt
```

## 快速开始

### 1. 环境安装

```bash
# Python 3.10+ 推荐
pip install -r requirements.txt
```

依赖：numpy, scipy, torch, pygame, PyOpenGL, matplotlib, pyyaml, opencv-python

### 2. 训练LSTM模型

```bash
python prediction/train.py
```

生成 `models/lstm_predictor.pth` 和 `models/normalization.npz`，约需2-3分钟（GPU）。

### 3. 运行演示

```bash
python demo_auto_3d.py       # 3D自动捕获（推荐）
python demo_manual_3d.py     # 3D手动游戏（WASD+QE控制）
python demo_auto.py          # 2D自动捕获
python demo_manual.py        # 2D手动游戏
```

### 4. 跑批量实验

```bash
python run_experiments.py
```

运行三组对比实验（LSTM预测 / 物理预测 / 纯反应式），生成 `reports/` 下的图表和 `experiment_log.txt`。

### 5. 录制视频

```bash
python record_demo.py
```

## 实验结果

| 方法 | 成功率 | 平均捕获时间 | 平均最小距离 |
|------|--------|-------------|-------------|
| LSTM预测 | 80.0% | 8.46s | 0.298m |
| 物理预测 | 73.3% | 8.83s | 0.343m |
| 纯反应式 | 75.6% | 9.81s | 0.462m |

LSTM预测在高速段（v≥1.6 m/s）优势显著，成功率比纯反应式高约22个百分点。

## 论文

- `paper.md` — 完整版论文
- `paper_brief.md` — 简略版
- `paper_ch1to4.tex` — 前四章LaTeX源码（XeLaTeX编译）
