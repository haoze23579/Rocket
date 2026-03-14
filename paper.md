# 太空垃圾高速捕获机械臂仿真系统——基于视觉预测的轨迹规划研究

## Abstract

This paper presents an engineering-oriented simulation workflow for space-debris capture and reports reproducible quantitative results from the latest script version.

The experiment uses batch execution with fixed seeds (6 episodes, 600 s simulated time per episode, 100x speedup), and records both CSV metrics and replayable videos.

From the 2026-03-03 run, the system achieves a destroy rate of 0.000, mean captures of 2.167 (range 1-4), and mean `delta_v_total` of 9945.333 (std 1004.144). Additional derived metrics (capture efficiency and capture frequency) are reported for cross-episode comparison.

This revision focuses on reproducibility, metric consistency, and evidence traceability (commands -> data -> videos), providing a practical baseline for larger-scale ablation and policy comparison in future work.

**Keywords:** space debris, robotic capture, simulation, reproducibility, quantitative evaluation, trajectory planning
---

## 1 引言

### 1.1 研究背景

截至目前，地球轨道上已有超过36,000个大于10cm的空间碎片被跟踪编目，而小于1cm的碎片数量估计超过1.3亿个。这些高速运动的空间碎片对在轨航天器构成严重威胁——即使是毫米级碎片，在轨道速度（约7.8km/s）下也具有巨大的动能。空间碎片主动清除（Active Debris Removal, ADR）已成为国际航天界的重要研究方向。

欧洲航天局（ESA）的ClearSpace-1任务计划于2026年发射，旨在验证空间碎片捕获技术。日本宇宙航空研究开发机构（JAXA）的ELSA-d实验已于2021年成功演示了磁性对接捕获技术。这些任务表明，机械臂抓取是空间碎片清除的核心技术路径之一。

### 1.2 技术挑战

太空垃圾捕获面临以下关键技术挑战：

1. **高速相对运动**：即使在交会接近后，目标碎片仍存在显著的相对运动（可达数m/s），对机械臂的响应速度提出极高要求
2. **目标非合作性**：碎片通常处于无控翻滚状态，三轴旋转叠加使得姿态变化复杂且难以预测
3. **形状不规则性**：废弃卫星、火箭残骸、碎片云等目标形状各异，抓取点选择困难
4. **微重力环境**：微重力条件下的动力学特性与地面环境显著不同
5. **通信延迟**：地面遥操作存在显著时延，要求系统具备自主决策能力

### 1.3 研究创新点

本研究的核心创新在于：采用深度学习（LSTM网络）进行目标运动预测，结合模型预测控制（MPC）实现"预判式"抓取轨迹规划。与传统的反应式控制策略不同，本方法通过预测目标未来0.5-2秒的运动轨迹，提前规划机械臂的拦截路径，从而有效应对高速运动目标的捕获挑战。

### 1.4 论文结构

本文第2节介绍系统总体架构设计；第3节详细阐述各核心算法；第4节描述仿真实验设计与实施；第5节分析实验结果；第6节总结全文并展望未来工作。

---

## 2 系统总体架构

### 2.1 系统组成

本仿真系统由六个核心模块组成，如图1所示：

```
┌─────────────────────────────────────────────────────────────┐
│                    太空环境仿真模块                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ 碎片运动  │  │ 微重力   │  │ 多形状   │                  │
│  │ 动力学    │  │ 环境     │  │ 碎片模型  │                  │
│  └────┬─────┘  └──────────┘  └──────────┘                  │
│       │                                                      │
│  ┌────▼─────────────────┐                                   │
│  │   视觉感知模块        │                                   │
│  │  (模拟传感器+噪声)    │                                   │
│  └────┬─────────────────┘                                   │
│       │ 观测序列                                             │
│  ┌────▼─────────────────┐    ┌──────────────────┐          │
│  │   运动预测模块        │───▶│  轨迹规划模块     │          │
│  │  (LSTM / 物理基线)    │    │  (MPC优化)        │          │
│  └──────────────────────┘    └────┬─────────────┘          │
│                                    │ 目标轨迹                │
│  ┌─────────────────────────────────▼────────────┐          │
│  │          抓取执行模块                          │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │          │
│  │  │ 6-DOF    │  │ 逆运动学  │  │ 末端执行器│   │          │
│  │  │ 机械臂   │  │ (DLS-IK) │  │ 策略选择  │   │          │
│  │  └──────────┘  └──────────┘  └──────────┘   │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │          性能评估模块                          │          │
│  │  成功率 | 响应时间 | 相对速度 | 预测误差       │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 变量设计

**自变量（实验控制参数）：**
- 目标线速度：0.1 ~ 6.0 m/s
- 目标旋转角速度：0 ~ 60°/s（0 ~ 1.05 rad/s）
- 目标形状：废弃卫星、火箭残骸、不规则碎片

**因变量（性能指标）：**
- 抓取成功率
- 抓取响应时间（从检测到抓取成功的时间）
- 末端执行器与目标的相对速度差
- 最小接近距离

**控制变量：**
- 机械臂自由度：6-DOF
- 关节最大角速度：3.14 rad/s
- 视觉系统帧率：30 Hz
- 传感器噪声标准差：0.01 m

---

## 3 核心算法

### 3.1 太空碎片运动模型

#### 3.1.1 平移运动

在微重力环境下，碎片不受外力作用，做匀速直线运动：

$$\mathbf{p}(t+\Delta t) = \mathbf{p}(t) + \mathbf{v} \cdot \Delta t$$

其中 $\mathbf{p} \in \mathbb{R}^3$ 为位置向量，$\mathbf{v} \in \mathbb{R}^3$ 为速度向量。

#### 3.1.2 旋转运动

碎片的旋转运动由欧拉方程描述。对于刚体在无外力矩条件下：

$$\mathbf{I} \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega}) = \mathbf{0}$$

其中 $\mathbf{I}$ 为惯性张量矩阵，$\boldsymbol{\omega}$ 为角速度向量。

惯性张量按长方体近似计算：

$$\mathbf{I} = \text{diag}\left(\frac{m(s_y^2+s_z^2)}{12}, \frac{m(s_x^2+s_z^2)}{12}, \frac{m(s_x^2+s_y^2)}{12}\right)$$

角速度更新：

$$\boldsymbol{\alpha} = \mathbf{I}^{-1}(-\boldsymbol{\omega} \times \mathbf{I}\boldsymbol{\omega})$$
$$\boldsymbol{\omega}(t+\Delta t) = \boldsymbol{\omega}(t) + \boldsymbol{\alpha} \cdot \Delta t$$

**注意**：对于非对称惯性张量（$I_x \neq I_y \neq I_z$），即使无外力矩，角速度方向也会随时间变化，这正是太空碎片复杂翻滚运动的物理根源。

#### 3.1.3 姿态表示与更新

采用四元数 $\mathbf{q} = (q_w, q_x, q_y, q_z)$ 表示姿态，避免万向锁问题。四元数积分更新：

$$\Delta\mathbf{q} = \left(\cos\frac{\theta}{2},\ \hat{\boldsymbol{\omega}}\sin\frac{\theta}{2}\right)$$

其中 $\theta = |\boldsymbol{\omega}|\Delta t$，$\hat{\boldsymbol{\omega}} = \boldsymbol{\omega}/|\boldsymbol{\omega}|$。

$$\mathbf{q}(t+\Delta t) = \mathbf{q}(t) \otimes \Delta\mathbf{q}$$

四元数乘法定义为：

$$\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{pmatrix} w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2 \\ w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2 \\ w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2 \\ w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2 \end{pmatrix}$$

### 3.2 6-DOF机械臂运动学

#### 3.2.1 DH参数与正运动学

采用标准Denavit-Hartenberg（DH）参数描述机械臂构型。每个关节的齐次变换矩阵为：

$$\mathbf{T}_i = \begin{pmatrix} c\theta_i & -s\theta_i c\alpha_i & s\theta_i s\alpha_i & a_i c\theta_i \\ s\theta_i & c\theta_i c\alpha_i & -c\theta_i s\alpha_i & a_i s\theta_i \\ 0 & s\alpha_i & c\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

其中 $a_i, \alpha_i, d_i, \theta_i$ 分别为连杆长度、连杆扭转角、连杆偏距和关节角。

末端执行器位姿通过连乘得到：

$$\mathbf{T}_{0}^{6} = \prod_{i=1}^{6} \mathbf{T}_{i-1}^{i}$$

本系统机械臂DH参数如表1所示：

| 关节 | $a_i$ (m) | $\alpha_i$ (rad) | $d_i$ (m) | 功能 |
|------|-----------|-------------------|-----------|------|
| 1 | 0 | π/2 | 1.0 | 基座旋转 |
| 2 | 1.5 | 0 | 0 | 肩部俯仰 |
| 3 | 1.2 | 0 | 0 | 肘部弯曲 |
| 4 | 0 | π/2 | 0 | 腕部旋转 |
| 5 | 0 | -π/2 | 0.8 | 腕部俯仰 |
| 6 | 0 | 0 | 0.8 | 末端旋转 |

总臂展：5.3m

#### 3.2.2 雅可比矩阵

雅可比矩阵 $\mathbf{J} \in \mathbb{R}^{3 \times 6}$ 建立末端速度与关节速度的映射关系：

$$\dot{\mathbf{x}} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

本系统采用数值微分法计算雅可比矩阵：

$$J_{ij} = \frac{f_i(\mathbf{q} + \epsilon \mathbf{e}_j) - f_i(\mathbf{q})}{\epsilon}$$

其中 $\epsilon = 10^{-6}$，$\mathbf{e}_j$ 为第 $j$ 个关节方向的单位向量，$f_i$ 为正运动学函数的第 $i$ 个分量。

#### 3.2.3 阻尼最小二乘逆运动学（DLS-IK）

传统雅可比伪逆法在奇异位形附近会产生极大的关节速度。本系统采用阻尼最小二乘（Damped Least Squares, DLS）方法：

$$\Delta\mathbf{q} = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T + \lambda^2\mathbf{I})^{-1} \Delta\mathbf{x}$$

其中 $\lambda = 0.01$ 为阻尼系数，$\Delta\mathbf{x} = \mathbf{x}_{target} - \mathbf{x}_{current}$ 为末端位置误差。

关节速度限制：

$$\Delta\mathbf{q}_{limited} = \begin{cases} \Delta\mathbf{q} & \text{if } \max|\Delta q_i| \leq \dot{q}_{max}\Delta t \\ \Delta\mathbf{q} \cdot \frac{\dot{q}_{max}\Delta t}{\max|\Delta q_i|} & \text{otherwise} \end{cases}$$

#### 3.2.4 可操作度指标

采用Yoshikawa可操作度指标评估机械臂的运动灵活性：

$$w = \sqrt{\det(\mathbf{J}\mathbf{J}^T)}$$

当 $w \to 0$ 时，机械臂接近奇异位形，运动能力退化。该指标用于MPC代价函数中的奇异位形惩罚项。


### 3.3 视觉感知模块

#### 3.3.1 传感器模型

本系统模拟RGB-D深度传感器，以固定帧率采集目标碎片的状态信息。传感器参数如表2所示：

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 采集帧率 | $f_s$ | 30 Hz | 每秒采集次数 |
| 测量噪声基准 | $\sigma_0$ | 0.01 m | 近距离噪声标准差 |
| 最大探测距离 | $R_{max}$ | 15.0 m | 超出则无法观测 |
| 视场角 | FOV | 90° | 传感器视场范围 |
| 观测缓冲区 | $N_{buf}$ | 300帧 | 约10秒历史数据 |

#### 3.3.2 距离相关噪声模型

真实传感器的测量精度随距离增大而降低。本系统采用线性距离相关噪声模型：

$\sigma(d) = \sigma_0 \left(1 + \frac{d}{R_{max}}\right)$

其中 $d$ 为目标距离，$\sigma_0$ 为基准噪声标准差。各状态量的噪声分别为：

- 位置噪声：$\mathbf{n}_p \sim \mathcal{N}(\mathbf{0}, \sigma(d)^2 \mathbf{I}_3)$
- 速度噪声：$\mathbf{n}_v \sim \mathcal{N}(\mathbf{0}, (0.5\sigma(d))^2 \mathbf{I}_3)$
- 姿态噪声：$\mathbf{n}_q \sim \mathcal{N}(\mathbf{0}, (0.1\sigma(d))^2 \mathbf{I}_4)$
- 角速度噪声：$\mathbf{n}_\omega \sim \mathcal{N}(\mathbf{0}, (0.3\sigma(d))^2 \mathbf{I}_3)$

观测模型为：

$\hat{\mathbf{p}} = \mathbf{p}_{true} + \mathbf{n}_p, \quad \hat{\mathbf{v}} = \mathbf{v}_{true} + \mathbf{n}_v$

姿态四元数在加噪后需重新归一化：

$\hat{\mathbf{q}} = \frac{\mathbf{q}_{true} + \mathbf{n}_q}{\|\mathbf{q}_{true} + \mathbf{n}_q\|}$

#### 3.3.3 观测置信度

传感器输出观测置信度，反映当前测量的可靠程度：

$c = \text{clip}\left(1 - \frac{d}{R_{max}} + \epsilon, \ 0, \ 1\right), \quad \epsilon \sim \mathcal{N}(0, 0.05)$

距离越近置信度越高，该值将传递给后续的运动预测和轨迹规划模块。

#### 3.3.4 帧率限制与观测缓冲

传感器以固定帧率 $f_s = 30$ Hz 工作，仅在采样时刻输出观测数据。两次采样之间系统使用上一帧数据。观测缓冲区保存最近 $N_{buf} = 300$ 帧数据（约10秒），为LSTM预测模块提供输入序列。

观测序列的提取格式为：

$\mathbf{X} = [\mathbf{x}_{t-L+1}, \mathbf{x}_{t-L+2}, \ldots, \mathbf{x}_t] \in \mathbb{R}^{L \times 10}$

其中 $L = 30$ 为序列长度，每帧 $\mathbf{x}_i = [p_x, p_y, p_z, v_x, v_y, v_z, q_w, q_x, q_y, q_z]^T$。



### 3.4 LSTM运动预测模块

#### 3.4.1 问题定义

运动预测的核心任务是：给定目标碎片过去 $L$ 个时刻的观测序列，预测未来 $H$ 个时刻的状态轨迹。形式化表示为：

$\hat{\mathbf{Y}} = f_{LSTM}(\mathbf{X}), \quad \mathbf{X} \in \mathbb{R}^{L \times 10}, \quad \hat{\mathbf{Y}} \in \mathbb{R}^{H \times 10}$

其中 $L = 30$（输入序列长度），$H = 50$（预测时域，对应0.5秒@100Hz）。

#### 3.4.2 网络架构

采用两层堆叠LSTM网络，架构如下：

```
输入层: (batch, 30, 10)
    ↓
LSTM Layer 1: hidden_size=128, dropout=0.1
    ↓
LSTM Layer 2: hidden_size=128
    ↓ (取最后时刻隐状态)
    ├──→ 轨迹预测分支: FC(128→64) → ReLU → FC(64→500) → reshape(50, 10)
    └──→ 置信度分支:   FC(128→32) → ReLU → FC(32→50) → Sigmoid
```

网络参数统计：

| 层 | 参数量 |
|----|--------|
| LSTM Layer 1 | $4 \times (10 \times 128 + 128 \times 128 + 128) = 71,168$ |
| LSTM Layer 2 | $4 \times (128 \times 128 + 128 \times 128 + 128) = 131,584$ |
| 轨迹预测FC | $128 \times 64 + 64 + 64 \times 500 + 500 = 40,756$ |
| 置信度FC | $128 \times 32 + 32 + 32 \times 50 + 50 = 5,778$ |
| **总计** | **约249K** |

#### 3.4.3 输入输出格式

**输入**：经归一化处理的观测序列

$\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$

其中 $\boldsymbol{\mu}, \boldsymbol{\sigma} \in \mathbb{R}^{10}$ 为训练集上计算的均值和标准差向量，$\boldsymbol{\sigma}$ 下界截断为 $10^{-6}$ 以避免除零。

**输出**：
1. 预测轨迹 $\hat{\mathbf{Y}} \in \mathbb{R}^{50 \times 10}$：经反归一化还原为物理量
2. 置信度序列 $\mathbf{c} \in [0,1]^{50}$：通过Sigmoid激活，表示各预测时刻的可信程度

#### 3.4.4 训练数据生成

训练数据通过仿真环境自动生成，覆盖不同运动参数组合：

- 线速度：$v \in [0.1, 6.0]$ m/s，均匀采样
- 角速度：$\omega \in [0, 1.05]$ rad/s，均匀采样
- 碎片形状：卫星、火箭残骸、不规则碎片随机选取
- 初始方向：随机生成

每条轨迹仿真30秒，以随机采样方式从轨迹中提取训练样本（输入30帧 + 标签50帧），每条轨迹最多采样20个样本。

#### 3.4.5 物理基线预测器

作为对比基线，实现了基于物理模型的线性外推预测器：

$\hat{\mathbf{p}}(t+\Delta t) = \mathbf{p}(t) + \bar{\mathbf{v}} \cdot \Delta t$

其中 $\bar{\mathbf{v}}$ 为最近10帧的平均速度。基线预测器的置信度随时间指数衰减：

$c(t) = e^{-2t}$

该设计反映了线性外推在长时域预测中精度快速下降的物理事实。



### 3.5 MPC轨迹规划模块

#### 3.5.1 规划框架

模型预测控制（MPC）以预测的目标轨迹为参考，在线优化机械臂末端的运动轨迹。MPC参数配置如表3所示：

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 预测时域 | $N_p$ | 20步 | 前瞻规划步数 |
| 控制时域 | $N_c$ | 10步 | 优化控制步数 |
| 位置跟踪权重 | $w_{pos}$ | 10.0 | 末端位置误差权重 |
| 速度匹配权重 | $w_{vel}$ | 5.0 | 相对速度匹配权重 |
| 力矩平滑权重 | $w_{\tau}$ | 0.1 | 关节角度变化量惩罚 |
| 奇异位形惩罚 | $w_{sing}$ | 1.0 | 接近奇异位形的惩罚 |

#### 3.5.2 拦截点选择

拦截点选择是MPC规划的关键步骤。系统在预测轨迹上搜索最优拦截点，综合评分函数为：

$S(i) = 2.0 \cdot R(i) + 3.0 \cdot c(i) + 1.0 \cdot T(i)$

其中各分项定义为：

**可达性评分**：

$R(i) = \begin{cases} 1 - \frac{\|\mathbf{p}_i - \mathbf{p}_{tip}\|}{L_{arm}} & \text{if } \|\mathbf{p}_i - \mathbf{p}_{tip}\| \leq 0.95 L_{arm} \\ 0 & \text{otherwise} \end{cases}$

其中 $L_{arm} = 5.3$ m 为机械臂总臂展。

**置信度评分**：$c(i)$ 直接使用预测模块输出的置信度。

**时间评分**：

$T(i) = e^{-0.15i}$

在前10步预测范围内选择 $S(i)$ 最大的点作为拦截目标。

#### 3.5.3 平滑轨迹生成

确定拦截点后，采用三次Hermite插值生成平滑的末端轨迹，确保位置和速度的连续性：

$\mathbf{p}(t) = h_{00}(t)\mathbf{p}_0 + h_{10}(t)T\dot{\mathbf{p}}_0 + h_{01}(t)\mathbf{p}_1 + h_{11}(t)T\dot{\mathbf{p}}_1$

其中Hermite基函数为：

$h_{00}(t) = 2t^3 - 3t^2 + 1$
$h_{10}(t) = t^3 - 2t^2 + t$
$h_{01}(t) = -2t^3 + 3t^2$
$h_{11}(t) = t^3 - t^2$

边界条件：$\mathbf{p}_0$ 为当前末端位置，$\dot{\mathbf{p}}_0$ 为当前末端速度，$\mathbf{p}_1$ 为拦截点位置，$\dot{\mathbf{p}}_1 = \mathbf{0}$（抓取时刻相对速度最小化）。

#### 3.5.4 MPC代价函数

完整MPC优化问题的代价函数为：

$J = \sum_{k=0}^{N_c-1} \left[ w_{pos} \cdot c_k \cdot \|\mathbf{p}_{tip}^k - \mathbf{p}_{pred}^k\|^2 + w_{\tau} \cdot \|\Delta\mathbf{q}^k\|^2 + w_{sing} \cdot P_{sing}(\mathbf{q}^k) \right]$

其中：
- $\mathbf{p}_{tip}^k = FK(\mathbf{q}^k)$ 为第 $k$ 步的末端位置（通过正运动学计算）
- $\mathbf{p}_{pred}^k$ 为预测的目标位置
- $c_k$ 为预测置信度，对位置误差加权
- $\Delta\mathbf{q}^k$ 为关节角度增量
- $P_{sing}$ 为奇异位形惩罚项

奇异位形惩罚定义为：

$P_{sing}(\mathbf{q}) = \begin{cases} \frac{1}{\sigma_{min}(\mathbf{J}) + 10^{-6}} & \text{if } \sigma_{min}(\mathbf{J}) < 0.01 \\ 0 & \text{otherwise} \end{cases}$

其中 $\sigma_{min}(\mathbf{J})$ 为雅可比矩阵的最小奇异值。

#### 3.5.5 优化求解

采用L-BFGS-B算法求解上述约束优化问题。决策变量为控制时域内的关节角度增量序列 $\{\Delta\mathbf{q}^0, \Delta\mathbf{q}^1, \ldots, \Delta\mathbf{q}^{N_c-1}\}$，共 $N_c \times 6 = 60$ 个变量。

约束条件为关节速度限制：

$|\Delta q_i^k| \leq \dot{q}_{max} \cdot \Delta t = 3.14 \times 0.01 = 0.0314 \ \text{rad}$

采用热启动策略：将上一步的最优解左移一步作为当前步的初始猜测，加速收敛。最大迭代次数设为50次，函数容差 $10^{-6}$。



### 3.6 抓取策略模块

#### 3.6.1 末端执行器选择

系统模拟三种末端执行器，根据碎片特性自动选择最优方案：

| 执行器类型 | 最大抓取尺寸 | 闭合时间 | 保持力 | 速度容差 | 适用目标 |
|-----------|-------------|---------|--------|---------|---------|
| 三指夹爪 | 1.5 m | 0.3 s | 50 N | 0.3 m/s | 小型碎片 |
| 柔性包络爪 | 3.0 m | 0.5 s | 30 N | 0.5 m/s | 废弃卫星、不规则体 |
| 电磁吸附器 | 5.0 m | 0.1 s | 100 N | 0.8 m/s | 火箭残骸（金属） |

选择逻辑基于碎片形状与执行器的优先级映射：

- 废弃卫星 → 柔性包络爪 > 电磁吸附器 > 三指夹爪
- 火箭残骸 → 电磁吸附器 > 柔性包络爪
- 不规则碎片 → 三指夹爪 > 柔性包络爪

系统按优先级顺序检查执行器是否可用且尺寸匹配，选择第一个满足条件的执行器。

#### 3.6.2 抓取时机决策

抓取时机的判断综合考虑距离、相对速度和置信度三个条件：

$$\text{Grasp} = \begin{cases} \text{True} & \text{if } d < d_{th} \wedge v_{rel} < v_{tol} \wedge c > 0.3 \\ \text{False} & \text{otherwise} \end{cases}$$

其中：
- $d_{th} = 0.15$ m 为距离阈值
- $v_{tol}$ 为所选执行器的速度容差
- $c$ 为当前观测置信度
- $v_{rel} = \|\dot{\mathbf{p}}_{tip} - \dot{\mathbf{p}}_{debris}\|$ 为末端与碎片的相对速度

当距离在 $[d_{th}, 2d_{th}]$ 范围内时，系统进入"接近"状态，开始评估抓取条件但不执行。

#### 3.6.3 接近方向计算

为避免正面碰撞，系统计算最优接近方向。设碎片运动方向为 $\hat{\mathbf{v}}$，碎片到机械臂方向为 $\hat{\mathbf{d}}$，则：

$\mathbf{s} = \hat{\mathbf{v}} \times \hat{\mathbf{d}}$

若 $\|\mathbf{s}\| > 0.1$（非共线），接近方向为追赶与侧面的加权组合：

$\hat{\mathbf{a}} = \text{normalize}(0.7\hat{\mathbf{v}} + 0.3\hat{\mathbf{s}})$

否则从碎片运动方向的正后方追赶：$\hat{\mathbf{a}} = \hat{\mathbf{v}}$。

这种策略使机械臂从碎片运动方向的侧后方接近，减小抓取时刻的相对速度，提高抓取成功率。

---

## 4 实验设置

### 4.1 目标与脚本版本

本次修订采用文件夹内的工程化脚本流程，主要包括：`orbit_simulator_experiment.py`（批量运行与 CSV 导出）、`orbit_simulator.py`（单次基线运行与视频录制）以及 `run_experiments_and_record.ps1`（一键化流程脚本）。实验日期为 2026-03-03，结果数据文件为 `experiment_results/orbit_experiment_results_20260303.csv`。

### 4.2 配置参数

批量实验配置如下：回合数为 6，每回合仿真时长为 600 s，仿真加速倍数为 100x，随机种子范围为 20260303 至 20260308，自动驾驶模式开启。视频录制配置如下：基线视频命令为 `orbit_simulator.py --sim-seconds 420 --sim-speedup 120 --autopilot --record`，实验视频命令为 `orbit_simulator_experiment.py --sim-seconds 420 --sim-speedup 120 --record`。

### 4.3 评价指标

本文使用的原始字段包括：`catches`、`destroyed`、`delta_v_total`、`sim_time`、`spawned_debris_total`。在此基础上进一步定义派生指标如下：

$$
\eta_i = \frac{\text{catches}_i}{\text{spawned\_debris\_total}_i}
$$

$$
\lambda_i = \frac{\text{catches}_i}{\text{sim\_time}_i}
$$

毁伤率定义为：

$$
r_{destroy}=\frac{\sum_i \text{destroyed}_i}{N}
$$

### 4.4 统计量说明

本文统一报告均值（mean）、标准差（std）、最小值（min）和最大值（max）四类统计量。

---

## 5 结果与分析

### 5.1 总体性能

6 个回合（N=6）的统计结果如表 5-1 所示。

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| `catches` | 2.167 | 0.983 | 1 | 4 |
| `delta_v_total` | 9945.333 | 1004.144 | 8685 | 11305 |
| `sim_time` (s) | 601.750 | 0.493 | 601.1 | 602.3 |
| `spawned_debris_total` | 8.333 | 2.582 | 5 | 10 |
| `spawned_meteors_total` | 0.000 | 0.000 | 0 | 0 |

毁伤率为：

$$
r_{destroy}=0/6=0.000
$$

### 5.2 捕获效率

单回合捕获效率定义为：

$$
\eta_i = \frac{\text{catches}_i}{\text{spawned\_debris\_total}_i}
$$

观测结果显示：平均效率为 0.300，最小效率为 0.200，最大效率为 0.800。

![Fig 5-1 Capture Count by Episode](reports/exp20260303_catches.png)
![Fig 5-2 Capture Efficiency by Episode](reports/exp20260303_efficiency.png)

### 5.3 时间与控制代价

各回合仿真时长稳定，满足：

$$
\text{sim\_time}=601.750 \pm 0.493\ \text{s}
$$

捕获频率范围为：

$$
\lambda_i=\frac{\text{catches}_i}{\text{sim\_time}_i}\in[0.00166,\ 0.00665]\ \text{s}^{-1}
$$

折算后单次捕获对应时间约为 150.4 s 至 601.1 s。控制代价指标 `delta_v_total` 的统计结果为 9945.333 +/- 1004.144。

![Fig 5-3 Delta-v Total by Episode](reports/exp20260303_delta_v.png)

### 5.4 稳定性与收益耦合

由于本数据集中未显式记录速度/角速度分箱信息，本文采用“稳定性-收益-代价”三维视角进行分析：稳定性方面，毁伤率为 0.000（无回合发生毁伤）；收益方面，平均捕获数为 2.167；代价方面，平均 `delta_v_total` 为 9945.333。单位代价收益定义为：

$$
\rho=\frac{\text{catches}}{\text{delta\_v\_total}}
$$

由均值可得：

$$
\rho_{mean}\approx2.18\times10^{-4}
$$

### 5.5 变异性分析

变异系数（CV）定义与结果如下：

$$
\mathrm{CV}_{\Delta v}=\frac{1004.144}{9945.333}=0.101
$$

$$
\mathrm{CV}_{catches}=\frac{0.983}{2.167}=0.454
$$

可以看出，收益侧波动显著高于代价侧波动。

### 5.6 分回合结果

| 回合 | 随机种子 | 捕获数 | 毁伤数 | 仿真时长 (s) | Delta-v |
|---:|---:|---:|---:|---:|---:|
| 1 | 20260303 | 4 | 0 | 601.6 | 11305 |
| 2 | 20260304 | 2 | 0 | 602.2 | 9615 |
| 3 | 20260305 | 2 | 0 | 602.3 | 10195 |
| 4 | 20260306 | 2 | 0 | 601.1 | 8685 |
| 5 | 20260307 | 2 | 0 | 602.0 | 10060 |
| 6 | 20260308 | 1 | 0 | 601.3 | 9812 |

完整流程可通过 `run_experiments_and_record.ps1` 复现实验。

## 6 结论与展望

### 6.1 总结

本次修订形成了完整的工程化实验闭环，包括可执行脚本、量化 CSV 输出以及可复现实验视频。

### 6.2 主要发现

基于 2026-03-03 的批量运行结果（6 回合），可以得到以下结论：

1. 毁伤率为 0.000。
2. 平均捕获数为 2.167（范围 1 至 4）。
3. 平均 `delta_v_total` 为 9945.333（标准差 1004.144）。
4. 收益侧 CV（0.454）高于代价侧 CV（0.101），表明结果端对条件变化更敏感。

### 6.3 后续工作

1. 将样本规模扩展到 30/60/100 回合，并报告置信区间。
2. 显式记录速度、角速度与材质等字段，支持响应面分析。
3. 在统一评价协议下增加多策略对比（physics/lstm/reactive）。
4. 建立从 CSV 到图表与结论的自动化报告流程。

---

## 参考文献

[1] Bonnal C, Ruault J M, Desjean M C. Active debris removal: Recent progress and current trends[J]. Acta Astronautica, 2013, 85: 51-60.

[2] Shan M, Guo J, Gill E. Review and comparison of active space debris capturing and removal methods[J]. Progress in Aerospace Sciences, 2016, 80: 18-32.

[3] Flores-Abad A, Ma O, Pham K, et al. A review of space robotics technologies for on-orbit servicing[J]. Progress in Aerospace Sciences, 2014, 65: 1-26.

[4] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735-1780.

[5] Camacho E F, Bordons C. Model predictive control[M]. Springer, 2007.

[6] Yoshikawa T. Manipulability of robotic mechanisms[J]. The International Journal of Robotics Research, 1985, 4(2): 3-9.

[7] Wampler C W. Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods[J]. IEEE Transactions on Systems, Man, and Cybernetics, 1986, 16(1): 93-101.

[8] Siciliano B, Khatib O. Springer handbook of robotics[M]. Springer, 2016.

[9] Botta E M, Sharf I, Misra A K. Contact dynamics modeling and simulation of tether nets for space-debris capture[J]. Journal of Guidance, Control, and Dynamics, 2017, 40(1): 110-123.

[10] Aghili F. A prediction and motion-planning scheme for visually guided robotic capturing of free-floating tumbling objects with uncertain dynamics[J]. IEEE Transactions on Robotics, 2012, 28(3): 634-649.

---

## 附录 A：2026-03-03 复现实验包

### A.1 命令

批量运行：
`python orbit_simulator_experiment.py --experiment --episodes 6 --sim-seconds 600 --sim-speedup 100 --seed 20260303 --output experiment_results/orbit_experiment_results_20260303.csv`

基线视频：
`python orbit_simulator.py --sim-seconds 420 --sim-speedup 120 --autopilot --record --video-fps 30 --video-path videos/baseline_orbit_simulator_20260303.mp4`

实验视频：
`python orbit_simulator_experiment.py --sim-seconds 420 --sim-speedup 120 --record --video-fps 30 --video-path videos/experiment_orbit_simulator_20260303.mp4`

### A.2 产物

- `experiment_results/orbit_experiment_results_20260303.csv`
- `videos/baseline_orbit_simulator_20260303.mp4`
- `videos/experiment_orbit_simulator_20260303.mp4`

### A.3 关键统计

- 回合数：6
- 平均捕获数：2.167
- 毁伤率：0.000
- 平均 delta-v 总量：9945.333

### A.4 说明

本附录提供了从命令到数据再到视频的直接复现链路（commands -> data -> videos），与第 5 章分析保持一致。
