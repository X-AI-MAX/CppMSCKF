# CppMSCKF-Zero —— 从零实现的视觉惯性里程计

**语言**: C++17  
**协议**: MIT License  

---

## 项目简介

本项目是从零开始实现的MSCKF（多状态约束卡尔曼滤波）算法，无任何第三方依赖。包含自定义数学库、传感器驱动、视觉处理模块，适用于GPS拒止环境下的无人机自主导航。

**明确声明**：本项目不包含数据集测试基准，专注于算法实现和工程化部署。

---

## 算法原理

### MSCKF核心

松耦合视觉-惯性融合，滑动窗口维护多个相机位姿，特征点逆深度参数化。

### 关键创新

1. **自适应协方差调节**：根据运动激励度动态调整过程噪声，包含自适应鲁棒核函数（Huber、Cauchy、Tukey、McClure、Geman-McClure）
2. **增量式舒尔补加速**：利用Woodbury恒等式降低视觉更新复杂度，支持稀疏Cholesky分解
3. **多机协同与语义辅助**（可选模块）：支持多无人机协同定位和语义分割辅助

---

## 功能特性

### 基础功能

- 视觉-惯性融合定位
- 实时位姿输出
- 滑动窗口优化
- 在线外参标定

### 工程特性

- 零动态内存分配（静态内存池）
- 无锁队列通信（SPSC Queue）
- 硬件抽象层（HAL）
- 固定点数学运算支持

### 可选功能

- 多机协同定位
- 语义分割辅助
- 一键返航（RTH）
- 低电量决策
- 磁干扰抑制
- 黑匣子记录
- 参数自整定

---

## 编译部署

### 环境要求

- ARM GCC 10+
- CMake 3.16+
- 裸机或FreeRTOS SDK

### 编译步骤

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=arm-none-eabi.cmake
make
```

### 烧录运行

使用J-Link或OpenOCD烧录至目标板，串口输出位姿数据。


## 开源协议

MIT License

自由使用、修改、分发，保留版权声明。

---

## 致谢

感谢MSCKF原始论文作者。本项目参考了开源项目的架构设计，所有代码均为从零实现，非代码复制。

---

---

# CppMSCKF-Zero —— A From-Scratch Visual-Inertial Odometry

**Language**: C++17  
**License**: MIT License  
---

## Introduction

A from-scratch implementation of MSCKF (Multi-State Constraint Kalman Filter) with zero third-party dependencies. Includes custom math library, sensor drivers, and visual processing modules for GPS-denied UAV navigation.

**Disclaimer**: No dataset benchmarking included. Focused on algorithm implementation and deployment.

---

## Algorithm

### MSCKF Core

Loosely-coupled visual-inertial fusion with sliding window and inverse depth parameterization.

### Key Innovations

1. **Adaptive Covariance Tuning**: Dynamic process noise adjustment based on motion excitation, with adaptive robust kernels (Huber, Cauchy, Tukey, McClure, Geman-McClure)
2. **Incremental Schur Complement Acceleration**: Woodbury identity for reduced visual update complexity, with sparse Cholesky decomposition
3. **Multi-UAV Collaboration and Semantic Assistance** (Optional): Multi-drone cooperative localization and semantic segmentation aid

---

## Features

### Basic

- VIO localization
- Real-time pose output
- Sliding-window optimization
- Online extrinsic calibration

### Engineering

- Zero dynamic allocation (static memory pool)
- Lock-free queues (SPSC Queue)
- Hardware abstraction layer (HAL)
- Fixed-point math support

### Optional

- Multi-UAV collaboration
- Semantic assistance
- Return-to-home (RTH)
- Battery-aware planning
- Magnetic rejection
- Black box recorder
- Auto-tuning

---

## Build & Deploy

### Requirements

- ARM GCC 10+
- CMake 3.16+
- Bare-metal or FreeRTOS SDK

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=arm-none-eabi.cmake
make
```

### Deployment

Flash via J-Link or OpenOCD, pose output via UART.

## License

MIT License

Free to use, modify, distribute. Retain copyright notice.

---

## Acknowledgments

Thanks to MSCKF original paper authors. This project references open-source architectures only; all code is implemented from scratch, not copied.
