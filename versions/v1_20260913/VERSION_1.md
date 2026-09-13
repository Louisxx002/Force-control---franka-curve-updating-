# curve_wipe_demo 第一版（v1）

- 保存日期：2026-09-13
- 版本标识：`v1_20260913`
- 基线：当前工作区中经过硬件验证的原始方案（不使用 MoveIt）

## 已固化功能

- RealSense 单次扫描识别目标胶带并生成完整中心线。
- 支持 `red`、`white`、`black` 三种目标颜色；黑色胶带支持斜向主轴提取。
- 以夹爪中心作为擦拭参考点，擦拭点沿夹爪局部 Z 轴前移 15 mm。
- 根据曲面空间法向量预先生成位姿轨迹，使夹爪 Z 轴与曲面法向对齐。
- 接触前自由空间不启用力控制；接触后启用法向导纳，目标法向力 1 N。
- 连续接触判定 1 s；总力和法向力软件上限均为 8 N。
- 轨迹采用匀速段和 0.5 s 的受限进出段，默认速度倍率 1.5（擦拭速度约 4.5 mm/s）。
- 擦拭完成后执行有界回位修正，并校验回到擦拭开始位姿。

## 本版硬件验证

黑色斜胶带单次扫描、单条完整中心线擦拭已完成：

- 状态：`completed_all_lanes_and_returned`
- 中心线长度：263.58 mm
- 擦拭段最大法向力：2.149 N
- 全过程最大法向力：2.567 N
- 全过程最大合力：4.114 N
- 回位误差：0.362 mm、0.044°（最终检查 0.387 mm、0.047°）
- 验证记录：`validation/black_tape_wipe_run_20260913/`

## 软件验证

在保存前执行：

```text
137 passed in 3.44s
```

## 使用方式

从项目根目录运行，例如：

```bash
./run_full_wipe.sh --execute --target-color black --lanes 1 \
  --max-surface-height-mm 30 --max-normal-angle-deg 45 \
  --speed-scale 1.5 --output output/cycle_black_tape_wipe_run_YYYYMMDD
```

版本快照位于 `versions/v1_20260913/`，压缩包为 `curve_wipe_demo_v1_20260913.tar.gz`。后续实验可复制该快照作为基线。

## 已知边界

- 当前日志记录了力和轨迹，但 `contact_coverage_verified` 仍为 `false`，覆盖率尚未由独立传感器确认。
- 白色和黑色胶带的深度缺失点由邻近曲面点局部拟合重建；拟合残差超过 2 mm 时会拒绝规划。
- 快照不包含 Python 虚拟环境、pytest 缓存和完整历史输出；本版硬件验证所需的关键报告与轨迹已收录在 `validation/`。
