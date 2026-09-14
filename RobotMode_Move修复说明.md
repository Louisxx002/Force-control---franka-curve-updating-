# RobotMode.Move 停止问题修复

`RobotMode.Move` 是正常运动状态，不能把日志中的这个名称当作根因。之前的 `robot.has_errors or mode not in allowed_modes` 把两种原因合并成同一条消息，而且 `has_errors` 会再次读取机器人状态，可能与先前读到的模式不属于同一帧。

## 已有证据

- `output/visible_extent_white_exec_retry_20260913/lane_00/wipe.json` 的 `cleanup_errors` 包含笛卡尔及关节加速度不连续，命令成功率为 1。
- `output/visible_extent_white_exec_retry2_20260913/lane_00/wipe.json` 的 `cleanup_errors` 包含笛卡尔速度、加速度不连续和 `communication_constraints_violation`，命令成功率为 0.59。
- 这些错误是在清理阶段取出的。旧日志没有故障瞬间的完整状态，因此无法仅凭旧日志确定所有错误的先后因果。

## 当前主程序修改

入口仍为 `run_full_wipe.sh`，修改位于 `curve_wipe/execute.py`。

1. 用同一帧 `current_errors` 和 `robot_mode` 判断。正常 Idle/Move 可通过，真实错误和其他模式仍停止。逐个读取错误布尔字段；Python 中空 `franky.Errors()` 对象的真值也是 True，不能直接用 `bool(errors)`。
2. 停机前保存真实错误名称、历史运动错误、控制命令成功率及阶段到 `robot_fault`，每条采样也保存成功率。
3. 接触确认后，短暂低于 0.2 N 不再直接冻结切向进度；法向导纳继续调整，持续失去接触超过 2 s 仍停止。正反向端点仍使用原有平滑加减速。
4. 法向导纳改为时间常数 0.2 s 的一阶速度响应，从接近速度连续进入力调节。目标力保持 1 N，原有力、力矩与穿透保护继续生效。
5. Python 调度延迟不再转换成单步追赶位移：运动积分步长最多 10 ms；超时和失去接触判断继续使用真实时间。

短暂力下降导致切向停走、无状态导纳导致速度阶跃，是代码中确认存在的不连续来源；尚未证明它们是两次硬件故障的唯一原因。通信约束错误需要运行时继续观察，不能通过允许 Move 或放宽力阈值解决。

## 验证范围

146 项离线测试通过，包括实际 Franky 枚举/空错误对象、Move 状态下真实故障、异常模式、调度延迟和导纳速度连续性测试。随后完成一次实机往返验证（见下方记录）。当前仓库已与唯一活动版本 `main@712d7d8` 对齐。

## 实机验证记录

`output/move_fix_verify_20260913_01/cycle_report.json` 确认 `completed_all_lanes_and_returned`。白色胶带可见路径 182.45 mm，完成正向、反向和回位，清理无异常。回位误差 0.352 mm、0.043°。

全过程记录最大法向力 3.361 N，擦拭阶段均值 0.999 N，采样中最低命令成功率 0.000。本次未复现停机，但一次成功不等于排除了间歇通信故障。
