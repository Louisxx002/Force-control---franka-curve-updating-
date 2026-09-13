# 自动擦拭流程

**当前选择：原 Demo，MoveIt 服务已停止，以下 MoveIt 章节保留作备用开发记录。** 原自动几何预览使用 `./run.sh autowipe`，不带 MoveIt 参数。对比见 [原 Demo 与 MoveIt 方案对比](原Demo与MoveIt方案对比.md)。

备用方案：MoveIt + RealSense，无现场 CAD。 环境由相机点云生成观测体素，末端附件用简单尺寸包络；下方早期手工环境模型说明是历史方案，最新命令见末尾“无 CAD 点云方案”。

目标是一次配置工作区域和工具，之后执行“扫描 → 识别 → 接近规划 → 接近 → 复扫 → 力控擦拭 → 退回 → 效果复查”，只有异常或场景改变时才要求人工处理。

## 已实现：自动扫描和规划预览

```bash
cd /home/pnp/curve_wipe_demo
./run.sh autowipe
```

该入口读取相机和机器人位姿，不发送运动命令。相机须未被预览或其他程序占用。
默认输出到时间戳目录，不需要填写 ROI、选择历史 plan 或手工逐段试跑。

- 全图检测具有有效深度的红色连通区域，过滤小噪点。
- 自动生成每个区域的局部表面路径，逐段检查固定姿态适用性。
- 按擦拭长度、接近路程排序候选，记录每条拒绝原因。
- 生成保持姿态的“必要时抬高 → 悬空平移 → 降至预接触点”几何方案。
- 接近方案不设 12 cm / 50 cm 距离门槛；仅检查已观测局部表面相对于 TCP 的净空。这不是机械臂、工具或环境的碰撞验证。
- 自动生成的 plan 始终标记不可执行，不能直接传给现有 execute 实机入口。

`targets.png` 显示检测范围，`region_*/overlay.png` 和 `trajectory.png` 显示路径，`report.json` 保存候选、接近位姿、失败原因及缺失检查。
状态 `awaiting_motion_planning` 表示有几何候选但尚未通过实机路径检查；不是已完成擦拭。

离线复现本次现场图像：

```bash
./run.sh autowipe --snapshot output/retry_20260912_fourth/snapshot.npz
```

此命令完全不连接硬件。相机未检测到红色目标时输出 `no_target`，不回用旧轨迹；目标存在但没有固定姿态候选时输出 `no_fixed_orientation_candidate`。采集失败会保存 `failed` 报告，不进入运动。

## 实机闭环仍需完成

1. **限定工作对象。** 当前识别只按红色，不能区分痕迹、胶带、按钮和标识。一次性配置机器人基坐标下的工件区域和允许擦拭表面；像素范围随相机移动自动更新。全图检测结果只是候选。
2. **全臂接近规划。** 使用当前关节状态、实际机器人型号、当前 F_T_EE、工具及相机/传感器包络和环境模型，做连续可达性、自碰撞及环境碰撞检查。MoveIt 已安装，但没有运行的规划服务，也没有在本 demo 接入经过核对的工具/环境场景。直线路径失败后才考虑绕行；未知空间不能视为空闲。
3. **按验证过的轨迹执行。** 不能仅验证终点或检查另一条关节路径，再交给现有 CartesianMotion 自行选解。需要执行已验证轨迹，或确认实时跟踪的整臂构型与验证路径一致，并保留力、速度、超时、关节和跟踪保护。
4. **到位自动复扫。** 接近后确认目标仍在原位置、重新估计局部曲面并采集悬空力基线。安装或目标变化时废弃旧计划。
5. **工具姿态及力控。** 固定姿态只适用于局部浅曲面；变姿态擦拭仍需完成当前安装的重力标定。不能通过改状态字段使用未通过的标定。
6. **覆盖和结果复查。** 目前不保证整个擦子 footprint 的接触、整块痕迹覆盖，也不规划多段间的移动。应先完成单段自动闭环，再按擦子有效宽度规划覆盖。复扫后在同一工件坐标区域比较残留痕迹；设置次数/时间上限，避免胶带等不可擦目标引发无限重复。

原 execute / adaptive_execute 实机入口的距离保护本次未改动。自动接近执行链路和场景验证接通后，再用整条路径的验证结果替代该阈值；不是仅将数值调大。

## 本次验证

保存的真实图像自动识别出 1 个目标区域、15 条表面片段，其中 6 条通过固定姿态几何检查。优先候选长约 15.75 mm，直接到预接触点约 138.43 mm，生成的分段悬空方案总长约 189.67 mm。白色印字会切断红色片段；未跨越这些空白强行连接。

单元测试涵盖噪点/无深度目标过滤、超过 50 cm 的几何预览、TCP 偏移及局部净空、接触状态拒绝，以及合成数据全流程不会标记为可执行。未进行实机运动验证。

## 第二步已加入：MoveIt 只读路径审查

`check_approach` 会读取已生成的候选报告与保存的关节状态，调用现有 MoveIt 服务。它不启动控制器、不修改场景、不连接执行动作，也不自动开启 FCI。

```bash
./run.sh check_approach \
  --report output/autowipe_replay_initial/report.json \
  --config config/motion_scene.example.json \
  --output output/approach_audit_new.json
```

示例配置只是 FR3 命名示例，不表示当前机器人或场景已确认。必须核对实际型号、关节顺序、基坐标及法兰 link，并让场景包含真实工具、相机/传感器和工件、桌面、支架的碰撞几何。脚本不会用虚构尺寸补齐缺失物体。

检查包括：

- 保存的图像、plan、关节状态、F_T_EE 一致性；按完整变换将 EE 目标换算到法兰。
- 必需的工具附件和环境物体有碰撞几何；必需障碍物没有被允许碰撞矩阵整体/成对豁免。
- 机器人模型 FK 与保存的起始 EE 位姿误差不超过 2 mm / 0.5°。
- 启用避碰的笛卡尔路径必须完整返回；部分路径即使有解也不通过。
- 关节顺序、起始关节、时间戳、非有限数、0.10 rad 跳变及已有 joint-2 边界检查。
- 以不超过 0.01 rad 的关节增量加密检查返回路径，并核对终点 FK；检查期间场景变化则拒绝。

成功状态为 `sampled_scene_audit_passed`，仍标记 `executable=false`。这只代表在所提供场景中的采样路径检查，不代表连续扫掠体证明，也不代表物理环境模型已核对。还须接入实际已验证轨迹的执行、实时状态复核、擦拭和返回路径验证。

可以串在自动入口后，不必分别调用：

```bash
./run.sh autowipe --motion-config config/motion_scene.example.json
```

未提供配置时仍只生成几何预览。服务未启动、场景缺失、模型不匹配、路径部分成功或碰撞等都记录在 `motion_audit.json`，不会转入运动。`run.sh` 为审查入口使用 ROS Jazzy 系统 Python，其余入口仍使用项目虚拟环境。

本机首次实测返回 `MoveIt service unavailable: /get_planning_scene`，审查结果为 blocked，未移动机械臂。尚未在运行中的 MoveIt 场景验证成功路径。

服务接口另有 5 项模拟响应测试，覆盖完整路径、缺少几何、部分路径、模型 FK 不匹配及碰撞拒绝。这些是接口测试，不替代真实 MoveIt 场景验证。

## 2026-09-12 续接：本机规划服务已启动

新增独立服务入口（前台运行，Ctrl+C 停止）：

```bash
./run.sh planning
```

服务位于 `/autowipe`，使用本机 FR3 URDF/SRDF、KDL 和 OMPL。启动 robot_state_publisher 提供固定坐标变换；没有硬件驱动或控制器，不发布伪造的实时关节状态。`allow_trajectory_execution=false`，并禁用 Move/ExecuteTrajectory 动作能力。启动设置参考 [MoveIt 源码](https://moveit.picknik.ai/main/api/html/move__group_8cpp_source.html)。

本地配置使用 `config/motion_scene.local.json`；原示例配置仍保留。

```bash
./run.sh planning_probe \
  --snapshot output/retry_20260912_fourth/snapshot.npz \
  --config config/motion_scene.local.json \
  --output output/model_probe_new.json
```

实际服务结果保存在 `output/moveit_model_probe_tf.json`：保存姿态的法兰 FK 位置误差 8.59e-8 m，姿态误差 1.13e-5 度。场景读取、FK、状态有效性服务已成功响应。仅核对一个历史姿态，不代表实时状态或碰撞几何得到验证。第一次探测暴露了 base 与 fr3_link0 的固定 TF 缺失，加入状态发布器后复测通过。

实际路径审查记录 `output/approach_audit_service_live.json` 已从“服务不存在”推进到“缺少场景几何”。未生成通过审查的现场路径，未进行实机运动。

### 填入实测场景后继续

`config/scene_measurements.template.json` 是待测量表，空值有意保留，无法直接导入。复制为新文件，填写所有必需物体的包围盒：

- `size_m`：盒子三个轴方向的完整尺寸，单位米，包络需要覆盖完整物体。
- `T_frame_box`：4×4 刚体变换，盒子中心和方向相对于 `frame`；不是盒子角点。
- 桌面、工件、支架相对于 `fr3_link0`；擦子/夹具总成、相机/传感器总成相对于 `fr3_link8`。
- 核实实测数据后才设置 `measurements_confirmed=true`。旧 TCP 候选参数不能替代完整附件包络。

```bash
./run.sh load_scene --scene config/scene_measured.json --config config/motion_scene.local.json
./run.sh autowipe --motion-config config/motion_scene.local.json
```

导入只修改 `/autowipe` 规划场景，不发送运动。它校验单位、必需物体、唯一 ID、尺寸、坐标系和刚体变换；同名物体更新，其他已有物体保留。服务重启后需重新导入。当前导入器支持每个物体一个实测包围盒；复杂曲面应使用合适网格/分解几何，不能为了让路径通过而缩小包络。只有附件安装 link 被列为 touch link。

**剩余工作：** 提供真实场景尺寸并验证碰撞路径；验证实际 CartesianPath 返回和必要时的 OMPL 绕行；接通经过审查的原始关节轨迹执行及实时状态复核；随后接入复扫、力控擦拭和返回。当前不能执行完整自动擦拭。

## 无 CAD 点云方案

无需桌面、支架或工件 CAD。新增 `depth_scene` 将现有 RealSense RGB 光学坐标系深度点，经采集时的机器人位姿和手眼标定变换到 `fr3_link0`，按 20 mm 体素归并，每侧增加 10 mm 余量。这里使用 MoveIt CollisionObject 的盒子集合表示观测体素，不是原生 OctoMap，也不把曲面拟合成一个大平面。参数是初始规划预览设置，尚未按现场误差和最小通道宽度标定。

[MoveIt 官方感知说明](https://moveit.picknik.ai/main/doc/concepts/planning_scene_monitor.html)也支持点云/深度图占据地图；本项目先采用静态快照体素，以便与轨迹所用图像及位姿严格对应。

```bash
./run.sh planning
# 在另一终端执行；此命令读取硬件，但不移动机械臂
./run.sh autowipe --depth-scene
```

离线重放图像、连接真实规划服务（不连接相机或机械臂）：

```bash
./run.sh autowipe --snapshot output/retry_20260912_fourth/snapshot.npz \
  --output output/autowipe_depth_replay_new --depth-scene
```

只生成点云场景文件，不连接 ROS 服务时省略 `--apply`：

```bash
./run.sh depth_scene --snapshot output/retry_20260912_fourth/snapshot.npz \
  --output output/depth_scene_new --apply
```

输出 `voxels.npz`、`scene.json`、`motion_config.json`；成功导入时还有 `apply_result.json`。读回核对体素数量、尺寸、姿态、位置，并处理 MoveIt 将物体变换到模型根坐标系的情况。场景和轨迹审查绑定同一快照及手眼标定哈希。体素数量超过上限会拒绝，不偷偷删点；不按红色筛除其他障碍物、不填深度孔洞，也不删除已有环境物体。

### 真实服务验证

`output/depth_scene_verified/apply_result.json`：182,965 个有效深度点生成 197 个观测体素，导入并读回核对成功。使用历史图像，仅代表软件链路验证；不代表当前场景。当前没有机器人自过滤，机器人自身若进入画面可能导致保守碰撞拒绝。

新配置中的必需世界物体改为本次点云几何 ID，替代分别要求 table/workpiece/fixture。末端附件的检查仍保留。需要的是尺量的整体包络及其相对法兰的安装位置，无需 CAD。复制测量模板，只填写 `attached` 并确认测量后，可以单独导入附件，不改点云环境：

```bash
./run.sh load_scene --attachments-only \
  --scene config/tool_envelopes_measured.json --config config/motion_scene.local.json
```

### 当前边界与剩余工作

- 单幅相机只建立已观测障碍物；MoveIt 本身不会替本入口把未建模空间当成障碍。本程序始终禁止执行，尚需多视角观测或可验证的运动区域约束来处理遮挡/未知空间。
- 旧点云保留在场景里；工件移动后旧几何可能造成拒绝，不能直接以新单帧删除旧观测并宣称空闲。规划服务重启会清空场景，须重新加载附件和扫描。
- 尚缺真实末端附件包络，因此整条接近路径审查仍会阻塞。尚未打通接近轨迹执行、OMPL 绕行回退、到位复扫和自动力控闭环。
- 不需要提供现场 CAD；需要补的物理信息是附件尺寸/安装位置及观测覆盖。`--depth-scene` 与 `--motion-config` 互斥，前者自动生成配置。
