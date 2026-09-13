# curve_wipe_demo

RealSense 曲面估计与 Franka 原始方案擦拭 Demo。项目使用一次扫描生成完整目标中心线，根据局部曲面法向生成空间位姿，并在接触后进行法向力导纳控制。

## 快速运行

```bash
./run_full_wipe.sh --target-color black --lanes 1
```

加上 `--execute` 才会连接机械臂并执行运动：

```bash
./run_full_wipe.sh --execute --target-color black --lanes 1
```

支持 `red`、`orange`、`yellow`、`lime`、`green`、`cyan`、`blue`、`violet`、`purple`、`magenta`、`pink`、`brown`、`gray`、`white`、`black`。完整参数和硬件注意事项见 [端到端运行说明.md](端到端运行说明.md)，第一版记录见 [VERSION_1.md](VERSION_1.md)。

## 测试

```bash
pytest -q
```

当前基线包含单条完整中心线、夹爪中心到擦拭中心 15 mm 的局部 Z 偏移、曲面法向对齐和 1 N 法向目标力。
