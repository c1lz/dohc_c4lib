# DOHC2 C4 Library

新增彩色 EuRoC 入口：`collect_cam0_imu.py`（单目＋IMU，支持离线导入 Mocap）
与 `collect_fourcam_imu_euroc.py`（四目＋IMU）。运行命令、输出格式和
无真机测试见 [CAPTURE_EUROC.md](CAPTURE_EUROC.md)。

该目录集中保存 DOHC2/OAK-4P 四目采集、FSIN 检查、AprilGrid 引导采集和
BNO086 Allan 噪声计算所需的入口、内部 Python 模块、依赖清单及外参配置。

## 环境

从项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r dohc2_c4lib/requirements.txt
```

已有项目环境也可直接使用：

```bash
IMU_API/depthai-core/venv/bin/python -m dohc2_c4lib.record_oak4p --help
```

包内入口同时支持直接按文件路径执行，例如：

```bash
/home/descfly/dohc2_c4lib/.venv/bin/python \
  /home/descfly/dohc2_c4lib/record_oak4p.py --help
```

Windows 使用 `requirements-windows.txt`；其中额外包含 GUI 和打包依赖。

## 入口

```bash
# 四路 FSIN 同步预览（按安装方向上下、左右翻转）
python -m dohc2_c4lib.show_oak4p_fsin_sync_vertical_flip --sync-mode fsin

# 单相机内参采集
python -m dohc2_c4lib.collect_aprilgrid_intrinsics --camera cam0

# 两相机共视外参采集
python -m dohc2_c4lib.collect_aprilgrid_pairs \
  --pair cam0-cam1 --sync-mode fsin --max-skew-us 1000

# 四相机与板载 BNO086 连续采集
python -m dohc2_c4lib.record_oak4p \
  --mode imu_calib --sync-mode fsin --fps 30 --imu-rate 400 --codec png

# 四路满分辨率原彩（未压缩 NV12）与 IMU 十分钟压力测试，约需 103 GiB
python -m dohc2_c4lib.record_oak4p \
  --mode vio --codec nv12 --sync-mode fsin --fps 30 --imu-rate 400 \
  --duration 600 --headless --capture-strategy stream-first \
  --queue-size 1024 --require-usb3 --output-dir /path/with/at-least-110GiB-free

# 采集后检查
python -m dohc2_c4lib.validate_oak4p_dataset /path/to/session

# 从静止 RAW IMU 数据生成 Kalibr imu.yaml
python -m dohc2_c4lib.generate_bno086_imu_yaml /path/to/allan_session
```

硬件采集入口继续保持 `cam0..cam3 = CAM_A..CAM_D`。相机--IMU 标定必须使用
BNO086 的 `ACCELEROMETER_RAW` 和 `GYROSCOPE_RAW`，不要用融合姿态数据。

## 配置归档

- `configs/target.yaml`：6x6 AprilGrid 几何参数。
- `configs/rig_fsin.yaml`：FSIN、1 ms 门槛、omni-radtan，推荐的四目外参输入。
- `configs/rig_eucm_fsin.yaml`：FSIN、1 ms 门槛、eucm-none 备选模型。
- `configs/rig.yaml`、`configs/rig_ds_none.yaml`：20 ms/free-run 或模型对照配置。
- `configs/datawash.yaml`：CO-Calib 数据筛选参数。
- `configs/imu_bno086.yaml`：由 Allan 数据生成，不提供虚构默认值。

所有 rig 文件固定相机顺序和物理映射。Kalibr 的变换约定为
`T_cam_imu`（IMU 坐标到相机坐标），时间约定为
`t_imu = t_cam + timeshift_cam_imu`。刚性四目 VIO 的最终结果应使用工作流生成的
`rig-consistent-camchain-imucam.yaml`；各相机独立结果只用于诊断。

CO-Calib/Kalibr 本体仍由项目工作流及 `omnicalib-kalibr:latest` 镜像提供；本目录
不自动安装或拉取 Docker 镜像。建议先在项目根目录运行工作流的 `--dry-run`。

## 目录说明

五个指定入口的实现已迁移到本目录。`aprilgrid_common.py`、
`guided_capture.py`、`frame_sync.py`、`oak4p_dataset.py` 和
`show_oak4p_fsin_sync.py` 是它们的本地依赖；验证器与 IMU YAML 生成器也一并归档。
旧的 `oak4p_calibration.*` 入口是兼容包装层，现有调用无需立即修改。
