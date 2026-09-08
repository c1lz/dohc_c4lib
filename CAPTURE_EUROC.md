# 彩色相机＋IMU 采集与 Mocap 导入

两个入口均支持直接执行，无需配置 PYTHONPATH。默认 1280×800、30 FPS、
400 Hz RAW accel/gyro、FSIN、USB3、彩色无损 PNG。采集期间不抽帧，
不因为标定板暂时不可见而删帧。EUCM 内参和相机间外参由后续标定器加载。

## 1. cam0＋IMU 动态标定采集

在本项目目录、虚拟环境可用时执行：

```bash
.venv/bin/python collect_cam0_imu.py \
  --output-dir ./euroc_datasets/calibration \
  --duration 90 \
  --fps 30 --imu-rate 400 \
  --sync-mode fsin --max-skew-us 1000 \
  --manual-exposure-us 2000 --iso 400 \
  --queue-size 256 --image-workers-per-camera 2
```

只启用 CAM_A 和 IMU，输出的会话目录名含微秒时间戳，重复执行不会覆盖旧数据。
单目默认时长90秒。命令启动后预热2秒，打印数据目录后开始正式录制。
脚本不等待 AprilGrid 出现，操作者应提前固定好标定板。

固定 AprilGrid，移动 Rig：开始静止5秒，依次做 roll/pitch/yaw 正负旋转、
XYZ 加减速、平滑组合运动，最后静止5秒。保证 cam0 经常能清晰看到板。
录两条独立标定序列，再单独录验证序列；已有内参和相机间外参不在采集时更改。

画质每秒评估2次，默认门槛与内参采集 CLI 对齐：至少6个 tag、清晰度80、
平均亮度45–220、clipped不超过30%、板面积占比2.5%–65%。可追加
`--max-clipped-percent 5` 做更严格检查。clipped 指代码度量中的暗端与亮端
像素比例，不只代表白色过曝；不能把指标数值视为角点精度的保证。
质量评估只记录和告警，不使用“停稳0.5秒”条件过滤动态数据。

2000 µs / ISO400 是起点，不保证现场亮度或运动清晰度合格。需先检查图像，
根据灯光调整曝光和ISO，正式录制保持固定；镜头焦距必须维持已有内参标定时的状态。
若模组支持调焦，追加 `--manual-focus N`，N 必须是已知固定值，不要猜测。
脚本未锁定白平衡，manifest 会如实记录这一点。

## 2. 四相机＋IMU EuRoC 采集

```bash
.venv/bin/python collect_fourcam_imu_euroc.py \
  --output-dir ./euroc_datasets/vio \
  --duration 120 \
  --fps 30 --imu-rate 400 \
  --sync-mode fsin --max-skew-us 1000 \
  --manual-exposure-us 2000 --iso 400 \
  --queue-size 256 --image-workers-per-camera 2
```

启用 CAM_A/B/C/D。省略 `--duration` 时持续记录，Ctrl+C 正常停止并收尾。
不要求场景中存在 AprilGrid，无板不会停止采集。两个入口均为终端运行，无GUI。
每秒打印各流接收频率、写入数、队列深度和设备序号缺口。

外部 FSIN 触发源必须已配置并匹配目标频率；`--fps 30` 不会生成外部触发信号。
默认强制USB3；`--allow-usb2` 仅供诊断，不代表性能通过。
此前四目30 FPS存在缺帧，本版本尚未真机验证；首次接机先用
`--duration 30` 短测，不自动降低用户指定帧率。

## 3. 独立录制并导入 Mocap

动捕与 cam0＋IMU 在同一段 Rig 动作期间独立录制。准备项目标准 CSV：

```csv
timestamp_ns,px,py,pz,qw,qx,qy,qz,tracking_valid
1788839123456000000,1.2,0.3,1.1,1,0,0,0,1
1788839123461000000,1.2,0.3,1.1,1,0,0,0,1
```

时间戳为整数纳秒，位置单位米，四元数顺序wxyz，位姿方向
`T_mocap_world_mocap0`。也接受数据格式文档中完整带单位的 EuRoC 表头。
tracking_valid 可省略，省略即表示所有行有效；若有无效追踪应显式提供。
支持9列或8列，不直接接受厂商自定义列、浮点秒、毫米或xyzw。

```bash
.venv/bin/python collect_cam0_imu.py \
  --import-mocap /absolute/path/mocap.csv \
  --session-dir /absolute/path/cam0_imu_session \
  --mocap-clock-domain mocap_pc_clock
```

导入后保留源文件字节和原始时钟域，校验有效四元数范数及时间戳单调性。
已有 mocap0 目录时拒绝覆盖。源文件会留在 meta/mocap_source.csv，
规范化数据写入 mav0/mocap0/data.csv，并更新 manifest 和 checksums。

导入不进行时间对齐、外参求解或 ground truth 生成。Camera–IMU、IMU–Mocap
外参与时间偏移由后续标定工作流估计；设备与Mocap时间戳不能直接当作同一时钟。

## 4. 输出和验收

```text
session/
  mav0/
    cam0/data/<device_timestamp_ns>.png
    cam0/data.csv
    cam1/ … cam3/                  # 四目入口
    imu0/data.csv
    mocap0/data.csv                # 导入后
  meta/
    manifest.yaml
    cam0_frames.csv …             # 序号、曝光、ISO等原始索引
    raw_imu/accel.csv
    raw_imu/gyro.csv
    image_quality.csv
    events.csv
    aprilgrid.yaml
    recorder_stats.json
    camera_sync_stats.json
    checksums.sha256
```

manifest.yaml 使用JSON语法（兼容YAML 1.2），记录实际配置、软件版本、设备ID、
时钟域与未知项目。aprilgrid.yaml 是项目配置快照，需确认与实际打印尺寸一致。
events.csv 的 START/STOP 使用主机单调时钟相对纳秒，明确标注时钟，不用于传感器
时间对齐；本版不自动识别动作阶段。

EuRoC 相机 CSV 为时间戳、文件名两列；IMU CSV 为时间戳、角速度xyz(rad/s)、
加速度xyz(m/s²)七列。保留原始两条IMU流；仅在共同时间区间内将加速度线性插值
到 gyro 测量时刻，不外推，不跨超过两个标称加速度周期的间隙。
不同时间戳不会因来自同一传输packet而被强制认定为同一测量时刻。

结束后自动检查：彩色PNG尺寸与解码、CSV与文件一致性、设备序号缺口、
重复/乱序时间戳、帧率、IMU频率与间隙、四目配组，并生成SHA256。
大数据包收尾需要读回全部PNG和计算校验和，请等待结束。

退出码0表示数据完整性QA通过；2表示采集已收尾但QA失败；1表示运行或导入错误。
缺帧、写盘失败、明显间隙或队列溢出不会得到QA通过。
设备曝光时间戳参考语义尚未确认，图像质量和动作激励需单独验收，
因此QA通过不等于可以直接发布标定或GT。

离线重新生成 EuRoC 索引和报告：

```bash
.venv/bin/python collect_cam0_imu.py --finalize-only --session-dir /absolute/path/session
# 或
.venv/bin/python collect_fourcam_imu_euroc.py --finalize-only --session-dir /absolute/path/session
```

校验复制后的数据：

```bash
cd /absolute/path/session
sha256sum -c meta/checksums.sha256
```

新入口的格式请使用上述 `--finalize-only` 检查；旧
`validate_oak4p_dataset.py` 面向旧版 session.json 布局。
四路扩展 EuRoC 的具体相机数量/EUCM模型仍需下游 VIO/VSLAM 工具支持。

## 5. 无真机验证

```bash
.venv/bin/python -m unittest test_euroc_capture -v
```

测试使用模拟设备，不连接USB；覆盖单目/四目节点选择、彩色无损写盘、IMU对齐、
序号重复保护、写盘异常、Ctrl+C、Mocap导入及checksum。
真实30 FPS/400 Hz持续性能、FSIN精度、曝光画质和实际设备API行为需接机后验证。
