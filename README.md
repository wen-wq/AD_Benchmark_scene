# 北京大学东门至南门：校外公共道路 World Scenario v2

本工程从 OSM 地图与 OSRM 驾驶路线构造结构化 World Scenario，再生成俯视视频和 NVIDIA Cosmos Transfer 2.5 World Scenario Video (WSV) 原生五相机控制视频。本次改进重点是交通密度、平滑转弯和混合紧急事件。**WSV 是几何与运动条件视频，不是 Cosmos 模型生成的写实画面。**

## 本次场景

- 路线：东门外沿中关村北大街向南，平滑右转进入海淀路，停在南门外；约 969.9 m，仿真约 116.9 s。路线遵循 OSM/OSRM 的机动车道路，OSRM 起点回环与终点门口支路已裁掉。
- 转弯：在 0.5 m 采样的 OSRM 折线附近做 16 m 空间尺度的高斯平滑，再将自车轨迹偏至近似右侧车道中心并按 1 m 重采样。由曲率和最大横向加速度 1.7 m/s² 求速度上限，并以前后两次加速度约束传播，提前规划减速。逐帧结果的最大偏航角速度约 27.8°/s，最大横向加速度约 2.0 m/s²。此处是运动学轨迹，不是轮胎/转向器动力学仿真。
- 交通：68 条规则，包括中关村北大街 3 条近似车道上的 40 辆移动车、海淀路 18 辆对向车和 10 辆路边停车；另有 1 辆连续前车和 4 辆事件加塞车。全场活跃交通参与者中位数约 31、峰值约 43；自车前后各 100 m 范围内车辆中位数约 11、最少 5。规则车辆在车道内用固定速度与生成时刻移动，事件车辆和自车使用逐帧跟车逻辑。
- 紧急事件：本次会话构造的 `llm_events.json` 包含 5 次合成事件：E1 前车急刹，E2 左侧加塞，E3 右侧加塞，E4 路边车辆驶出并入，E5 较慢车辆从右侧近距离并入。加塞横向位置使用五次多项式，位置和朝向连续；自车把已并入车辆作为新的跟车对象。事件不代表真实道路记录。
- 质量门槛：`analyze_collisions.py` 按 0.1 s 逐帧检查车辆 2D 轮廓交叠，任何交叠都让管线失败；`validate.py` 检查五次制动响应、转弯速度规划、路线与 OSM 的距离、车辆数和视频完整性。当前版本交叠帧数为 0。
- 街景参考：Wikimedia Commons 保留 4 张周边普通照片；KartaView 的 2025 年前向行车序列在路线 0–600 m 每隔 25 m 提供 1 帧，共 25 帧（1280×960，单视角，非全景）。所选帧与路线的最大横向偏差为 5.7 m，最大纵向采样误差为 6 m。100 m 清单现有 7/10 个点有参考，700、800、900 m 仍缺；KartaView 在后段查询到的图像约偏离路线 90 m，因此未混入。尚无百度全景 AK。

## 文件

| 文件 | 作用 |
| --- | --- |
| `extract_osm.py` | 从北京 OSM PBF 提取路线走廊道路 |
| `llm_provider.py`, `llm_events.json` | 本次会话事件或 OpenAI 兼容 LLM 生成的事件规格及校验 |
| `pku_pipeline.py` | 路线、速度、交通、跟车、ClipGT、俯视视频 |
| `analyze_collisions.py` | 逐帧车辆 2D 交叠扫描和报告 |
| `native_render.py` | NVIDIA 原生 WSV 五相机渲染 |
| `native_mosaic.py` | 将五路原生视频合成带相机标签的预览 |
| `street_images.py` | Commons 照片采样与可选百度全景补图 |
| `harvest_kartaview.py` | 对公开 KartaView 序列做路线、方向匹配并按 25 m 采样 |
| `validate.py` | 结果检查 |
| `package.py` | 把管线、固定输入与结构化输出打成可复现压缩包 |
| `run_pipeline.sh` | 全流程入口 |
| `output/scenario.json`, `scenario.npz` | 地图坐标、事件、逐帧车辆运动数据 |
| `output/clipgt/*.parquet` | WSV 可读取的相机、车道、路界、自车与车辆轨迹 |
| `output/videos/topdown.mp4` | 俯视视频，1280×720，10 fps |
| `output/native_wsv/.../*.mp4` | 原生五视角视频，1280×720，30 fps |
| `output/videos/five_view_native_mosaic.mp4` | 五视角同屏预览，1920×720，30 fps |
| `output/street_images/manifest.json` | 100 m 站点、照片来源和缺口 |
| `output/street_images/kartaview_manifest.json` | 25 m 前视帧、来源、拍摄时间、位置及朝向误差 |
| `output/street_images/kartaview_contact_sheet.jpg` | 25 帧总览图 |
| `output/collision_report.json`, `validation_report.json` | 几何及交付检查结果 |
| `output/pku_world_scenario_pipeline_v2.zip` | 源码、40 MB 北京 PBF、额外依赖、ClipGT、照片与校验报告；视频单独交付 |

## 复现与模型替换

服务器目录：`/data/tianle/projects/pku-world-scenario-v2`。要求 Python 3.11、`requirements.txt` 的依赖、系统 `ffmpeg`/`ffprobe`，以及服务器安装的 Cosmos Transfer 2.5 WSV 渲染代码。项目的 `vendor/` 含有针对该机器补装的依赖。

```bash
cd /data/tianle/projects/pku-world-scenario-v2
bash run_pipeline.sh
```

默认读取本次会话编制的 `llm_events.json`。调用其它闭源或本地开源 LLM 时，提供一个 OpenAI 兼容 `/chat/completions` 服务：

```bash
LLM_PROVIDER=openai-compatible \
LLM_BASE_URL=https://your-provider.example/v1 \
LLM_MODEL=your-model \
LLM_API_KEY=your-key \
bash run_pipeline.sh
```

本地兼容服务通常不需要 `LLM_API_KEY`。生成的事件由 `validate_events()` 检查数量、顺序、位置和物理参数，再由确定性运动模型执行；不合规或发生车辆交叠的样本需要重新生成/修改。`subtype=curbside_pullout` 表示路边静止车辆驶出，其它左右加塞从相邻车道驶入。当前规则模型将此 subtype 与其它 cut-in 复用同一个横向轨迹函数。

`run_pipeline.sh` 首次运行可下载 OSM 北京 PBF 和 OSRM 路线。公开服务的数据版本会变化；若要精确复现本次坐标，请保留项目里的 `osrm_route.json` 与 `corridor_osm.json`。KartaView 的公开照片 API 不要求密钥，本项目固定检索序列 `10464985` 并在下载后验证位置与朝向；压缩包中也保留了本次采样帧。若配置 `BAIDU_AK`，街景脚本会另外尝试补全缺图点，服务没有全景时仍保持缺口标记。

## 近似与来源

OSM/OSRM 提供道路中心线与路线拓扑；车道宽度、路界、车流、相机标定和紧急事件是合成假设。局部路口使用曲线平滑，尚未建立真实停止线、交通信号相位、转向灯、行人或实测轨迹。KartaView 图像许可为 CC BY-SA 4.0，标注“© Grab and KartaView Contributors”；Commons 照片各自保留页面、作者及许可。全部照片都是单一相机视角，不能当作 360° 全景，也未与仿真逐帧配准。不能根据这份合成场景做实际上路决策。

资料：[OSM 北京数据](https://download.openstreetmap.fr/extracts/asia/china/)、[NVIDIA WSV 文档](https://docs.nvidia.com/cosmos/2.1.0/transfer2.5/world_scenario_video_generation.html)、[NVIDIA 五相机文档](https://docs.nvidia.com/cosmos/2.1.0/transfer2.5/auto-multiview_inference.html)、[KartaView 照片 API](https://kartaview.org/doc/photos)、[KartaView 使用条款](https://kartaview.org/terms)、[百度全景 API](https://lbs.baidu.com/docs/webapi?title=viewstatic%2Fviewstatic-base)。
