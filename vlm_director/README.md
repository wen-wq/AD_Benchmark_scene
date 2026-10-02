# VLM Director baseline

独立的自动驾驶 corner case Director：输入交通场景快照和可选 BEV，输出供后续 scenario executor 使用的高层事件。当前阶段只搭建推理、解析、校验和比较接口，不训练模型，也不执行车辆控制。

```text
Map / traffic generator -> Base scenario snapshot JSON
                       -> Director (rule / Qwen text / Qwen + BEV)
                       -> Scenario modification JSON
                       -> [future] scenario executor / simulator / world model
                       -> [future] driving policy evaluation
```

## 仓库调研与环境

实际代码根目录是 `pku_world_scenario_pipeline_v2/pku-world-scenario-v2/`；下面所有命令都从该目录运行。

- 已有 `llm_provider.py`：文本 LLM 的兼容 `/chat/completions` 请求、固定的五事件 prompt、Markdown fence 移除和 `json.loads`；默认 session 模式读取现有 `llm_events.json`，不是本地模型推理。
- 已有 `pku_pipeline.py::validate_events()`：按固定路线校验五个事件的位置、间距与运动参数，执行规则运动模型并导出轨迹。
- 原代码未包含 Qwen-VL、InternVL、LLaVA 的加载、图像输入或 VLM 推理。
- 新 API wrapper 复用原 HTTP/JSON-mode/`LLM_API_KEY` 的接入方式。没有直接 import `llm_provider.py`，因为它依赖 generator，并会写入原 `llm_events.json`。所有 generator 文件和已有输入输出保持原样。
- 2026-10-01 检查的本地环境：Python 3.12.6、PyTorch 2.5.1 CPU、无 CUDA、无 `transformers`，未发现 Qwen 权重。规则 baseline 和 HTTP 客户端仅需标准库；真实本地 Qwen 推理需另行准备依赖和权重。

本地 Qwen wrapper 使用明确的 `Qwen/Qwen2.5-VL-3B-Instruct` 和 Transformers 4.57.1 接口，参照 [Transformers Qwen2.5-VL 文档](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl) 和 [官方模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。固定版本方便后续复现实验；目前尚未完成真实权重推理验证。

## 新增结构

```text
vlm_director/
  models/vlm_wrapper.py        # Qwen 本地 / 兼容 API / 显式 fixture 回放
  prompts/director_prompt.py   # 两个 VLM baseline 共用的版本化 prompt
  schemas/scenario_schema.py  # 输入、事件契约与静态可行性检查
  baselines.py                # Baseline 0：规则事件
  inference.py                # 三种 baseline 共用的编排入口
  parser.py                   # 严格解析、fence 恢复与早期字段兼容
  cli.py                      # 公共参数、结果持久化、失败诊断
  evaluation.py               # JSON、静态可行性与多样性指标
  requirements-qwen.txt        # 可选本地 Qwen 依赖，不改变原 requirements
  README.md
scripts/
  run_director.py
  evaluate_director.py
  create_director_bev_demo.py
run_director.py                # 根目录入口，与 scripts/run_director.py 等价
examples/director/
  scenarios/urban.json         # 用户提供的最小输入
  scenarios/urban_full.json    # 含前车、邻车、行人、位置与车道拓扑的完整 demo
  responses/cut_in.json        # 显式回放 fixture，不是 VLM 生成结果
  bev/urban_full.png           # 与完整 demo 对齐的合成 BEV 示意图
tests/test_vlm_director.py
```

## 输入格式

最小格式与任务提供的示例一致：

```json
{
  "map": "urban",
  "ego": {"speed": 40, "lane": 2},
  "actors": [{"id": "car_1", "type": "vehicle", "lane": 3, "speed": 35}]
}
```

`speed_unit` 默认 `km/h`；可显式设为 `m/s`。读取后统一转换为 m/s，再传入规则或 VLM。单位不会按数值猜测。Ego 要有 `lane`、`speed`；actor 要有唯一的 `id`、`type`、`speed`，vehicle 还必须有 `lane`。当前支持 `vehicle`、`pedestrian` 两种 actor。行人 `lane` 可省略，但会产生未知起点的可行性告警。`direction` 默认为 `same`，也支持 `opposite`，后者不用于本阶段的车辆事件。

建议同时提供：

- Ego 和所有 actor 的 `s_m`：同一路线上的纵向进度，以米为单位。
- `map.lane_ids`：实际可用的车道 ID。
- `map.adjacent_lanes`：无向邻接对，优先于整数 ID 连续的默认假设。
- `map.lane_width_m`：用于保守的横向加速度估计；省略时假设 3.5 m。
- `scenario_id`、时间戳和 BEV 标定等元数据：允许在输入保留，未来 adapter/executor 需定义其语义。

完整示例见 `examples/director/scenarios/urban_full.json`。BEV 通过 `--bev` 独立传入，只支持本地 PNG/JPEG，单文件不超过 20 MiB。图像需要与 JSON 的时间和 actor ID 对齐；当前不会自动从视频采帧，也不会从图像恢复米制坐标。

**已有 `output/scenario.json` 不是这个输入契约。** 它是路线、统计、事件及轨迹文件引用的导出，没有 ego/actors 快照。当前遇到该格式会明确报错。后续应增加独立 snapshot adapter，从 `scenario.npz` 或 simulator state 获取同一时刻的 ego、actor、车道和坐标，再传给 Director；不会凭导出元数据猜测快照。

## 输出格式与触发语义

每次输出一个事件，`schema_version="1.0"`：

```json
{
  "schema_version": "1.0",
  "event": "cut_in",
  "actor": "car_1",
  "trigger": {
    "type": "distance", "value": 15.0, "unit": "m",
    "reference": "ego", "operator": "lte"
  },
  "behavior": {
    "type": "lane_change", "target_lane": "ego_lane", "duration_s": 3.0
  },
  "parameters": {}
}
```

`actor` 必须已存在于输入。`ego_lane` 表示快照中 ego 的车道，由 executor 在接收事件时解析为真实 ID。所有输出使用 SI 单位。

| event | behavior 必填字段 | 基础边界 |
| --- | --- | --- |
| `cut_in` | `type=lane_change`, `target_lane`, `duration_s` | 相邻、同向、移动中的车辆；时长 2–6 s；五次多项式横移加速度估计 ≤3 m/s² |
| `sudden_braking` | `type=brake`, `deceleration_mps2`, `duration_s` | 同车道、同向、移动中的前车；减速度 -8 至 -0.5 m/s²；时长 0.2–4 s；最终速度非负 |
| `pedestrian_crossing` | `type=cross_lane`, `target_lane`, `speed_mps` | 已存在且不在 ego 车道内的行人；横穿速度 0.5–3 m/s |

Distance trigger 在 **ego 与指定 actor 的绝对纵向路线距离 ≤ value 米** 时触发，不是原 pipeline 的 `distance_m` 路线站点，也不是像素距离或欧氏距离。Time trigger 是 `{"type":"time","value":2,"unit":"s"}`，相对输入快照时刻计时。Executor 应每个事件仅触发一次，并在仿真时间窗结束时报告未触发。若距离当前已经满足阈值，应在接收后首个 tick 触发。

`parameters` 在 v1.0 必须是空对象，禁止未经校验的额外控制字段。解析器支持早期 `corner_case_type` → `event`、`target_actor` → `actor`；trigger 和 behavior 仍必须是结构化对象，不会猜测自由文本的含义。合法的 JSON-string `parameters="{}"` 可无损转换。后续增加事件或参数应升级契约与校验器。

## 如何运行

以下命令在 PowerShell、bash 中均可按单行执行。默认输出到当前工作目录的 `outputs/`。

规则 baseline 无需安装原 scenario generator 的依赖：

```shell
python run_director.py --scenario examples/director/scenarios/urban.json --model rule
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model rule --event cut_in --output-dir outputs/demo_cut_in
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model rule --event sudden_braking --output-dir outputs/demo_braking
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model rule --event pedestrian_crossing --output-dir outputs/demo_crossing
```

`--event auto` 在合格的 actor/event 候选中按 `--seed` 采样，不新增 actor。没有可用 actor 时明确失败。自动采样对候选等概率，不保证三个类别等概率。

Baseline 1 / 2 可以使用相同 Qwen checkpoint，仅图像输入不同。推荐在独立 Python 环境安装可选依赖：

```shell
python -m pip install -r vlm_director/requirements-qwen.txt
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model qwen --baseline 1
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model qwen --baseline 2 --bev examples/director/bev/urban_full.png
```

`--model qwen` 默认对应 Qwen2.5-VL-3B-Instruct。首次真实运行会按 Hugging Face 常规方式加载/下载权重；已有本地权重时使用 `--model-id path/to/checkpoint --local-files-only`。`--device auto` 使用可用 CUDA，否则 CPU；CPU 使用 FP32，内存和耗时较高。当前只实现 Qwen2.5-VL 本地架构，不接受用其他架构 checkpoint 直接替换本地类。

也可接已经部署的 Qwen 兼容服务，Director 端不需要 torch/transformers：

```shell
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model qwen --baseline 1 --backend api --base-url http://localhost:8000/v1 --model-id served-qwen
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model qwen --baseline 2 --backend api --base-url http://localhost:8000/v1 --model-id served-qwen --bev examples/director/bev/urban_full.png
```

`--model-id` 必须匹配 server 的真实模型名。Baseline 2 要求服务端部署的是支持图像的 VLM；客户端会发送实际 image data URL，而不是本地路径字符串。认证沿用 `LLM_API_KEY`，URL 可通过 `LLM_BASE_URL` 设置。默认请求 JSON mode；只有服务明确指出不支持 `response_format/json_object` 时才去掉该字段重试，可通过 `--no-json-mode` 主动禁用。其他 HTTP 错误不会用规则结果替代。

`--temperature 0` 默认贪心生成，`--max-new-tokens 512` 限制输出长度，`--seed 0` 用于采样复现。API 服务是否遵守 seed、GPU 跨版本确定性需要另行验证。

环境没有权重或服务时，可显式验证文本/图像路径的 fixture 回放：

```shell
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model replay --baseline 1 --response-file examples/director/responses/cut_in.json --output-dir outputs/demo_replay_text
python scripts/run_director.py --scenario examples/director/scenarios/urban_full.json --model replay --baseline 2 --response-file examples/director/responses/cut_in.json --bev examples/director/bev/urban_full.png --output-dir outputs/demo_replay_vision
```

Replay 明确记录 `model.is_mock=true`，不调用 VLM、不分析图像，只有接口测试意义。BEV 示例是人工合成的对齐示意图，非 simulator 输出；可用 `python scripts/create_director_bev_demo.py` 重建（需要 Pillow）。这些回放不能用于报告 VLM 的生成质量。

每次运行保存：

- `<stem>_prediction.json`：纯高层事件，供 executor 使用，仅在 schema 合格且无已知可行性失败时生成。
- `<stem>_report.json`：baseline、模型/是否 mock、生成参数、JSON/schema 状态、可行性、错误、成功调用耗时、输入路径和 SHA256。
- `<stem>_raw.txt`：模型原始响应；模型调用前失败时为空。

解析器拒绝重复键、NaN/Infinity、多个对象、数组、解释性文字及未知字段；允许单个 JSON code fence，但会记录 strict JSON 不合格、fence 恢复成功。退出码 0 表示已输出事件，1 表示样本失败，2 表示 CLI 配置错误。同输出目录、同 stem 会覆盖上次产物；失败时会移除上次同名 prediction，避免 executor 读取旧结果。批量比较请给不同 baseline 使用不同输出目录。

## 比较脚本

```shell
python scripts/evaluate_director.py --scenarios examples/director/scenarios --model rule --samples-per-scenario 12 --output-dir outputs/rule_comparison
python scripts/evaluate_director.py --scenarios examples/director/scenarios/urban_full.json --model qwen --baseline 1 --samples-per-scenario 12 --temperature 0.3 --output-dir outputs/qwen_text
python scripts/evaluate_director.py --scenarios examples/director/scenarios/urban_full.json --model qwen --baseline 2 --bev-dir examples/director/bev --samples-per-scenario 12 --temperature 0.3 --output-dir outputs/qwen_vision
```

API 比较时追加与单次推理相同的 `--backend api --base-url ... --model-id ...`。三个 baseline 应使用同一份 snapshot 集、同一 seed 计划及相同 Qwen checkpoint/生成参数；重复贪心生成不会自然产生多样性。脚本复用一个 model 实例，每个场景的 seed 从 `--seed` 递增。

输出 `evaluation_report.json` 和每次采样的预测/原文/诊断。BEV 按 scenario stem 匹配 `.png/.jpg/.jpeg`；缺图、输入错误、API 错误和解析失败均保留在总尝试次数分母中。

| 指标 | 定义 |
| --- | --- |
| `strict_json_valid_rate` | 原始响应直接是唯一 JSON object 的比例；fence 不算 |
| `json_object_valid_rate` | 包含 fence 恢复后能解码为 object 的比例 |
| `schema_valid_rate` | 满足高层事件契约的比例 |
| `static_feasibility_pass/unknown/fail_rate` | 静态约束结果占所有尝试的比例；未完成 schema/调用的样本记 not_checked |
| `export_success_rate` | 无已知失败且已导出事件的比例；可能包含 unknown |
| `diversity` | 已导出事件的三类计数、类别数、类别 Shannon 熵和目标 actor 数 |
| `simulation_verified_rate` | 当前固定 0，只有接入 simulator 后才能产生该结果 |

出现 mock/replay 的汇总明确标记 `interface_test_only=true`。`export_success_rate` 不能当作物理可行率；多样性也不等于驾驶策略失败率。

## 已完成 demo 与验证范围

完整 demo 的输入是 40 km/h 的 ego（lane 2, s=100 m）、35 km/h 的 `car_1`（lane 3, s=120 m），另有同车道前车与行人。上述 cut-in 命令生成 `outputs/demo_cut_in/urban_full_prediction.json`，内容如输出格式示例：`car_1` 在距 ego 15 m 时，以 3 s 横移进入 ego 车道。检查结果为 JSON/schema 合格、静态 feasibility=pass、simulation_verified=false。

2 个场景 × 12 个 seed 的规则比较已经运行：24/24 输出 JSON/schema 合格；12 次 static pass、12 次 unknown（最小输入缺位置）；三类计数为 cut-in 17、急刹 4、行人横穿 3。仅说明本地规则接口可运行，不代表已完成真实 VLM 对比实验。

验证命令：

```shell
python -m unittest discover -s tests -p test_vlm_director.py -v
python -m compileall -q vlm_director scripts run_director.py
```

测试覆盖单位转换、三类规则事件、缺少 actor、seed 复现、解析失败、未知字段/低层命令拒绝、物理矛盾、车道拓扑、触发可达性、图像/文本隔离、失败产物、CLI、真实 localhost HTTP 字节传输及 JSON-mode 回退、Qwen completion token 截取。HTTP 的模型响应和 Qwen processor/model 是 fixture/double，只验证接口。真实 checkpoint 加载、显存、真实图像理解和真实生成质量尚未验证。

## 后续接 simulator / world model 的接口

1. **Snapshot adapter**：按指定仿真时刻输出该输入契约；稳定映射 actor ID 和 lane ID；提供速度单位、同向/对向关系、路线 `s_m`、车道邻接。现有轨迹导出必须先经 adapter。
2. **BEV provider**：返回与 snapshot 同帧的图像、时间戳、坐标轴、米/像素变换及 actor 标注约定。Director 目前只将图像传给模型，不做几何标定。
3. **Event executor**：接收 v1.0 事件，在接收时绑定 `ego_lane`；按明确的纵向距离或相对时间进行一次性触发；将 lane_change、brake、cross_lane 转为 simulator 能力。执行器负责路径、车辆动力学、行人起终点和当前时刻的可达性，不由 Director 控制 ego。
4. **Execution feedback**：返回 event ID、目标是否存在、accepted/rejected 及原因、是否触发/完成、触发时刻和位置、动力学/碰撞检测结果。这样才能把 static feasibility 替换成真实执行可行性统计。
5. **Policy evaluation**：传入 policy、seed、仿真时间窗和成功标准，返回碰撞、越界、TTC、舒适性等约定指标；ego 响应由受评策略产生，不能预设在 Director 输出里。

原固定五事件格式与新输出不兼容：原 `distance_m` 是路线站点，新 `trigger.value` 是 ego/actor 距离阈值；`left_cutin/right_cutin/lead_vehicle_hard_brake` 需要 adapter 映射；当前原 executor 没有 pedestrian crossing 支持。不能直接覆盖 `llm_events.json`，也不能把新的距离阈值直接塞进原事件位置字段。

## 当前 baseline 限制

单次只生成一个事件，仅支持三类；没有训练、重试搜索、critic、多事件调度、actor spawn 或低层控制。缺少 suitable actor 时规则 baseline 失败，VLM 不合格输出也会失败。

静态 pass 只检查必要的输入与粗略运动约束，不包含碰撞盒、空隙、路界、障碍物、交通灯、摩擦、遮挡和车辆动力学。缺少位置/拓扑/行人起点时为 unknown；距离阈值在恒速状态下不可达也为 unknown，需要 executor 继续判断。整数车道连续、默认 lane width 和默认同向关系均是 baseline 假设。当前事件要求目标在快照中位于 ego 前方，不覆盖后车超车等事件。

真实文本/图像 VLM 质量、事件物理可行率和策略评估结果，需要在模型服务及 simulator 接入后测量；当前 demo 和回放只完成基础接口与 pipeline 验证。
