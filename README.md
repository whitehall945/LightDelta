# LightDelta

单卡 H100 上面向 Qwen3.8-27B 的 GDN CUDA 后端与 vLLM V1 集成。

采用 C++17/CUDA、CMake、scikit-build-core 和 Python 包布局。

H100 实测已覆盖 10 个配置、60 组服务负载、600 个请求。最终自实现 D/E
链路中，`full` 相对未融合参考服务的吞吐提升为 3.39～6.96 倍；单独融合
GDN 计算加速 3.04～6.79 倍。实验条件和完整消融见
[性能报告](docs/performance.md)。首轮官方 D/E 数据单独保存在
`/home/tiger/LightDelta-first-round`。

```text
python/lightdelta/
  reference/       未融合 Torch GDN
  ops/             torch.library 注册
  integration/     vLLM GDN 层接入
  runtime/         LightDelta 图执行与 MTP 运行时
  backend.py       统一计算边界和状态访问策略
csrc/              CUDA 融合内核、状态拷贝、C++ dispatcher
include/lightdelta/ 原生接口声明
configs/           进程启动时确定的消融配置
tests/             数学、状态池、MTP 边界和 CUDA Graph 正确性
benchmarks/        算子、流式服务和报告生成
docs/              接口约定、实验复现、环境与实测结果
results/           最终矩阵数据、算子测量和生成对照
```

固定 PyTorch 2.13.0+cu129、vLLM 0.29.0+cu129、FlashInfer 0.6.18。
完整环境锁定在 `requirements/`。当前 Worker 使用 H100 80GB 和 CUDA 12.9。

首次安装（在 GPU Worker 上执行）：

```bash
~/miniconda3/bin/conda env create -f environment.yml
source scripts/activate.sh
python -m pip install -r requirements/runtime-cu129.txt
python -m pip install -e '.[dev]' --no-build-isolation
```

GPU 上开发和启动服务：

```bash
mlx worker login
cd /home/tiger/LightDelta
source scripts/activate.sh
lightdelta inspect-model
lightdelta serve --config configs/full.toml
# 一次完成原生对照、算子测量、各配置服务消融和报告
lightdelta experiment
```

服务监听 Worker 的 `127.0.0.1:8000`，提供 vLLM 的 OpenAI 兼容流式接口。
模型路径由 `LIGHTDELTA_MODEL_PATH` 指定，默认
`/mnt/bn/search-nlp-us/wanghenglong/Qwen3.8-27B`。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lightdelta","messages":[{"role":"user","content":"介绍一下 CUDA"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":128,"stream":true}'
```

每次更换配置重新启动进程。不存在请求期间切换后端或静默回退：

| 配置 | GDN 计算 | 状态访问 | 图执行 | MTP 候选数 |
| --- | --- | --- | --- | --- |
| `reference`（A） | Torch 分解 | 显式搬运 | 关 | 0 |
| `fusion`（B） | CUDA 融合 | 同一显式搬运路径 | 关 | 0 |
| `indexed`（C） | CUDA 融合 | 状态池索引 | 关 | 0 |
| `graph`（D） | CUDA 融合 | 状态池索引 | 开 | 0 |
| `full`（E） | CUDA 融合 | 状态池索引 | 开 | 3 |

`full_no_graph`、`full_packed` 分别移除图执行与索引访问；`mtp1`、`mtp2`
单独改变投机深度。`full_prefill` 在 E 上启用 prefill 状态初始化与拷贝融合。
本次测量中，该 prefill 改动的吞吐增益不足 1%，TTFT 变化不一致，因此
推荐使用 `full` 作为服务配置，保留 `full_prefill` 用于显式消融。
长 prefill 核心、Attention、FFN 和服务调度共用固定版本的上游实现；
torch.compile、前缀缓存在所有实验中关闭。

验证和复现实验均在 Worker 的 `lightdelta` 环境执行：

```bash
python -m pytest -q
# 完整实验统一从一个入口运行
lightdelta experiment
```

接口、配置和实验入口分别位于 `python/lightdelta/`、`configs/` 和
`lightdelta experiment`；实测数据、逐项消融和图表见
[performance.md](docs/performance.md)。
# LightDelta
