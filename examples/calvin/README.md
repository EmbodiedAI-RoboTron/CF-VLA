# CALVIN Benchmark（CF-VLA）

本目录用于评测 CF-VLA 在 CALVIN 上的多任务长序列性能。

## 运行前准备
- 安装项目与可选依赖：
  - `pip install -e .`
  - `pip install -e ".[bench-calvin]"`
- 准备 CALVIN 数据与配置，并设置环境变量：
  - `CALVIN_DATASET_PATH`
  - `CALVIN_CONFIG_PATH`
  - `CALVIN_EVAL_LOG_DIR`

## 推荐 checkpoint（full/calvin）
- `./checkpoints/Cf-vla-full（calvin）`

## 启动方式
```bash
# 启服务（示例）
python scripts/serve_policy.py --env CALVIN

# 评测
python examples/calvin/main.py
```

## 说明
- 仓库不分发 CALVIN 数据与模型权重文件。
- 请自行确认数据集与权重的使用许可。
- `eval_sequences.json` 提供可执行模板（默认空列表，可替换为你的任务序列）。
