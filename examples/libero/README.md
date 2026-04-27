# LIBERO Benchmark（CF-VLA）

本目录用于评测 CF-VLA 在 LIBERO 上的任务执行能力。

## 运行前准备
- 安装项目与可选依赖：
  - `pip install -e .`
  - `pip install -e ".[bench-libero]"`
- 根据你的环境准备 LIBERO 及相关依赖。

## 推荐 checkpoints
- phase2（libero）:
  - `./checkpoints/Cf-vla-phase2（libero）`
- full（libero）:
  - `./checkpoints/Cf-vla-full（libero）`

## 启动方式
```bash
# 启服务（示例）
python scripts/serve_policy.py --env LIBERO

# 评测
python examples/libero/main.py
```

## 说明
- 仓库不分发 LIBERO 数据与权重文件。
- 请自行确认数据集与权重的许可证。
