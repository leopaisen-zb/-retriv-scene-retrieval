# Retriv 实验工作区

维护者：zhuangbie@qq.com

## 目标

`Retriv/` 用于集中管理实验输入、输出、源代码与文档，避免内容分散，便于复现与协作。

## 目录结构

```text
Retriv/
├── input/                # 实验输入数据
│   ├── raw/              # 原始输入
│   └── processed/        # 预处理后输入
├── output/               # 实验输出结果
│   ├── artifacts/        # 中间产物、模型文件、缓存
│   └── reports/          # 结果报告与分析
├── src/                  # 实验源代码
│   ├── experiments/      # 实验入口与脚本
│   ├── pipelines/        # 数据与任务流水线
│   └── utils/            # 通用工具函数
└── docs/                 # 实验文档
    ├── design/           # 设计文档（架构、流程、方案）
    ├── api/              # API 接口说明
    ├── guide/            # 使用指南 / 操作手册
    ├── spec/             # 规范说明（格式、协议、结构）
    └── dev/              # 开发相关（TODO、变更记录）
```

## 约定

- 输入文件放在 `input/`，禁止与输出混放。
- 输出文件放在 `output/`，按实验批次或日期分目录。
- 代码统一放在 `src/`，按职责拆分子目录。
- TODO 与计划统一维护在 `docs/dev/TODO.md`。
- 文档补充流程时优先使用 `mermaid`。
