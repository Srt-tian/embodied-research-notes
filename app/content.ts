export type ContentKind = "paper" | "experiment" | "engineering" | "timeline" | "artifact";

export type ResearchEntry = {
  slug: string;
  kind: ContentKind;
  title: string;
  subtitle: string;
  date: string;
  tags: string[];
  status: "published" | "draft";
};

export type CodingEntry = {
  slug: string;
  day: number;
  title: string;
  subtitle: string;
  date: string;
  tags: string[];
  score?: number;
  status: "published" | "in-progress";
};

// 只有代码与测试结果经过检查后，练习记录才设为 published。
export const codingEntries: CodingEntry[] = [
  {
    slug: "day17-velocity-network",
    day: 17,
    title: "Flow Matching Velocity Network",
    subtitle: "融合 noised action、timestep 与机器人条件，为完整 action chunk 预测速度场",
    date: "2026-08-05",
    tags: ["PyTorch", "Flow Matching", "Velocity Network", "Action Chunk"],
    status: "published",
  },
  {
    slug: "day16-timestep-embedding",
    day: 16,
    title: "Sinusoidal Timestep Embedding",
    subtitle: "用多频正弦特征编码连续生成进度，并审计归一化时间下的频率尺度",
    date: "2026-08-04",
    tags: ["PyTorch", "Flow Matching", "Time Embedding", "Broadcast"],
    status: "published",
  },
  {
    slug: "day15-flow-matching-path",
    day: 15,
    title: "Flow Matching 直线路径与训练目标",
    subtitle: "从噪声与真实 action chunk 构造 x_t、target velocity，并连接 Euler 采样",
    date: "2026-08-03",
    tags: ["PyTorch", "Flow Matching", "Euler Sampler", "Action Chunk"],
    status: "published",
  },
  {
    slug: "day14-rope-cache",
    day: 14,
    title: "RoPE 与 KV Cache 位置对齐",
    subtitle: "理解 Q/K 旋转如何引入相对位置，并让 cached inference 延续全局绝对位置",
    date: "2026-07-29",
    tags: ["PyTorch", "RoPE", "KV Cache", "Position Encoding"],
    status: "published",
  },
  {
    slug: "day13-kv-cache",
    day: 13,
    title: "Causal Self-Attention with KV Cache",
    subtitle: "缓存历史 K/V，构造绝对位置 causal mask，并验证完整、分块与逐 token 推理一致性",
    date: "2026-07-27",
    tags: ["PyTorch", "KV Cache", "Causal Attention", "Inference"],
    status: "published",
  },
  {
    slug: "day12-transformer-action-policy",
    day: 12,
    title: "Transformer Action Policy",
    subtitle: "将 padded state sequence 编码、汇聚为最后有效状态，并一次预测完整 action chunk",
    date: "2026-07-24",
    tags: ["PyTorch", "Transformer", "Attention Mask", "Action Chunk"],
    status: "published",
  },
  {
    slug: "day11-transformer-encoder",
    day: 11,
    title: "Pre-Norm Transformer Encoder Block",
    subtitle: "串联 Self-Attention、LayerNorm、残差连接与 FFN，并理解梯度直通路径",
    date: "2026-07-23",
    tags: ["PyTorch", "Transformer", "Pre-Norm", "Residual"],
    status: "published",
  },
  {
    slug: "day10-multihead-attention",
    day: 10,
    title: "从零实现 Multi-Head Self-Attention",
    subtitle: "完整走通 QKV 投影、多头拆分、缩放点积、causal mask 与输出合并",
    date: "2026-07-22",
    tags: ["PyTorch", "Transformer", "Self-Attention", "Causal Mask"],
    status: "published",
  },
  {
    slug: "day08-multi-epoch-trainer",
    day: 8,
    title: "多轮训练与 Early Stopping",
    subtitle: "组合单轮训练与评估，记录 loss、保存最佳快照并恢复最优模型",
    date: "2026-07-21",
    tags: ["PyTorch", "Training Loop", "Early Stopping", "State Dict"],
    status: "published",
  },
  {
    slug: "day07-epoch-trainer",
    day: 7,
    title: "完整 Epoch 训练与评估",
    subtitle: "全局 masked MSE、梯度裁剪、train/eval 模式与 Tensor 标量转换",
    date: "2026-07-20",
    tags: ["PyTorch", "Epoch", "Masked Loss", "Gradient Clipping"],
    status: "published",
  },
  {
    slug: "day06-action-policy",
    day: 6,
    title: "端到端 Action Chunk Policy",
    subtitle: "从可变长度 episode 到 Dataset、masked loss 与策略训练闭环",
    date: "2026-07-17",
    tags: ["PyTorch", "Dataset", "Action Chunk", "Masked Loss"],
    score: 9.0,
    status: "published",
  },
  {
    slug: "day05-pytorch-hooks",
    day: 5,
    title: "PyTorch Hook 导读",
    subtitle: "了解 forward hook、Tensor gradient hook 与模型内部观测接口",
    date: "2026-07-16",
    tags: ["PyTorch", "Hook", "Debug", "Level C"],
    status: "published",
  },
  {
    slug: "day04-gradient-debug",
    day: 4,
    title: "PyTorch 梯度诊断",
    subtitle: "定位断梯度、模式错误、梯度范数与裁剪顺序问题",
    date: "2026-07-16",
    tags: ["PyTorch", "Autograd", "Debug", "Gradient Clipping"],
    score: 9.3,
    status: "published",
  },
  {
    slug: "day03-action-mlp",
    day: 3,
    title: "PyTorch Action MLP",
    subtitle: "从机器人 state 预测 action chunk，并完成 loss、反向传播与参数更新",
    date: "2026-07-15",
    tags: ["PyTorch", "MLP", "Autograd", "Action Chunk"],
    score: 9.4,
    status: "published",
  },
  {
    slug: "day02-trajectory-stats",
    day: 2,
    title: "轨迹数据结构与流式统计",
    subtitle: "用 dataclass 对齐机器人轨迹，并以批量 Welford 合并均值与方差",
    date: "2026-07-15",
    tags: ["NumPy", "Dataclass", "Statistics", "Trajectory"],
    score: 9.2,
    status: "published",
  },
  {
    slug: "day01-action-chunks",
    day: 1,
    title: "Action Chunk 切片器",
    subtitle: "从多个 episode 构造定长动作窗口、padding 与 valid mask",
    date: "2026-07-14",
    tags: ["NumPy", "Sequence", "Mask", "Dataset"],
    score: 9.0,
    status: "published",
  },
];

// 后续新增内容时，优先在这里增加记录；页面结构无需重写。
export const researchEntries: ResearchEntry[] = [
  {
    slug: "masquerade",
    kind: "paper",
    title: "Masquerade",
    subtitle: "Learning from In-the-wild Human Videos using Data-Editing",
    date: "2025-08",
    tags: ["Human Video", "Co-training", "Data Editing", "Diffusion Policy"],
    status: "published",
  },
  {
    slug: "pi05-open-world-generalization",
    kind: "paper",
    title: "π0.5",
    subtitle: "A Vision-Language-Action Model with Open-World Generalization",
    date: "2025-04",
    tags: ["VLA", "FAST", "Flow Matching", "OpenPI"],
    status: "published",
  },
  {
    slug: "lingbot-va",
    kind: "paper",
    title: "LingBot-VA",
    subtitle: "Causal World Modeling for Robot Control",
    date: "2026-03",
    tags: ["World Model", "Flow Matching", "MoT", "Async Control"],
    status: "published",
  },
];

export const expansionModules = [
  { code: "02", title: "论文精读", description: "网络结构、Loss、时间对齐与代码证据", kind: "paper" },
  { code: "03", title: "实验记录", description: "训练配置、指标、消融与失败案例", kind: "experiment" },
  { code: "04", title: "工程日志", description: "环境配置、报错定位与可复用命令", kind: "engineering" },
  { code: "05", title: "研究时间线", description: "VLA、WAM 与世界模型方向的演进", kind: "timeline" },
  { code: "06", title: "成果文件", description: "论文、图表、汇报与可下载附件", kind: "artifact" },
] as const;
