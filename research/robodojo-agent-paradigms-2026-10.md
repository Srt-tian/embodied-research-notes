# RoboDojo Agent 三种技术范式：控制、反思与模型进化

> 研究笔记｜2026-10-19｜主题：Embodied Agent / RoboDojo / RSI  
> 范围：概念与公开架构总结。各项目成绩与开源状态随时间变化，以项目官方仓库为准。

## 0. 核心结论

目前值得沿着三个方向理解 RoboDojo 上的 Agent 系统：

1. **Online Coding / Skill Agent（EmbodiedEvo）**：大模型参与 Episode 内高层决策，生成、调用或修改代码式 Skill；Python/Robot Runtime 执行原生动作。
2. **Recursive Self-Harness（PhysicalRSI）**：System 1 使用既有 Harness 执行；System 2 主要跨 Rollout 反思、提出修改、验证并继承新 Harness，在线不需要逐步调用反思 Agent。
3. **Joint Agent–Harness–Model RSI（WorldScape-R²SI，SenseNova-RoboRSI 的模型改进方向）**：除外部 Skill/Harness 的演化外，也尝试让验证过的 Rollout 支持反应式模型参数更新。WorldScape-R²SI 的部分 Rollout 仍由在线 Inner Agent 调用大模型决策，另一部分可由反应式模型直接执行。

**分类提醒**：这三种是为了理解现有代表系统而作的“系统范式”归纳，并非彼此排斥的三类算法。Coding-as-Policy 与 Skill Agent 高度重叠；Harness RSI 和 Model RSI 是不同的“更新对象”，可与任意执行架构组合。

## 1. 两条正交轴

| 轴 | 选项 | 判断问题 |
|---|---|---|
| Episode 内执行 | 在线 LLM Agent + Skills / 直接 Reactive Policy | **谁在 Rollout 中决定下个子任务与动作？** |
| Episode 间改进 | 无更新 / Harness 更新 / Model 参数更新 | **经验最后被写到哪里？** |

### 总览

```text
                    EXECUTION (episode 内)
Task + RGB + State ──────────────────────────────────────────────────────────┐
                    │                                                       │
        Online LLM / Coding Agent                                    Reactive VLA / WAM
        decide subgoal / skill / code                                observation → actions
                    │                                                       │
              Code / Skill / Tool                                            │
                    └──────────────────────┬────────────────────────────────┘
                                           ↓
                                 Robot Runtime / Rollout
                                           ↓
                                 Trajectory + Feedback
                                           ↓
                    RECURSIVE IMPROVEMENT (episode 间)
                           ┌───────────────┴────────────────┐
                           ↓                                ↓
                   Harness update                    Model training
                 Skills / Memory / Code              θ_k → θ_(k+1)
                           └───────────────┬────────────────┘
                                           ↓
                               Independent evaluation
```

## 2. 范式一：Online Coding-as-Policy / Dynamic Skills

**代表：EmbodiedEvo。**

**角色分工**

| 部件 | 做什么 | 不做什么 |
|---|---|---|
| Coding Agent (LLM) | 看观测、理解任务、决定阶段/手臂/目标、写或修改 Python、调用 Skill | 不必为每个 Native Action 分别推理 |
| Python Code / Skill | 用循环、分支、插值、条件检查把目标转换成多次 API 调用 | 没有额外 LLM 时，不会自动获得开放式语义推理 |
| Robot Runtime | 接收 `step(action)`、推进动力学、返回状态 | 不负责自主任务理解 |

**一次 Episode 的关键时序**

```text
LLM observe → write/run code C1 → [step(action) × n1] → result
LLM observe → revise/run C2    → [step(action) × n2] → result
...
```

(n_k) 是可变的，不等于预设的 VLA Horizon；代码内部也可以在连续调用 `step` 之间读取观测并执行既定的程序级条件判断。**15 次代码执行不是 15 次 Native Actions，也不能直接等同于 15 次 LLM forward。**

以论文展示的堆叠案例为例：15 次代码执行共消耗 395 个 Native Actions；Agent 决定抓取顺序、目标与何时根据视觉重新对齐，Skill 实现下降、闭合、抬升及搬运等多步控制。

**记忆与迁移**：可复用代码为 Skills；前置条件、失败教训、诊断规则为 Lessons。Playground 探索后将经验迁移到独立 Test；这种经验迁移主要不是参数训练。

**本质判断**：Code 本身可视为“动态生成的 Skill”。创新在于 Agent 可以构造/修改 Skill 并复用跨场景经验，而非消除高层 LLM 决策。

**挑战**：物体定位与坐标变换、语义判断时的 LLM 延迟、长程序期间反馈不足、接触控制、动态代码安全。

## 3. 范式二：Recursive Self-Harness

**代表：PhysicalRSI。**

**System 1 — Execution**：在已提交的 Harness 版本下按当前观测和任务执行，Harness 包含可调用策略、控制代码、Skills、Memory、Tools 和必要的路由机制。执行可以使用 π0.5、稀疏视觉记忆或 code-policy 等路径；**不要求 System 2 每一步调用 LLM**。

**System 2 — Improvement Agent**：读取成功/失败 Rollout 和既有 Harness，对问题进行归因，提出和改写候选，然后独立验证、选择、继承。

```text
H_k → rollout τ_k → evidence / failure attribution
                         ↓
                propose H'_1, H'_2, ...
                         ↓
          contract checks + controlled evaluation
                         ↓
                  select / reject / inherit
                         ↓
                        H_(k+1)
```

形式上可写：

```text
H_(k+1) = Select({ H_k, Vary_Agent(H_k, evidence_k) }, evaluation)
θ_(k+1) = θ_k                    # 纯 Self-Harness 配置
```

**Agent 是反思迭代者，不是默认的逐步动作决策器。** 某些实现可能在 Episode 开始有 Skill Selection 或初始任务计划，不能据此误写为每个 timestep 都有 Agent 推理。

**为什么“验证”重要**：LLM 认为代码修好了，不代表机器人任务成功。需版本冻结、相同预算比较、未参与修改的评估场景、失败可回滚；否则容易过拟合某一条失败轨迹。

**能更新什么**：状态记忆、技能前置条件、路由、工具组合、控制程序、验证逻辑；原则上不直接修改底层 VLA 权重。

**挑战**：失败因果归因、多候选评估的 Rollout 成本、经验跨任务泛化以及“自我修改改进器”与“仅改执行 Harness”的界限。

## 4. 范式三：Joint Agent–Harness–Model RSI

**代表：WorldScape-R²SI；SenseNova-RoboRSI 的模型级迭代方向。**

这里不能简单说成“一个纯粹离线训练 VLA 的 Agent”。WorldScape-R²SI 区分：

- **Inner Robot Agent**：在 Agent + Skills 路径中，大模型可在 Rollout 期间按需做任务/技能级决策；动作仍由技能或反应式策略执行。
- **Direct Reactive Policy 路径**：可跳过外部在线 LLM，由 WSP 或 Simate-beta 等策略直接观察并输出动作。
- **Outer Recursion Agent**：跨 Rollout 评估、修订 Skills/Harness，并在具备数据和训练链路时推动 Reactive Model 改进。

```text
                      ┌─ Online Inner Agent (LLM) → Skill/Code ─┐
Task + Observation ───┤                                         ├─→ Rollout
                      └─ Direct Reactive Model (no outer LLM) ─┘        │
                                                                        ↓
                                                           Evidence + Verifier
                                                                        ↓
                                      ┌─────────────────────────────────┴──────────┐
                                      ↓                                            ↓
                                 Update Harness                             Train θ
                                      └─────────────────────────────────┬──────────┘
                                                                        ↓
                                                                 Re-evaluate
```

### Reactive Model、VLA、WAM 的区别

- **Reactive** 指执行模式：策略直接根据观测和条件产生动作，并按控制时序重复调用；不表示模型一定无记忆、无内部规划。
- **VLA** 指视觉语言动作模型，例如 π0.5：`(RGB, State, Instruction) → Action Chunk`。
- **WAM** 指联合建模世界/视频未来与动作的模型；**WSP（WorldScape Policy）属于 WAM**，也能充当 Reactive Controller。
- **Simate-beta** 在项目中属于反应式模型；公开资料不足以确定它完整网络是否满足特定 VLA/WAM 定义。
- 无外部 LLM 决策不代表 WAM 内部完全没有 VLM 或潜在规划。

### 从经验到模型参数

设 `τ = (o_t, a_t, feedback)_t`，需要先由验证器确认轨迹质量，再构造监督与训练数据：

```text
verified rollouts → curate data → train θ_(k+1) → independent evaluation
```

训练对象可以是高层 Skill 决策器或底层连续动作 VLA/WAM。**这两者的标签、Loss 和时序完全不同**。语言动作分类与连续 Action Chunk Flow Matching 只是可行设计示例，不应当作 R²SI 已公开的具体损失配方。

验证建议采用 **2×2 消融**：

| | 原 Harness H₀ | 新 Harness H₁ |
|---|---|---|
| 原模型 θ₀ | A Baseline | B Harness only |
| 新模型 θ₁ | C Model only | D Joint |

需对齐交互预算与训练数据，避免把 Harness 的收益误判成模型权重更新。

## 5. 三范式对照：最值得记忆的一张表

| 问题 | EmbodiedEvo | PhysicalRSI | WorldScape-R²SI |
|---|---|---|---|
| Rollout 中 LLM 决策？ | 是，高层分段决策 | System 1 通常自主执行；System 2 不逐步参与 | Agent 路径是；Direct Reactive 路径不需要外部 LLM |
| 动作是谁下发？ | Code/Skill → Runtime | Harness / VLA / Code Policy | Skill 或 Reactive Model |
| Agent 主要产物 | 当前控制程序、可复用 Skills/Lessons | 新 Harness 候选 + 评估记录 | Skills/Harness 候选及模型迭代数据/过程 |
| 更新发生在哪？ | Episode 内修订 + 跨场景知识迁移 | 主要跨 Rollout | 在线执行 + 跨 Rollout / 模型训练 |
| 默认需要更新权重？ | 否 | 否 | 部分 Model RSI 路线需要 |
| 成功的关键 | 代码是否正确执行并随反馈修正 | 修改是否通过独立验证 | 新 Harness 与新模型是否真实泛化 |

## 6. 对双臂 Piper 的启示

现有三路 RGB、30 Hz、实测 EEF 锚点和 EEF Delta → IK → Qpos 的链路很适合将问题分层：

1. **Online Agent（可选）**：决定抓哪一个、选左/右臂、何时使用抓取/搬运/验证 Skill。
2. **System 1**：受约束的 EEF Skill 或 π0.5/VLA，负责实际机器人运动；明确 Base/EEF 坐标系及双臂碰撞、速度、关节限制。
3. **System 2**：在 Rollout 后检视真实 state、控制指令与视觉反馈，修订 Skill/Harness；使用独立场景验证。
4. **Model Update（后续）**：只有当积累了验证过的示范和规范化动作标签时才启动后训练；区分高层决策监督与低层连续动作监督。

建议研究顺序：先建立 **固定 System 1 + 自动评估 + Harness 迭代**，再验证 Online Agent 是否增加收益，最后加入模型权重更新。这便于分别测量三种能力来源。

## 7. 待核实事项与证据边界

- 不应把项目方榜单自报分数视为已认证结果；榜单会更新。
- WorldScape Policy 2.0 模型资源公开，不等于 R²SI 完整 Agent/RSI 系统已经完整开源。
- 不能仅凭模型进化示意图断言使用了特定 BC、Flow Matching 或 RL 训练损失。
- Rollout 中在线 Agent 的决策次数、频率、Token 开销需要具体日志核算；一次代码执行不等于一次模型调用。
- 真机探索需要独立的安全检查、Action API 沙箱和人工急停机制。

## 8. 项目与阅读入口

- [EmbodiedEvo — GitHub](https://github.com/Einsia/EmbodiedEvo) ｜[论文](https://arxiv.org/abs/2610.18867)
- [PhysicalRSI — GitHub](https://github.com/PhysicalRSI/PhysicalRSI) ｜[Architecture](https://github.com/PhysicalRSI/PhysicalRSI/blob/main/docs/architecture.md)
- [WorldScape Policy — GitHub](https://github.com/manifoldai-research/WorldScape-Policy)
- [RoboDojo 官方榜单](https://robodojo-benchmark.com/leaderboard)

---
**一句话总结**：EmbodiedEvo 解决 **Agent 怎么在线组织行为**；PhysicalRSI 解决 **Agent 怎么改进外部执行系统**；WorldScape-R²SI 的联合路线进一步研究 **怎么把执行经验变成可验证的 Skill 与模型能力**。
