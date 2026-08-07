import Link from "next/link";
import { SiteHeader } from "../../site-header";

const datasetCode = `# Dataset 单个样本
{
    "condition": condition,      # (C,)
    "actions": actions,          # (H,A)
    "action_mask": action_mask,  # (H,)
}

# DataLoader 自动堆叠
condition = batch["condition"]      # (B,C)
actions = batch["actions"]          # (B,H,A)
mask = batch["action_mask"]         # (B,H)`;

const objectiveCode = `batch_size = actions.shape[0]

# 每次训练现场生成，不存入 Dataset
noise = torch.randn_like(actions)                    # (B,H,A)
timesteps = torch.rand(
    batch_size,
    device=actions.device,
    dtype=actions.dtype,
)                                                     # (B,)

t = timesteps[:, None, None]                         # (B,1,1)
noised_actions = (1.0 - t) * noise + t * actions     # (B,H,A)
target_velocity = actions - noise                    # (B,H,A)

predicted_velocity = model(
    noised_actions,
    timesteps,
    condition,
)                                                     # (B,H,A)`;

const lossCode = `squared_error = (predicted_velocity - target_velocity).square()
expanded_mask = mask.unsqueeze(-1)                   # (B,H,1)
valid_element_count = mask.sum() * actions.shape[-1]
loss = (squared_error * expanded_mask).sum() / valid_element_count`;

const completedCode = `def build_flow_matching_path(actions, noise, timesteps):
    t = timesteps[:, None, None]                     # (B,1,1)
    noised_actions = (1.0 - t) * noise + t * actions
    target_velocity = actions - noise
    return noised_actions, target_velocity


def masked_velocity_mse(predicted_velocity, target_velocity, mask):
    squared_error = (predicted_velocity - target_velocity) ** 2
    expanded_mask = mask.unsqueeze(-1)               # (B,H,1)
    error_sum = (squared_error * expanded_mask).sum()
    valid_count = mask.sum() * predicted_velocity.shape[-1]
    return error_sum / valid_count


def flow_matching_loss(model, actions, condition, mask, *, generator=None):
    batch_size = actions.shape[0]
    noise = torch.randn(
        actions.shape,
        generator=generator,
        device=actions.device,
        dtype=actions.dtype,
    )
    timesteps = torch.rand(
        batch_size,
        generator=generator,
        device=actions.device,
        dtype=actions.dtype,
    )
    x_t, target_velocity = build_flow_matching_path(
        actions, noise, timesteps
    )
    predicted_velocity = model(x_t, timesteps, condition)
    return masked_velocity_mse(
        predicted_velocity, target_velocity, mask
    )`;

export default function Day18FlowMatchingObjective() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 18</span></div>
      <div className="coding-note-title"><span>DAY 18 · 2026-08-05</span><h1>Training<br />Objective</h1><p>把真实观测与 action chunk 变成 Flow Matching 训练样本：随机采样噪声和生成时间，构造 noised action、目标速度与 masked loss。</p></div>
      <div className="coding-score"><span>OBJECTIVE · COMPLETE</span><b>05</b><i>core TODOs</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#pipeline">01 / 完整数据流</a><a href="#dataloader">02 / DataLoader 边界</a><a href="#sample">03 / 单个训练样本</a><a href="#batch">04 / Batch 与 B</a><a href="#velocity">05 / 目标速度</a><a href="#loss">06 / Masked Loss</a><a href="#paths">07 / 非直线路径</a><a href="#summary">08 / 最终总结</a><a href="#implementation">09 / 完成实现</a><a href="#review">10 / 答题复盘</a></aside>
      <div className="coding-note-content">
        <section id="pipeline"><span className="record-label">01 · END-TO-END DATA FLOW</span><h2>Day 15–17 在这里汇成一个 loss</h2><div className="shape-callout"><span>TRAINING RELATION</span><b>(observation, x_t, t) → velocity</b><p>观测决定要生成什么动作，<code>x_t</code> 表示当前含噪动作位置，<code>t</code> 表示生成进度；网络输出与 action chunk 同 shape 的速度。</p></div><div className="equation">actions + noise + random t → x_t + target velocity → model → masked MSE</div><p>Day 15 定义路径与速度，Day 16 编码 timestep，Day 17 构建 velocity network；Day 18 负责把三者组合成一次可反向传播的前向训练目标。</p></section>

        <section id="dataloader"><span className="record-label">02 · DATALOADER BOUNDARY</span><h2>DataLoader 只加载真实数据</h2><pre className="code-block"><code>{datasetCode}</code></pre><p>Dataset 提供当前观测条件、真实 action chunk 和有效位置 mask。到 episode 尾部时，chunk 只在本 episode 内取动作，其余位置 padding 并标为无效，不能读取下一个 episode。</p><div className="insight"><span>WHY NOT PRECOMPUTE?</span><p><code>noise</code>、<code>timesteps</code>、<code>noised_actions</code> 与 <code>target_velocity</code> 应在训练函数中现场生成。同一条真实数据在不同 epoch 会遇到不同起点和路径位置，从而覆盖更完整的速度场。</p></div></section>

        <section id="sample"><span className="record-label">03 · ONE FM SAMPLE</span><h2>一个真实样本如何变成速度回归样本</h2><p>单个样本先有条件 <code>c: (C,)</code> 与真实动作 <code>x₁: (H,A)</code>，再独立采样噪声 <code>x₀: (H,A)</code> 和标量 <code>t∈[0,1]</code>。</p><div className="equation">x_t = (1 − t)x₀ + tx₁　　v* = dx_t/dt = x₁ − x₀</div><p>于是得到监督关系：</p><div className="shape-callout"><span>ONE SAMPLE</span><b>c + x_t + t → v*</b><p>当前观测、当前含噪动作和生成进度共同决定这一位置应该往哪里移动。</p></div><pre className="code-block"><code>{objectiveCode}</code></pre></section>

        <section id="batch"><span className="record-label">04 · BATCH ALIGNMENT</span><h2>三个输入里的 B 是同一批样本</h2><div className="table-wrap"><table><thead><tr><th>Tensor</th><th>Shape</th><th>第 b 项含义</th></tr></thead><tbody><tr><td>condition</td><td>(B,C)</td><td>第 b 个样本的当前观测特征</td></tr><tr><td>actions / noise / x_t</td><td>(B,H,A)</td><td>第 b 个样本的动作块、噪声与路径位置</td></tr><tr><td>timesteps</td><td>(B,)</td><td>第 b 个样本独立采样的生成时间</td></tr><tr><td>target / prediction</td><td>(B,H,A)</td><td>第 b 个动作块每个元素的速度</td></tr><tr><td>mask</td><td>(B,H)</td><td>第 b 个动作块的有效未来位置</td></tr></tbody></table></div><p><code>condition[b]</code>、<code>noised_actions[b]</code>、<code>timesteps[b]</code> 与 <code>target_velocity[b]</code> 必须一一对应。概念上是多个单样本堆叠，实际实现通常是 DataLoader 先形成 batch，再一次性向量化采样整批 noise 和 timestep。</p></section>

        <section id="velocity"><span className="record-label">05 · TARGET VELOCITY</span><h2>为什么不是 actions − noised_actions</h2><p>直线路径对生成时间求导后，速度恒为：</p><div className="equation">dx_t/dt = x₁ − x₀ = actions − noise</div><p><code>actions - noised_actions</code> 表示从当前位置到终点的剩余位移：</p><div className="equation">x₁ − x_t = (1 − t)(x₁ − x₀)</div><p>它会随着 t 接近 1 而缩小，不是当前参数化下的时间导数。只有当模型改成预测 clean action 或剩余残差时，target 才可能使用 <code>actions</code> 或 <code>actions - noised_actions</code>。</p></section>

        <section id="loss"><span className="record-label">06 · MASKED VELOCITY MSE</span><h2>只让有效 action timestep 参与监督</h2><pre className="code-block"><code>{lossCode}</code></pre><div className="equation">L = Σ m(b,h)(v̂ − v*)² / [Σ m(b,h) · A]</div><p><code>mask: (B,H)</code> 在最后补一维后沿 action_dim 广播。不能直接对 <code>error * mask</code> 调用普通 <code>mean()</code>，因为 padding 的零误差仍会进入分母，使 padding 较多的 batch 得到被人为压低的 loss。</p></section>

        <section id="paths"><span className="record-label">07 · PATH CHOICE</span><h2>target velocity 永远由所选路径的导数决定</h2><p><code>actions - noise</code> 只对应当前采用的匀速直线路径。若改成 <code>x_t=(1-t²)x₀+t²x₁</code>，路径仍在同一直线上，但速度会变为 <code>2t(x₁-x₀)</code>；真正的曲线路径也会产生另一套随 t 变化的 target。</p><div className="insight"><span>GENERAL RULE</span><p><code>target_velocity = dx_t / dt</code>。此外，即使每对 noise→action 的条件路径是直线，模型从大量路径学习到的是条件平均速度场，推理时 ODE 的整体轨迹也不一定是一条直线。</p></div></section>

        <section id="summary"><span className="record-label">08 · USER SUMMARY</span><h2>最终形成的心智模型</h2><ol className="step-list"><li>真实数据先提供当前观测 <code>c</code> 和 action chunk <code>x₁</code>。</li><li>训练时为每个样本从 <code>[0,1]</code> 独立采样 timestep，并采样同 shape 的 noise <code>x₀</code>。</li><li>由路径公式生成 <code>noised_action x_t</code>，由路径对 t 的导数生成对应速度 <code>v*</code>。</li><li>一条监督样本就是 <code>观测 / noised_action / timestep → velocity</code>。</li><li>多个样本由 DataLoader 和向量化操作组成 batch，于是所有 Tensor 增加并共享同一个 B 维。</li></ol><div className="test-result"><span>EXERCISE TARGET</span><b>6 TESTS</b><i>完整 optimizer train_step 留到 Day 19</i></div></section>

        <section id="implementation"><span className="record-label">09 · COMPLETED IMPLEMENTATION</span><h2>完成代码与本次错误修正</h2><pre className="code-block"><code>{completedCode}</code></pre><div className="insight"><span>FIX LOG</span><p>第一次实现 masked MSE 时只做了逐元素除法，loss 仍是 <code>(B,H,A)</code>；修正为先对有效误差 <code>.sum()</code>，再除以 <code>mask.sum() × action_dim</code>，最终得到可直接 <code>backward()</code> 的标量。为兼容不同 PyTorch 版本，带 generator 的噪声采样使用 <code>torch.randn(shape, ...)</code>，而不是 <code>randn_like(..., generator=...)</code>。</p></div><p>这里的 mask 只屏蔽 action chunk 越过 episode 末尾后补出的 padding。真实记录中的静止动作仍然有效，必须保留监督，否则策略可能学不会停止或等待。</p></section>

        <section id="review"><span className="record-label">10 · REVIEW ANSWERS</span><h2>5 道复盘题</h2><ol className="step-list"><li><b>为什么 batch 内每个样本独立采样 timestep？</b><br />为了覆盖生成路径的不同阶段，使网络在不同 <code>t</code> 和 <code>x_t</code> 处都学到速度。<code>t</code> 来自 <code>torch.rand</code> 的 [0,1] 均匀分布，不是 <code>randn</code> 正态分布。</li><li><b>为什么推理要从 noise 出发？</b><br />推理时没有真实 actions，需要根据网络学到的速度场求解 ODE，将初始噪声逐步运输到条件动作分布。</li><li><b>为什么 mask 既影响分子也影响分母？</b><br />padding 不是真实监督，既不能累计误差，也不能计入有效元素数；否则 padding 越多，loss 越容易被人为稀释。</li><li><b>为什么训练前向不能放进 no_grad？</b><br />前向无论是否启用梯度都会使用权重；真正的区别是训练必须记录计算图，才能通过 <code>loss.backward()</code> 为权重计算梯度。</li><li><b>一次标准参数更新的顺序？</b><br /><code>optimizer.zero_grad() → loss.backward() → optimizer.step()</code>：清除旧梯度、计算当前梯度、更新参数。</li></ol><div className="test-result"><span>REVIEW</span><b>CORE LOGIC PASSED</b><i>重点修正：rand / randn 与 weight / gradient tracking</i></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT TRAINING</span><h2>Day 19 · 完整 Flow Matching train_step</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 18 · TRAINING OBJECTIVE</span></footer>
  </main>;
}
