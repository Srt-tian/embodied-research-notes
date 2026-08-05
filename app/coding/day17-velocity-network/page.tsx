import Link from "next/link";
import { SiteHeader } from "../../site-header";

const forwardCode = `def forward(
    self,
    noised_actions: Tensor,  # (B,H,A)
    timesteps: Tensor,       # (B,)
    condition: Tensor,       # (B,C)
) -> Tensor:
    action_features = self.action_encoder(noised_actions)  # (B,H,D)

    time_embedding = sinusoidal_timestep_embedding(
        timesteps,
        self.time_embedding_dim,
    )
    time_features = self.time_encoder(time_embedding)      # (B,D)
    time_features = time_features[:, None, :]               # (B,1,D)

    condition_features = self.condition_encoder(condition) # (B,D)
    condition_features = condition_features[:, None, :]     # (B,1,D)

    fused = action_features + time_features + condition_features
    return self.velocity_head(fused)                        # (B,H,A)`;

const eulerCode = `velocity = model(x_t, timesteps, condition)  # (B,H,A)
x_next = x_t + delta_t * velocity             # (B,H,A)`;

export default function Day17VelocityNetwork() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 17</span></div>
      <div className="coding-note-title"><span>DAY 17 · 2026-08-05</span><h1>Velocity<br />Network</h1><p>把 noised action、生成时间和机器人条件编码到同一隐藏维度，在 action chunk 的每个未来位置预测 Flow Matching 速度。</p></div>
      <div className="coding-score"><span>NETWORK MODULE · COMPLETE</span><b>05</b><i>core TODOs</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#contract">01 / 输入输出</a><a href="#branches">02 / 三路编码</a><a href="#broadcast">03 / 沿 Horizon 广播</a><a href="#fusion">04 / 特征融合</a><a href="#output">05 / 输出 Shape</a><a href="#time">06 / 两种时间</a><a href="#review">07 / 课后复盘</a></aside>
      <div className="coding-note-content">
        <section id="contract"><span className="record-label">01 · MODEL CONTRACT</span><h2>真正开始学习条件速度场</h2><div className="shape-callout"><span>FLOW POLICY I/O</span><b>(B,H,A) → (B,H,A)</b><p>输入还包括 <code>timesteps: (B,)</code> 与 <code>condition: (B,C)</code>；输出与当前 noised action 逐元素对应。</p></div><pre className="code-block"><code>{forwardCode}</code></pre></section>

        <section id="branches"><span className="record-label">02 · THREE ENCODERS</span><h2>先把三种语义映射到同一 hidden_dim</h2><div className="table-wrap"><table><thead><tr><th>分支</th><th>原始 shape</th><th>编码后</th><th>含义</th></tr></thead><tbody><tr><td>Action</td><td>(B,H,A)</td><td>(B,H,D)</td><td>每个未来位置当前的含噪动作</td></tr><tr><td>Time</td><td>(B,)</td><td>(B,D)</td><td>整个 chunk 共用的生成进度</td></tr><tr><td>Condition</td><td>(B,C)</td><td>(B,D)</td><td>图像、状态或其他机器人条件</td></tr></tbody></table></div><p>三路都映射为 D 维，才能通过逐元素相加完成简洁融合。这里的 condition 已假设是上游观测编码器输出的定长特征。</p></section>

        <section id="broadcast"><span className="record-label">03 · HORIZON BROADCAST</span><h2><code>(B,D)</code> 要变成 <code>(B,1,D)</code></h2><p>补在中间的 singleton 维明确对应 horizon：batch 维 B 和特征维 D 一一对齐，大小为 1 的维度自动广播到 H。</p><div className="equation">(B,H,D) + (B,1,D) + (B,1,D) → (B,H,D)</div><p>这不是实际复制 H 份数据；广播在运算中以共享视图语义扩展。若误写为 <code>(B,D,1)</code>，D 会与 horizon 错误对齐。</p></section>

        <section id="fusion"><span className="record-label">04 · FEATURE FUSION</span><h2>共享 time/condition，不等于所有位置输出相同</h2><p>每个 horizon 位置拥有不同的 <code>action_features[:,h,:]</code>，但共享同一个 t 和 condition。逐元素相加后，各位置仍保留自己的含噪动作上下文，并受共同生成阶段与观测条件调制。</p><div className="insight"><span>DESIGN BOUNDARY</span><p>这个轻量 MLP 不显式建模不同 horizon 位置之间的交互。后续可用 Transformer 或 1D temporal block 增强跨位置依赖；本日只练清三路条件融合。</p></div></section>

        <section id="output"><span className="record-label">05 · VELOCITY SHAPE</span><h2>速度必须与被更新变量一一对应</h2><pre className="code-block"><code>{eulerCode}</code></pre><p>Flow Matching 的速度是 <code>x_t</code> 对生成时间 t 的导数。既然 <code>x_t</code> 是完整 action chunk <code>(B,H,A)</code>，速度也必须是相同 shape，Euler 更新才能对每个未来位置和动作维逐元素执行。</p></section>

        <section id="time"><span className="record-label">06 · TWO TIME AXES</span><h2>Flow timestep 与 action horizon 不是一回事</h2><div className="table-wrap"><table><thead><tr><th>变量</th><th>shape</th><th>语义</th><th>是否共享</th></tr></thead><tbody><tr><td>t</td><td>(B,)</td><td>从噪声到动作的生成进度</td><td>一个样本的整个 chunk 共用</td></tr><tr><td>h</td><td>H positions</td><td>chunk 内未来动作位置</td><td>每个位置单独输出速度</td></tr></tbody></table></div><p>因此“每一个时刻都需要一个 action chunk 速度”中的“时刻”若指 t 是正确的；但不能把 t 与机器人未来时间位置 h 混为同一维。</p></section>

        <section id="review"><span className="record-label">07 · USER REVIEW</span><h2>课后回答与修正</h2><ol className="step-list"><li><code>(B,D) → (B,1,D)</code>：回答“方便广播”方向正确；更关键的是显式把 singleton 放在 horizon 轴，使语义正确对齐。</li><li>是否需要 <code>expand</code>：回答正确，PyTorch 会从 1 自动广播到 H，无需真实复制。</li><li>为什么输出是 <code>(B,H,A)</code>：回答正确，速度与 action chunk 的每个元素一一对应。</li><li>为什么所有 horizon 位置共享一个 t：整体 chunk 在同一个 Flow Matching 进度下被更新，但 H 个位置分别预测不同速度。</li><li>时间编码复用审计：发现 Day 16/17 原始 helper 的频率范围与归一化 t 不匹配，后续应统一采用适合 [0,1] 的频率尺度。</li></ol><div className="test-result"><span>STATIC CHECK</span><b>SYNTAX &amp; ARCHIVE VERIFIED</b><i>PyTorch tests not run in generator</i></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT TRAINING</span><h2>Day 18 · Flow Matching 完整训练目标</h2><Link href="/coding/day18-flow-matching-objective">进入下一篇 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 17 · VELOCITY NETWORK</span></footer>
  </main>;
}
