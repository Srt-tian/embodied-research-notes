import Link from "next/link";
import { SiteHeader } from "../../site-header";

const embeddingCode = `def sinusoidal_timestep_embedding(
    timesteps: Tensor,       # (B,), normalized to [0,1]
    embedding_dim: int,
    *,
    max_frequency: float = 100.0,
) -> Tensor:
    half_dim = embedding_dim // 2
    frequencies = torch.logspace(
        0.0,
        math.log10(max_frequency),
        steps=half_dim,
        device=timesteps.device,
        dtype=timesteps.dtype,
    )
    angles = 2 * math.pi * timesteps[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding`;

const shapeCode = `timesteps[:, None]    # (B,1)
frequencies[None, :]  # (1,E/2)
angles                # (B,E/2)

sin(angles)           # (B,E/2)
cos(angles)           # (B,E/2)
cat(..., dim=-1)      # (B,E)`;

const oldCode = `# 原始练习版本：适合较大的离散 timestep，
# 直接用于归一化 t∈[0,1] 时变化过小
exponents = torch.linspace(0.0, 1.0, half_dim)
frequencies = torch.exp(-math.log(10_000.0) * exponents)
# 1 → ... → 0.0001`;

export default function Day16TimestepEmbedding() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 16</span></div>
      <div className="coding-note-title"><span>DAY 16 · 2026-08-04</span><h1>Timestep<br />Embedding</h1><p>把 Flow Matching 的标量生成进度编码为多频正弦特征，让 velocity network 同时识别粗粒度阶段与细粒度时间差异。</p></div>
      <div className="coding-score"><span>TIME CONDITION · COMPLETE</span><b>04</b><i>core tests</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#why">01 / 为什么需要 t</a><a href="#principle">02 / 多频编码</a><a href="#shape">03 / Shape 数据流</a><a href="#sincos">04 / Sin 与 Cos</a><a href="#scale">05 / 频率尺度修正</a><a href="#velocity">06 / 与速度场的关系</a><a href="#review">07 / 课后复盘</a></aside>
      <div className="coding-note-content">
        <section id="why"><span className="record-label">01 · WHY TIME CONDITION</span><h2>t 是生成进度，不是机器人帧号</h2><p><code>t=0</code> 表示 action chunk 仍是噪声，<code>t=1</code> 表示已到真实动作。训练时通常为 batch 中每个样本随机采样 t；推理时则用 <code>linspace</code> 产生从 0 到 1 的有序积分网格。</p><div className="equation">vθ(x_t, t, condition) → predicted velocity</div><p>同一个数值的 <code>x_t</code> 在早期和后期含义可能不同，因此 velocity network 需要知道当前生成阶段。</p></section>

        <section id="principle"><span className="record-label">02 · MULTI-FREQUENCY BASIS</span><h2>把一个标量翻译成一组不同速度的“时钟”</h2><p>低频分量缓慢变化，适合表达前期、中期、后期；高频分量变化更快，可以区分更细小的时间差异。网络不必只从一个普通标量中自行学习这些尺度。</p><div className="equation">emb(t) = [sin(ω₀t), …, sin(ωₙt), cos(ω₀t), …, cos(ωₙt)]</div><pre className="code-block"><code>{embeddingCode}</code></pre></section>

        <section id="shape"><span className="record-label">03 · BROADCAST</span><h2><code>(B,)</code> 与所有频率组合</h2><pre className="code-block"><code>{shapeCode}</code></pre><p><code>[:,None]</code> 和 <code>[None,:]</code> 分别补出频率轴与 batch 轴，通过自动广播一次得到每个样本在每个频率下的角度。</p></section>

        <section id="sincos"><span className="record-label">04 · PHASE FEATURES</span><h2>为什么同时使用 sin 和 cos</h2><p>一个频率对应单位圆上的二维坐标 <code>[sin(ωt), cos(ωt)]</code>。两者提供互补相位信息，使时间变化对应平滑的特征变化；尤其在 t=0 时，sin 全为 0，但 cos 仍为 1。</p><p>当 embedding_dim 为奇数时，sin/cos 拼接只能生成 <code>2×floor(E/2)</code> 维，因此在最后补一个 0，维持稳定的 <code>(B,E)</code> 接口。</p></section>

        <section id="scale"><span className="record-label">05 · SCALE AUDIT</span><h2>频率公式必须匹配 timestep 的数值范围</h2><pre className="code-block"><code>{oldCode}</code></pre><p>用户在 Day 17 复用时间编码时发现：这与讲解中的 <code>1,10,100</code> 不一致。问题不只是排列顺序，而是尺度。若 t 只在 [0,1]，<code>t×0.0001</code> 几乎不变，大量维度接近常数。</p><div className="error-card"><span>DESIGN CORRECTION</span><h3>归一化时间应保留真正的高频变化</h3><code>frequencies = logspace(0, 2, ...)  # 1 → ... → 100</code><p>再乘 <code>2πt</code> 后，从低频到高频覆盖多个时间尺度。原来的递减公式更常见于数值较大的离散 timestep，不能脱离时间范围直接照搬。</p></div></section>

        <section id="velocity"><span className="record-label">06 · CONDITIONAL FIELD</span><h2>单条路径速度恒定，整体速度场仍依赖 t</h2><p>对于已知噪声和真实动作的一条直线路径，target velocity 的确是恒定的 <code>x₁-x₀</code>。但推理时网络不知道真实端点，只看到 <code>x_t</code>、t 与机器人条件；它从大量相交路径中学习条件平均速度，因此整体场通常随时间变化。</p><div className="equation">v*(x,t,c) = E[x₁-x₀ | x_t=x, t, c]</div></section>

        <section id="review"><span className="record-label">07 · USER REVIEW</span><h2>课后回答与修正</h2><ol className="step-list"><li><code>timesteps[:,None] × frequencies[None,:]</code> 的目的：回答“广播”正确，更完整地说是构造每个样本与全部频率的两两组合。</li><li>多频率的作用：回答基本正确；它同时表达粗粒度阶段与细粒度差异，但有限维周期编码不保证任意范围绝对唯一。</li><li>Velocity Network 的 I/O：正确指出输入包含 noised action 与 timestep，输出速度；还需加入图像/状态等 condition。</li><li>需要 t 的原因：不能仅说“当前速度不受时间影响”。单条监督路径不显含 t，但模型学习的是依赖 t 的整体条件速度场。</li></ol><div className="test-result"><span>STATUS</span><b>CONCEPT &amp; SCALE REVIEWED</b><i>frequency mismatch corrected</i></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 17 · Velocity Network</h2><Link href="/coding/day17-velocity-network">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 16 · TIME EMBEDDING</span></footer>
  </main>;
}
