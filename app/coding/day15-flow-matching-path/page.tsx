import Link from "next/link";
import { SiteHeader } from "../../site-header";

const targetCode = `def build_flow_matching_targets(
    actions: Tensor,     # (B,H,A)
    timesteps: Tensor,   # (B,)
    noise: Tensor,       # (B,H,A)
) -> tuple[Tensor, Tensor]:
    t = timesteps.reshape(-1, 1, 1)

    noised_actions = (1.0 - t) * noise + t * actions
    target_velocity = actions - noise
    return noised_actions, target_velocity`;

const samplerCode = `times = torch.linspace(0.0, 1.0, steps=num_steps + 1)
x = torch.randn_like(action_template)

for start_t, end_t in zip(times[:-1], times[1:]):
    t = torch.full((B,), start_t, device=x.device)
    velocity = model(x, t, condition)
    delta_t = end_t - start_t
    x = x + delta_t * velocity`;

const importCode = `cd ~/coding/day15_coding-training-v2/day15-flow-matching-path
source /home/user/coding/.venv/bin/activate
python -m pytest -q

# 或显式指定唯一 uv 项目
uv run --project /home/user/coding -- python -m pytest -q`;

export default function Day15FlowMatchingPath() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 15</span></div>
      <div className="coding-note-title"><span>DAY 15 · 2026-08-03</span><h1>Flow Matching<br />Path</h1><p>从真实 action chunk 与随机噪声构造直线路径上的中间状态，明确训练 target velocity 的方向，并把训练目标连接到 Euler 推理。</p></div>
      <div className="coding-score"><span>CORE FORMULA · COMPLETE</span><b>03</b><i>key ideas</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#goal">01 / 任务定义</a><a href="#path">02 / 直线路径</a><a href="#shape">03 / Timestep 广播</a><a href="#velocity">04 / 速度方向</a><a href="#euler">05 / Euler Sampler</a><a href="#environment">06 / 导入问题</a><a href="#review">07 / 课后复盘</a></aside>
      <div className="coding-note-content">
        <section id="goal"><span className="record-label">01 · TRAINING GOAL</span><h2>训练速度场之前，先把监督目标构造正确</h2><p>每个样本包含真实动作 <code>x₁=actions</code>、随机噪声 <code>x₀=noise</code> 和独立采样的生成时间 <code>t</code>。本模块只负责生成模型输入 <code>x_t</code> 与监督速度，不包含 velocity network。</p><div className="shape-callout"><span>I/O CONTRACT</span><b>(B,H,A)</b><p><code>actions</code>、<code>noise</code>、<code>noised_actions</code> 与 <code>target_velocity</code> shape 完全一致；每个 batch 样本只对应一个标量 t。</p></div></section>

        <section id="path"><span className="record-label">02 · LINEAR PATH</span><h2>从噪声匀速走向真实动作</h2><div className="equation">x_t = (1-t)x₀ + tx₁ = x₀ + t(x₁-x₀)</div><pre className="code-block"><code>{targetCode}</code></pre><p><code>t=0</code> 时得到纯噪声，<code>t=1</code> 时得到真实动作；中间值位于两者连线上。训练时通常用 <code>torch.rand(B)</code> 为 batch 中每个样本独立采样 t。</p></section>

        <section id="shape"><span className="record-label">03 · BROADCAST SEMANTICS</span><h2><code>(B,)</code> 必须变成 <code>(B,1,1)</code></h2><p>PyTorch 广播从最后一维向前对齐。直接用 <code>(B,)</code> 与 <code>(B,H,A)</code> 运算，会尝试让 B 对齐 action 维 A，而不是 batch 维；即使数值碰巧相等，语义也是错的。</p><div className="equation">(B,1,1) × (B,H,A) → (B,H,A)</div><p><code>reshape(-1,1,1)</code> 明确表示：一个样本共享一个生成时间，该 t 同时作用于整个 action chunk 的 H 个位置和 A 个动作维度。</p></section>

        <section id="velocity"><span className="record-label">04 · TARGET DIRECTION</span><h2>速度必须是 action − noise</h2><div className="equation">dx_t / dt = x₁ - x₀ = actions - noise</div><p>单条直线路径的导数不含 t，因此目标速度恒定。若误写为 <code>noise - actions</code>，Euler 推理会从噪声沿反方向移动，越来越远离真实动作。</p><div className="error-card"><span>DIRECTION BUG</span><h3>起点和终点的顺序决定速度方向</h3><code>正确：target_velocity = actions - noise</code><p>目标不是“预测一个差值”这么笼统，而是预测从 t=0 的噪声走向 t=1 的数据所需的向量场。</p></div></section>

        <section id="euler"><span className="record-label">05 · INFERENCE CONNECTION</span><h2>Euler sampler 把预测速度积分成动作</h2><pre className="code-block"><code>{samplerCode}</code></pre><p><code>torch.linspace</code> 在 0 到 1 之间生成等间隔时间点。每一步使用 <code>x_next = x + Δt·v</code> 更新当前 action chunk，重复后从初始噪声到达生成动作。</p></section>

        <section id="environment"><span className="record-label">06 · ENGINEERING REVIEW</span><h2>外部虚拟环境不等于模块自动可导入</h2><pre className="code-block"><code>{importCode}</code></pre><p>本次遇到 <code>ModuleNotFoundError: No module named 'flow_path'</code>。外部 <code>.venv</code> 只决定 Python 和依赖，模块搜索路径仍由运行方式决定。统一使用根目录唯一 uv 项目，并在当天目录执行 <code>python -m pytest</code>，可避免测试入口依赖偶然的当前路径。</p><div className="insight"><span>LONG-TERM RULE</span><p>每个 Day 只保留 README、练习代码与 tests；不再各自创建 <code>pyproject.toml</code>、<code>uv.lock</code> 或 <code>.venv</code>。</p></div></section>

        <section id="review"><span className="record-label">07 · USER REVIEW</span><h2>课后回答与修正</h2><ol className="step-list"><li>关于 timestep reshape：最初回答“维度不同，不能广播”。修正为：广播允许维度不同，但 <code>(B,)</code> 会从末维错误对齐，必须显式变为 <code>(B,1,1)</code>。</li><li>Euler sampler：通过离散更新 <code>x_next=x+Δt·v</code> 对速度场做数值积分。</li><li>目标速度为何与 t 无关：回答正确，因为选用的是直线路径，沿 t 匀速移动。</li></ol><div className="test-result"><span>STATUS</span><b>FORMULA &amp; SHAPES REVIEWED</b><i>runtime result not recorded</i></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 16 · Timestep Embedding</h2><Link href="/coding/day16-timestep-embedding">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 15 · FLOW PATH</span></footer>
  </main>;
}
