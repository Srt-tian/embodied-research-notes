import Link from "next/link";
import { SiteHeader } from "../../site-header";

const odeCode = `dx_t / dt = v_theta(x_t, t, condition)

# 训练：用 velocity MSE 优化 v_theta 的参数
# 推理：固定 v_theta，用数值积分计算 x_0 -> x_1`;

const completedCode = `@torch.no_grad()
def sample_actions_euler(
    model,
    condition,
    *,
    horizon,
    action_dim,
    num_steps,
    generator=None,
):
    batch_size = condition.shape[0]
    action_shape = (batch_size, horizon, action_dim)

    x = torch.randn(
        action_shape,
        generator=generator,
        device=condition.device,
        dtype=condition.dtype,
    )
    dt = 1.0 / num_steps

    for step in range(num_steps):
        timesteps = torch.full(
            (batch_size,),
            step * dt,
            device=condition.device,
            dtype=condition.dtype,
        )
        velocity = model(x, timesteps, condition)

        if velocity.shape != action_shape:
            raise ValueError(
                "model must return velocity with shape (B, H, A)"
            )

        x = x + dt * velocity

    return x`;

const timelineCode = `num_steps = 4, dt = 0.25

模型输入 t=0.00  -> 更新得到 x_0.25
模型输入 t=0.25  -> 更新得到 x_0.50
模型输入 t=0.50  -> 更新得到 x_0.75
模型输入 t=0.75  -> 更新得到 x_1.00`;

export default function Day19FlowMatchingEuler() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 19</span></div>
      <div className="coding-note-title"><span>DAY 19 · 2026-08-10</span><h1>Euler<br />Sampling</h1><p>把 Day 18 训练出的条件速度网络用于推理：从高斯噪声出发，用 Euler 法沿速度场迭代生成完整 action chunk。</p></div>
      <div className="coding-score"><span>SAMPLER · COMPLETE</span><b>05</b><i>core TODOs</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#boundary">01 / 训练与推理边界</a><a href="#ode">02 / ODE 是什么</a><a href="#euler">03 / Euler 离散化</a><a href="#flow">04 / 推理数据流</a><a href="#implementation">05 / 完成实现</a><a href="#endpoint">06 / 时间端点</a><a href="#batch">07 / Batch 时间</a><a href="#modes">08 / no_grad 与 eval</a><a href="#fixes">09 / 错误修正</a><a href="#review">10 / 答题复盘</a></aside>
      <div className="coding-note-content">
        <section id="boundary"><span className="record-label">01 · TRAINING VS INFERENCE</span><h2>训练优化网络，推理积分状态</h2><div className="table-wrap"><table><thead><tr><th>阶段</th><th>已知</th><th>执行内容</th><th>是否更新参数</th></tr></thead><tbody><tr><td>训练</td><td>action、noise、t、condition</td><td>预测速度并计算 velocity MSE</td><td>是，backward + optimizer</td></tr><tr><td>推理</td><td>初始 noise、condition、速度网络</td><td>重复预测速度并更新 x</td><td>否</td></tr></tbody></table></div><p>训练目标仍是让 <code>v_theta</code> 拟合目标速度；推理阶段没有 loss，也没有参数优化，只计算在固定速度场下从 <code>x_0</code> 到 <code>x_1</code> 的轨迹。</p></section>

        <section id="ode"><span className="record-label">02 · ODE</span><h2>ODE 不是另一个模型，也不是优化目标</h2><pre className="code-block"><code>{odeCode}</code></pre><p>提供速度的是网络 <code>v_theta</code>。整个等式是一个带初值的常微分方程，规定动作状态 <code>x_t</code> 应如何随生成时间 <code>t</code> 变化。这里说“求解 ODE”，是根据初始值和速度函数计算轨迹或终点，不是用梯度下降优化 ODE。</p><div className="insight"><span>STRICT WORDING</span><p>训练：优化速度网络。推理：固定网络，并用 Euler 等数值方法近似求解该网络所定义的 ODE。</p></div></section>

        <section id="euler"><span className="record-label">03 · EULER METHOD</span><h2>用当前速度近似下一小段位移</h2><div className="equation">x_(k+1) = x_k + dt · v_theta(x_k, t_k, c)</div><p>积分区间是 <code>[0,1]</code>，均匀切成 <code>num_steps</code> 段，因此 <code>dt = 1 / num_steps</code>。每一步用区间左端点的速度近似整段速度。</p><pre className="code-block"><code>{timelineCode}</code></pre><p>当速度恒为 2 时，无论切成多少个等长步骤，总位移都是 <code>num_steps × (1/num_steps) × 2 = 2</code>；测试正是用这个可解析场景检查更新逻辑。</p></section>

        <section id="flow"><span className="record-label">04 · SAMPLING DATA FLOW</span><h2>一个 batch 如何从 noise 变成 action chunk</h2><div className="shape-callout"><span>MODEL INTERFACE</span><b>(x_t, t, condition) → velocity</b><p><code>x_t: (B,H,A)</code>，<code>t: (B,)</code>，<code>condition: (B,C)</code>，输出速度仍为 <code>(B,H,A)</code>。</p></div><ol className="step-list"><li>按 <code>(B,H,A)</code> 采样高斯噪声作为 <code>x_0</code>。</li><li>计算固定步长 <code>dt</code>。</li><li>在每个 <code>t_k</code> 调用速度模型。</li><li>执行一次 <code>x = x + dt × velocity</code>。</li><li>重复 <code>num_steps</code> 次，返回 <code>x_1</code> 作为动作块。</li></ol><p>推理从与训练路径一致的高斯先验出发；随机起点还允许同一个条件产生不同的合理动作。使用全零会造成起点分布不一致，并丢失这种多样性。</p></section>

        <section id="implementation"><span className="record-label">05 · COMPLETED IMPLEMENTATION</span><h2>修正后的 Euler sampler</h2><pre className="code-block"><code>{completedCode}</code></pre><p><code>generator</code> 让初始噪声可复现；noise、timestep 与 condition 使用相同的 device 和浮点 dtype。模型输出 shape 被显式检查，避免广播悄悄改变 Euler 更新语义。</p><div className="test-result"><span>EXERCISE SUITE</span><b>10 TEST CASES</b><i>shape · seed · constant velocity · time · dtype · no grad · validation</i></div></section>

        <section id="endpoint"><span className="record-label">06 · LEFT ENDPOINTS</span><h2>最后一次模型输入为何是 1 − dt</h2><p>第 <code>k</code> 次更新使用左端点 <code>t_k = k × dt</code>。最后一段区间是 <code>[1-dt, 1]</code>，所以模型最后在 <code>t=1-dt</code> 预测速度，随后一步更新到 <code>t=1</code>。</p><div className="equation">k = 0, ..., N−1　→　t_k = 0, dt, ..., 1−dt</div><p>若在 <code>t=1</code> 再调用模型并执行同样更新，就会额外走到 <code>1+dt</code>，超出规定积分区间。</p></section>

        <section id="batch"><span className="record-label">07 · BATCH TIMESTEPS</span><h2>数值相同，也仍然需要 (B,)</h2><p>同一轮 Euler 更新中，各样本共享当前积分时间，例如 <code>[0.25, 0.25, 0.25]</code>。构造成 <code>(B,)</code> 不是因为各样本噪声不同，而是为了让每个 batch 样本都有与之对齐的时间输入，并沿用训练时的模型接口。</p><div className="table-wrap"><table><thead><tr><th>输入</th><th>Shape</th><th>第 b 项</th></tr></thead><tbody><tr><td>x</td><td>(B,H,A)</td><td>第 b 个候选动作块</td></tr><tr><td>timesteps</td><td>(B,)</td><td>第 b 个样本当前的生成进度</td></tr><tr><td>condition</td><td>(B,C)</td><td>第 b 个样本的观测条件</td></tr></tbody></table></div></section>

        <section id="modes"><span className="record-label">08 · INFERENCE MODES</span><h2>no_grad 与 eval 负责不同事情</h2><div className="table-wrap"><table><thead><tr><th>接口</th><th>作用</th><th>不会做什么</th></tr></thead><tbody><tr><td>@torch.no_grad()</td><td>关闭 autograd 计算图记录</td><td>不会自动关闭 Dropout 或切换 BatchNorm</td></tr><tr><td>model.eval()</td><td>切换 Dropout、BatchNorm 等层的模式</td><td>不会关闭梯度记录</td></tr></tbody></table></div><p>二者不等价。完整推理通常先调用 <code>model.eval()</code>，再在 <code>torch.no_grad()</code> 或 <code>torch.inference_mode()</code> 环境中执行 sampler。</p></section>

        <section id="fixes"><span className="record-label">09 · FIX LOG</span><h2>这次代码与概念的关键修正</h2><ol className="step-list"><li><b>timestep 的 device / dtype：</b>最初的 <code>torch.full((B,), step*dt)</code> 默认创建 CPU float32 Tensor；已补上 <code>device=condition.device</code> 与 <code>dtype=condition.dtype</code>，否则 GPU 或 float64 条件下会不一致。</li><li><b>推理速度的表述：</b><code>num_steps</code> 增大时，模型调用次数和耗时增加，因此推理速度变慢；通常 Euler 离散误差随步长减小而下降。</li><li><b>Batch 时间的原因：</b><code>(B,)</code> 来自批处理接口与样本对齐，不是因为每个样本的高斯噪声彼此不同。</li></ol></section>

        <section id="review"><span className="record-label">10 · REVIEW ANSWERS</span><h2>6 道复盘题</h2><ol className="step-list"><li><b>为什么从随机噪声而非全零开始？</b><br />训练路径的起点分布是高斯先验，推理也必须从相同先验出发；随机起点还能支持条件动作的多样性。</li><li><b>为什么 dt = 1 / num_steps？</b><br />长度为 1 的区间均分成 N 段，每段长度就是 1/N，N 次更新后恰好覆盖总生成时间 1。</li><li><b>为何最后输入 1 − dt 而非 1？</b><br />Euler 使用每段左端点速度；最后在 1−dt 预测，再更新到 1。</li><li><b>为何 timestep 仍是 (B,)？</b><br />它需要与 <code>x[b]</code>、<code>condition[b]</code> 按 batch 对齐，并保持训练与推理统一的模型签名。</li><li><b>num_steps 增大有什么影响？</b><br />模型调用与耗时增加、推理速度下降；步长变小，通常 Euler 积分误差降低。</li><li><b>no_grad 是否等价于 eval？</b><br />不等价：前者关闭梯度图，后者切换部分网络层的行为，真实推理通常同时需要。</li></ol><div className="test-result"><span>REVIEW</span><b>CORE LOGIC COMPLETE</b><i>重点巩固：ODE / optimizer、left endpoint、batch interface</i></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT TRAINING</span><h2>Day 20 · 待开启</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 19 · EULER SAMPLING</span></footer>
  </main>;
}
