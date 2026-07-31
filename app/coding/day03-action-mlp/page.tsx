import Link from "next/link";
import { SiteHeader } from "../../site-header";

const solution = `class ActionMLP(nn.Module):
    def __init__(self, state_dim, action_dim, horizon, hidden_dim=128):
        super().__init__()
        dims = (state_dim, action_dim, horizon, hidden_dim)
        if not all(type(dim) is int and dim > 0 for dim in dims):
            raise ValueError("All dimensions must be positive integers.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, states):
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected (B, {self.state_dim}), got {states.shape}"
            )
        output = self.network(states)
        return output.reshape(
            states.shape[0], self.horizon, self.action_dim
        )


def action_mse_loss(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    return torch.mean((predictions - targets) ** 2)


def train_step(model, optimizer, states, targets):
    optimizer.zero_grad()
    predictions = model(states)
    loss = action_mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()
    return loss.item()`;

export default function Day03ActionMLP() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 03</span></div>
      <div className="coding-note-title"><span>DAY 03 · 2026-07-15</span><h1>PyTorch<br />Action MLP</h1><p>第一次进入模型模块：将单步机器人状态映射为完整 action chunk，并打通 forward、MSE、backward 与 optimizer update。</p></div>
      <div className="coding-score"><span>FINAL SCORE</span><b>9.4</b><i>/ 10</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#task">01 / 题目</a><a href="#idea">02 / 思路</a><a href="#code">03 / 代码</a><a href="#tests">04 / 测试</a><a href="#review">05 / 复盘</a><a href="#knowledge">06 / 知识点</a></aside>
      <div className="coding-note-content">
        <section id="task"><span className="record-label">01 · TASK</span><h2>题目与约束</h2><p>实现一个 shape-safe 的 <code>ActionMLP</code>：输入 <code>(B, state_dim)</code>，输出 <code>(B, horizon, action_dim)</code>；同时实现禁止意外广播的 MSE loss 和一次完整训练更新。</p><div className="constraint-grid"><div><b>模型输出</b><p>最后一个 Linear 输出 horizon × action_dim，再恢复 action chunk 结构。</p></div><div><b>训练闭环</b><p>zero_grad → forward → loss → backward → optimizer.step。</p></div></div></section>

        <section id="idea"><span className="record-label">02 · APPROACH</span><h2>解题思路</h2><ol className="step-list"><li>验证所有网络维度为正整数，并显式排除 bool。</li><li>用三层 Linear 与 ReLU 构建 Sequential MLP。</li><li>利用 Linear 只变换最后一维的规则处理整个 batch。</li><li>从输入动态取得 B，不写死 batch size。</li><li>loss 前要求预测与目标 shape 完全一致，再进行反向传播和更新。</li></ol><div className="shape-callout"><span>MODEL FLOW</span><b>(B, S) → (B, H, A)</b><p>先输出 (B, H × A)，再 reshape 为动作窗口；一个 state 条件对应一个完整 action chunk。</p></div></section>

        <section id="code"><span className="record-label">03 · IMPLEMENTATION</span><h2>最终代码</h2><pre className="code-block"><code>{solution}</code></pre></section>

        <section id="tests"><span className="record-label">04 · VERIFICATION</span><h2>测试结果</h2><div className="test-result"><span>PYTEST</span><b>18 passed in 1.39s</b><i>100%</i></div><p>覆盖网络类型与结构、非法维度、空 batch、输出 shape 与 dtype、错误输入、精确 MSE、禁止广播、梯度生成以及参数实际更新。</p></section>

        <section id="review"><span className="record-label">05 · ERROR REVIEW</span><h2>错误复盘</h2><div className="error-card"><span>PYTHON TYPE SYSTEM</span><h3>bool 是 int 的子类</h3><code>isinstance(True, int)  # True</code><p>仅检查 <code>isinstance(dim, int)</code> 会让 <code>horizon=True</code> 通过。最终使用严格类型判断 <code>type(dim) is int</code> 排除布尔值。</p></div><div className="error-card"><span>IMPORT PATH</span><h3>pytest 根目录导入配置遗漏</h3><p>练习包最初没有把项目根目录加入 pytest 的 import path，导致 <code>action_mlp.py</code> 无法导入。正确配置是在 pyproject.toml 中加入 <code>pythonpath = [&quot;.&quot;]</code>，或使用 <code>python -m pytest</code>。</p></div><div className="error-card"><span>BROADCASTING</span><h3>能计算不代表训练语义正确</h3><p>预测与目标 shape 不同但可广播时，PyTorch 仍可能计算 loss。手动检查 shape 是为了阻止样本被错误复用，而不是只为了处理空数组。</p></div></section>

        <section id="knowledge"><span className="record-label">06 · KNOWLEDGE</span><h2>知识点总结</h2><div className="knowledge-grid"><article><span>LINEAR</span><h3>只变换最后一维</h3><p><code>nn.Linear(in, out)</code> 不需要 batch size；输入前导维度保持不变。</p></article><article><span>SHORT CIRCUIT</span><h3>or 的短路求值</h3><p>先检查 ndim，可以避免一维 Tensor 继续访问不存在的 <code>shape[1]</code>。</p></article><article><span>AUTOGRAD</span><h3>backward 计算梯度</h3><p><code>loss.backward()</code> 将梯度写入 parameter.grad；optimizer.step 使用梯度更新参数。</p></article><article><span>GRADIENT</span><h3>梯度默认累加</h3><p>普通训练每步先 zero_grad；只有 gradient accumulation 才故意保留多次 backward 的梯度。</p></article><article><span>SCALAR</span><h3>mean 与 item</h3><p>MSE 的 mean 产生零维 Tensor；<code>item()</code> 将它转成脱离计算图的 Python float。</p></article><article><span>FLOW MATCHING</span><h3>连接后续模型</h3><p>Action MLP 是动作模型的最小雏形；后续会加入 noisy action、时间 t 和观测条件以预测速度场。</p></article></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 4 · Debug 与梯度诊断</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 03 · VERIFIED</span></footer>
  </main>;
}
