import Link from "next/link";
import { SiteHeader } from "../../site-header";

const solution = `class DropoutActionMLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states):
        return self.network(states)


def action_mse_loss(model, states, targets):
    predictions = model(states)
    return ((predictions - targets) ** 2).mean()


def gradient_l2_norm(model):
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().pow(2).sum().item()
    return total ** 0.5


def train_step(model, optimizer, states, targets, max_grad_norm):
    model.train()
    optimizer.zero_grad()

    loss = action_mse_loss(model, states, targets)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm
    ).item()
    optimizer.step()
    return loss.item(), grad_norm


def predict(model, states):
    model.eval()
    with torch.no_grad():
        return model(states)`;

export default function Day04GradientDebug() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 04</span></div>
      <div className="coding-note-title"><span>DAY 04 · 2026-07-16</span><h1>PyTorch<br />梯度诊断</h1><p>接手一段语法正确但训练语义错误的代码，定位断梯度、全局梯度范数、训练/推理模式与梯度裁剪顺序。</p></div>
      <div className="coding-score"><span>FINAL SCORE</span><b>9.3</b><i>/ 10</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#task">01 / 题目</a><a href="#idea">02 / 思路</a><a href="#code">03 / 代码</a><a href="#tests">04 / 测试</a><a href="#review">05 / 复盘</a><a href="#knowledge">06 / 知识点</a></aside>
      <div className="coding-note-content">
        <section id="task"><span className="record-label">01 · TASK</span><h2>题目与约束</h2><p>修复一个带 Dropout 的动作预测训练管线。公开函数签名和测试不可修改；训练函数必须返回标量 loss 与裁剪前的全局梯度 L2 norm，推理必须确定且不构建计算图。</p><div className="constraint-grid"><div><b>训练</b><p>保留计算图、清除旧梯度、反向传播、裁剪、更新参数。</p></div><div><b>推理</b><p>关闭 Dropout，并禁用 autograd 图记录。</p></div></div></section>

        <section id="idea"><span className="record-label">02 · APPROACH</span><h2>排查思路</h2><ol className="step-list"><li>先检查 loss 的 requires_grad 和 grad_fn，定位断图位置。</li><li>检查 backward 后每个 parameter.grad 是否存在。</li><li>用已知 3、4、12 梯度验证全局 L2 norm 应为 13。</li><li>核对 zero_grad、backward、clip、step 的生命周期顺序。</li><li>分别检查 train/eval 模式和 autograd 是否启用。</li></ol><div className="shape-callout"><span>TRAIN ORDER</span><b>zero → fwd → bwd → clip → step</b><p>梯度只有在 backward 后才存在；裁剪必须发生在参数更新之前。</p></div></section>

        <section id="code"><span className="record-label">03 · IMPLEMENTATION</span><h2>最终代码</h2><p>以下为删除全部故障标记后的干净版本。</p><pre className="code-block"><code>{solution}</code></pre></section>

        <section id="tests"><span className="record-label">04 · VERIFICATION</span><h2>测试结果</h2><div className="test-result"><span>PYTEST</span><b>12 passed</b><i>100%</i></div><p>覆盖可微 loss、参数梯度、全局 L2 norm、旧梯度清理、参数更新、训练模式、裁剪前后 norm、确定性推理以及禁用 autograd。</p></section>

        <section id="review"><span className="record-label">05 · ERROR REVIEW</span><h2>错误复盘</h2><div className="error-card"><span>DETACH</span><h3>模型输出过早脱离计算图</h3><p>对 predictions 调用 <code>detach()</code> 后，loss 不再知道自己与模型参数的关系，<code>requires_grad=False</code>，无法反向传播。</p></div><div className="error-card"><span>GLOBAL NORM</span><h3>不能直接相加各参数的 norm</h3><p>全局 L2 norm 是所有梯度元素平方求和后再开方；分别计算 norm 后直接相加会得到错误结果。</p></div><div className="error-card"><span>RETURN VALUE</span><h3>裁剪前 norm 应读取 API 返回值</h3><p><code>clip_grad_norm_</code> 原地裁剪梯度，但返回的是裁剪前的总 norm。裁剪后再调用自定义函数得到的是另一项诊断值。</p></div></section>

        <section id="knowledge"><span className="record-label">06 · KNOWLEDGE</span><h2>知识点总结</h2><div className="knowledge-grid"><article><span>AUTOGRAD</span><h3>forward 记录，backward 计算</h3><p>启用梯度时，forward 构建并保存计算图；backward 才沿图计算参数梯度。</p></article><article><span>DETACH / ITEM</span><h3>两种脱离方式</h3><p><code>detach()</code> 返回不追踪梯度的 Tensor；<code>item()</code> 将单元素 Tensor 转为 Python 数值。</p></article><article><span>MODE</span><h3>eval 不等于 no_grad</h3><p>eval 控制 Dropout/BatchNorm 行为；no_grad 控制是否记录 autograd 图，推理通常两者都要。</p></article><article><span>CLIPPING</span><h3>裁剪不是清零</h3><p>梯度爆炸时按总 norm 缩放梯度方向；必须在 backward 后、step 前执行。</p></article><article><span>ZERO GRAD</span><h3>清除的是梯度</h3><p>PyTorch 默认累加 parameter.grad；普通训练每轮 backward 前应清除旧梯度，而不是“清除 loss”。</p></article><article><span>DEBUG</span><h3>先检查梯度链路</h3><p>训练不更新时依次检查 requires_grad、grad_fn、parameter.grad、参数前后差异和 optimizer 顺序。</p></article></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 5 · PyTorch 源码阅读与 Hook</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 04 · VERIFIED</span></footer>
  </main>;
}
