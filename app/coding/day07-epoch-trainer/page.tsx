import Link from "next/link";
import { SiteHeader } from "../../site-header";

const coreCode = `def masked_error_sum_and_count(predictions, targets, mask):
    _validate_masked_inputs(predictions, targets, mask)

    squared_error = (predictions - targets) ** 2
    expanded_mask = mask.unsqueeze(-1)
    error_sum = (squared_error * expanded_mask).sum()

    action_dim = predictions.shape[-1]
    valid_count = int(mask.sum().item()) * action_dim
    return error_sum, valid_count


def train_epoch(model, optimizer, loader, device="cpu",
                max_grad_norm=None):
    model.train()
    total_error = 0.0
    total_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        predictions = model(batch["state"])
        error_sum, valid_count = masked_error_sum_and_count(
            predictions, batch["actions"], batch["mask"]
        )
        loss = error_sum / valid_count
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )
        optimizer.step()

        total_error += error_sum.item()
        total_count += valid_count

    if total_count == 0:
        raise ValueError("Loader must contain valid action elements.")
    return total_error / total_count`;

const evalCode = `def evaluate_epoch(model, loader, device="cpu"):
    model.eval()
    total_error = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            predictions = model(batch["state"])
            error_sum, valid_count = masked_error_sum_and_count(
                predictions, batch["actions"], batch["mask"]
            )
            total_error += error_sum.item()
            total_count += valid_count

    if total_count == 0:
        raise ValueError("Loader must contain valid action elements.")
    return total_error / total_count`;

export default function Day07EpochTrainer() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 07</span></div>
      <div className="coding-note-title"><span>DAY 07 · 2026-07-20</span><h1>完整 Epoch<br />训练与评估</h1><p>把单 batch 的策略训练扩展为完整 epoch，正确累计全局 masked MSE，并掌握梯度裁剪、评估模式与 Tensor 标量转换。</p></div>
      <div className="coding-score"><span>STATUS · DEBUG REVIEWED</span><b>4</b><i>个错误定位</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#task">01 / 题目</a><a href="#idea">02 / 核心思路</a><a href="#code">03 / 修复代码</a><a href="#review">04 / 错误复盘</a><a href="#syntax">05 / 语法笔记</a><a href="#answers">06 / 问答总结</a></aside>
      <div className="coding-note-content">
        <section id="task"><span className="record-label">01 · TASK</span><h2>训练目标</h2><p>实现 <code>train_epoch</code> 与 <code>evaluate_epoch</code>：模型输出形状为 (B,H,A)，mask 为 (B,H)，padding 元素不参与 loss；训练阶段支持可选的梯度范数裁剪，评估阶段不创建计算图。</p><div className="constraint-grid"><div><b>全局指标</b><p>返回整个 epoch 所有有效 action 标量的 MSE，而不是 batch loss 的简单平均。</p></div><div><b>梯度链路</b><p><code>error_sum</code> 必须是保留计算图的标量 Tensor，直到完成 <code>backward()</code>。</p></div></div></section>

        <section id="idea"><span className="record-label">02 · GLOBAL REDUCTION</span><h2>核心思路</h2><ol className="step-list"><li>先计算形状为 (B,H,A) 的逐元素平方误差。</li><li>把 mask 从 (B,H) 扩成 (B,H,1)，利用广播清零 padding 位置。</li><li>对 masked error 调用 <code>.sum()</code>，得到可反向传播的标量 <code>error_sum</code>。</li><li>分母是有效 timestep 数乘 action_dim，即有效 action 标量总数。</li><li>每个 batch 用 <code>error_sum / valid_count</code> 反向传播；epoch 指标则累计总误差和总元素数后再相除。</li></ol><div className="shape-callout"><span>GLOBAL MSE</span><b>Σ error / Σ count</b><p>不同 batch 的有效长度不同时，不能让每个 batch 拥有相同权重。</p></div></section>

        <section id="code"><span className="record-label">03 · FIXED IMPLEMENTATION</span><h2>修复代码</h2><h3>Masked error 与训练</h3><pre className="code-block"><code>{coreCode}</code></pre><h3>完整评估循环</h3><pre className="code-block"><code>{evalCode}</code></pre></section>

        <section id="review"><span className="record-label">04 · ERROR REVIEW</span><h2>四个失败的根因</h2><div className="error-card"><span>NON-SCALAR</span><h3>masked error 没有求和</h3><code>masked_error_sum = squared_error * mask</code><p>这里仍是 (B,H,A)，导致 <code>backward()</code> 报“只能对标量输出隐式创建梯度”。正确写法末尾需要 <code>.sum()</code>。</p></div><div className="error-card"><span>COUNT</span><h3>分母漏掉 action_dim</h3><p><code>mask.sum()</code> 只数有效 timestep；每个 timestep 有 A 个动作标量，因此必须再乘 <code>predictions.shape[-1]</code>。</p></div><div className="error-card"><span>PYTHON TYPE</span><h3>int 与 Tensor 混用</h3><p><code>valid_count</code> 返回 Python <code>int</code> 后不能再调用 <code>.item()</code>。epoch 累加器保持 Python 数值，可避免布尔判断歧义并确保最终返回 float。</p></div><div className="error-card"><span>EVALUATION</span><h3>评估累计了 Tensor</h3><p>评估循环应累加 <code>error_sum.item()</code>，同时配合 <code>model.eval()</code> 和覆盖整个循环的 <code>torch.no_grad()</code>。</p></div></section>

        <section id="syntax"><span className="record-label">05 · PYTORCH SYNTAX</span><h2>本次语法笔记</h2><div className="knowledge-grid"><article><span>CLAMP</span><h3>数值限幅</h3><p><code>torch.clamp(x, min=0, max=1)</code>；<code>torch.clip</code> 是别名，<code>clamp_</code> 为原地版本。</p></article><article><span>GRAD CLIP</span><h3>梯度范数裁剪</h3><p><code>clip_grad_norm_</code> 原地修改参数的 <code>.grad</code>，返回裁剪前的总梯度范数。</p></article><article><span>ORDER</span><h3>裁剪顺序</h3><p><code>zero_grad → backward → clip_grad_norm_ → step</code>。裁剪必须发生在梯度产生之后、参数更新之前。</p></article><article><span>ITEM</span><h3>Tensor 转 Python 标量</h3><p><code>.item()</code> 返回与 dtype 对应的 Python <code>float</code>、<code>int</code> 或 <code>bool</code>，并切断计算图。</p></article><article><span>DETACH</span><h3>detach().item()</h3><p>和直接 <code>.item()</code> 得到的数值相同；若最终就是 Python 标量，显式 <code>detach()</code> 通常重复。</p></article><article><span>SUM</span><h3>.sum() 与 sum()</h3><p><code>x.sum()</code> 默认对所有元素求和；Python <code>sum(x)</code> 遍历第 0 维，通常相当于 <code>x.sum(dim=0)</code>。</p></article></div></section>

        <section id="answers"><span className="record-label">06 · CONCEPT CHECK</span><h2>问答总结</h2><ol className="step-list"><li>不能平均 batch loss：各 batch 的有效 action 元素数不同，应按 <code>valid_count</code> 加权。</li><li>分别返回误差和计数，便于计算整个 epoch 的全局 loss。</li><li><code>.item()</code> 会离开计算图，只用于日志与统计，不能在 <code>backward()</code> 前把 loss 转成 Python 数值。</li><li><code>model.eval()</code> 改变 Dropout、BatchNorm 等模块行为；<code>torch.no_grad()</code> 才负责关闭梯度记录，二者不能替代。</li><li>只有 <code>backward()</code> 后才产生梯度；只有 <code>optimizer.step()</code> 前裁剪才会影响本次参数更新。</li></ol></section>
      </div>
    </article>

    <section className="next-note shell"><span>NEXT</span><h2>Day 8 · 多轮训练与 Early Stopping</h2><Link href="/coding/day08-multi-epoch-trainer">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 07 · DEBUG REVIEWED</span></footer>
  </main>;
}
