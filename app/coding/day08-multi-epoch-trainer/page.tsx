import Link from "next/link";
import { SiteHeader } from "../../site-header";

const fitCode = `def fit(
    model, train_batches, val_batches, optimizer, *,
    device, num_epochs, patience, min_delta=0.0,
    max_grad_norm=None,
):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_batches, optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        val_loss = evaluate(model, val_batches, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        improved = val_loss < best_val_loss - min_delta

        if improved:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("training finished without a best model state")

    model.load_state_dict(best_state)
    return history`;

export default function Day08MultiEpochTrainer() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 08</span></div>
      <div className="coding-note-title"><span>DAY 08 · 2026-07-21</span><h1>多轮训练与<br />Early Stopping</h1><p>承接 Day 7 的单 epoch 函数，完成真实的多轮策略训练：记录训练与验证 loss，判断有效改善，保存最佳参数快照，并在早停后恢复泛化最好的模型。</p></div>
      <div className="coding-score"><span>STATUS · TESTED</span><b>11</b><i>tests passed</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#background">01 / 背景</a><a href="#flow">02 / 控制流</a><a href="#code">03 / 最终实现</a><a href="#review">04 / 代码复盘</a><a href="#syntax">05 / 语法笔记</a><a href="#answers">06 / 问答总结</a></aside>
      <div className="coding-note-content">
        <section id="background"><span className="record-label">01 · BACKGROUND</span><h2>为什么需要多轮训练</h2><p>Day 7 已经能够完成一轮 action policy 的训练与评估。Day 8 将这两个单轮函数组合成完整训练循环。关键认知是：训练结束时的模型不一定是训练过程中泛化能力最好的模型；训练 loss 可能继续下降，而验证 loss 已经因过拟合开始上升。</p><div className="constraint-grid"><div><b>可重复迭代</b><p><code>train_batches</code> 与 <code>val_batches</code> 应像 DataLoader 一样能够被每个 epoch 重新遍历，不能为了方便整体转成 list。</p></div><div><b>验证集选模</b><p>Early Stopping 观察验证集 loss，因为它更能衡量模型对未参与参数更新数据的泛化能力。</p></div></div></section>

        <section id="flow"><span className="record-label">02 · TRAINING CONTROL</span><h2>每个 epoch 的控制流</h2><ol className="step-list"><li>调用 <code>train_one_epoch</code> 更新模型，得到全局训练 loss。</li><li>调用 <code>evaluate</code> 在无梯度模式下计算验证 loss。</li><li>把两个 Python float 追加到 <code>history</code>。</li><li>用严格不等式 <code>val_loss &lt; best_val_loss - min_delta</code> 判断有效改善。</li><li>改善时保存独立参数快照并清零计数；否则未改善计数加一。</li><li>计数达到 <code>patience</code> 时停止，最后恢复最佳模型参数。</li></ol><div className="shape-callout"><span>EARLY STOPPING</span><b>连续未改善次数</b><p><code>patience=3</code> 不是最多训练 3 个 epoch，而是连续 3 轮未达到改善标准后才停止。</p></div></section>

        <section id="code"><span className="record-label">03 · FINAL IMPLEMENTATION</span><h2>最终实现</h2><pre className="code-block"><code>{fitCode}</code></pre><div className="test-result"><span>PYTEST</span><b>11 passed in 1.27s</b><i>完整训练流程与边界测试通过</i></div></section>

        <section id="review"><span className="record-label">04 · CODE REVIEW</span><h2>本次代码复盘</h2><div className="error-card"><span>EPOCH STATE</span><h3><code>improved</code> 应每轮重新计算</h3><p>最初版本先写 <code>improved = None</code>，满足条件时才改为 True，并在分支内重置 False。虽然测试通过，但该变量描述的是“当前 epoch 是否改善”，直接在循环内计算布尔表达式更清晰，也不会意外携带上一轮状态。</p></div><div className="error-card"><span>SNAPSHOT</span><h3>不能只保存普通 state_dict 引用</h3><code>best_state = model.state_dict()</code><p>返回字典里的 Tensor 可能仍与模型参数共享底层数据。后续 optimizer 更新参数时，以为保存好的“最佳值”也可能变化，因此必须使用 <code>deepcopy(model.state_dict())</code> 创建独立快照。</p></div><div className="error-card"><span>RESTORE</span><h3>最后一轮不等于最好一轮</h3><p>Early Stopping 只决定何时停止，停止时模型仍处于最后执行的 epoch。必须调用 <code>model.load_state_dict(best_state)</code>，把原模型原地恢复到验证 loss 最优的状态。</p></div></section>

        <section id="syntax"><span className="record-label">05 · PYTHON & PYTORCH</span><h2>本次语法笔记</h2><div className="knowledge-grid"><article><span>TYPE ALIAS</span><h3>History 不是新类</h3><p><code>History = dict[str, list[float]]</code> 是类型别名，表示 key 为字符串、value 为 float 列表的字典；不会在运行时创建新类。</p></article><article><span>OBJECTS</span><h3>数据也是类的实例</h3><p><code>3</code> 是 <code>int</code> 的对象，NumPy 数组是 <code>np.ndarray</code> 的对象，PyTorch Tensor 是 <code>torch.Tensor</code> 的对象。</p></article><article><span>STATE DICT</span><h3>模型状态字典</h3><p><code>model.state_dict()</code> 通常返回有序映射；key 是参数或 buffer 名称，value 通常为 Tensor。它也包含 BatchNorm 等模块的 buffer。</p></article><article><span>ITEM</span><h3>转换为 Python 标量</h3><p>单元素 Tensor 的 <code>.item()</code> 会根据 dtype 转成 Python <code>float</code>、<code>int</code> 或 <code>bool</code>，转换后不再连接计算图。</p></article><article><span>VALIDATION</span><h3>不是 optimizer.no_grad()</h3><p>正确写法是 <code>torch.no_grad()</code>。它让验证前向传播不创建计算图；<code>model.eval()</code> 则负责切换 Dropout 与 BatchNorm 的行为。</p></article><article><span>KEYWORD ONLY</span><h3>星号后的形参</h3><p>函数签名中的 <code>*</code> 表示之后的参数必须按名称传递，如 <code>device=&quot;cuda&quot;</code>，使调用含义更清楚。</p></article></div></section>

        <section id="answers"><span className="record-label">06 · CONCEPT CHECK</span><h2>问答总结</h2><ol className="step-list"><li>保存最佳模型需要 <code>deepcopy</code>，否则快照里的 Tensor 可能随模型继续训练而变化。</li><li><code>patience=3</code> 表示连续三轮没有有效改善后早停，不是总共只训练三轮；一旦改善，计数归零。</li><li>当 <code>best=0.5</code>、<code>min_delta=0.1</code>、<code>val=0.45</code> 时不算改善，因为 <code>0.45 &lt; 0.4</code> 为 False。</li><li>验证集 loss 比训练集 loss 更能反映未见数据上的泛化能力，因此适合用来选择最佳模型。</li><li>循环结束后加载 <code>best_state</code>，是把最优参数原地恢复到 <code>model</code>；函数返回的仍是训练历史 <code>history</code>。</li></ol></section>
      </div>
    </article>

    <section className="next-note shell"><span>LATEST</span><h2>Day 10 · Multi-Head Self-Attention</h2><Link href="/coding/day10-multihead-attention">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 08 · 11 TESTS PASSED</span></footer>
  </main>;
}
