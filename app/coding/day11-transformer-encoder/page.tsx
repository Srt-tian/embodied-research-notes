import Link from "next/link";
import { SiteHeader } from "../../site-header";

const blockCode = `self.ffn = nn.Sequential(
    nn.Linear(embed_dim, ffn_hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(ffn_hidden_dim, embed_dim),
)

x = x + dropout1(attention(norm1(x), attention_mask))
x = x + dropout2(ffn(norm2(x)))`;

const headsCode = `# (B,T,E) -> (B,T,H,D) -> (B,H,T,D)
x = x.reshape(B, T, num_heads, head_dim)
x = x.transpose(1, 2)

# (B,H,T,D) -> (B,T,H,D) -> (B,T,E)
x = x.transpose(1, 2).contiguous()
x = x.reshape(B, T, embed_dim)`;

export default function Day11TransformerEncoder() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 11</span></div>
      <div className="coding-note-title"><span>DAY 11 · 2026-07-23</span><h1>Pre-Norm<br />Transformer</h1><p>在 Day 10 Self-Attention 的基础上，加入 LayerNorm、FFN、Dropout 与两条残差支路，完成 Transformer Encoder Block。</p></div>
      <div className="coding-score"><span>REFERENCE · VERIFIED</span><b>12</b><i>tests passed</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#block">01 / Block</a><a href="#ffn">02 / FFN</a><a href="#prenorm">03 / Pre-Norm</a><a href="#heads">04 / Head 变形</a><a href="#review">05 / 错误复盘</a><a href="#gradient">06 / 梯度下降</a><a href="#status">07 / 掌握边界</a></aside>
      <div className="coding-note-content">
        <section id="block"><span className="record-label">01 · BLOCK FLOW</span><h2>两个 Pre-Norm 残差子层</h2><pre className="code-block"><code>{blockCode}</code></pre><div className="shape-callout"><span>RESIDUAL</span><b>(B,T,E) + (B,T,E)</b><p>Attention 和 FFN 最终都必须回到 embed_dim，才能与原输入逐元素相加。</p></div></section>

        <section id="ffn"><span className="record-label">02 · FFN</span><h2>逐 token 变换，不混合序列位置</h2><p><code>nn.Linear</code> 只处理最后一维，所以 <code>(B,T,E) → (B,T,hidden) → (B,T,E)</code>，不需要手动展平。Attention 负责 token 间交互，FFN 用共享参数独立变换每个 token。</p><div className="constraint-grid"><div><b>E → hidden → E</b><p>扩维提供更大的特征空间，回到 E 则满足残差相加。</p></div><div><b>GELU + Dropout</b><p>GELU 引入非线性；Dropout 是训练期正则化，eval 时关闭。</p></div></div></section>

        <section id="prenorm"><span className="record-label">03 · PRE-NORM</span><h2>当前先掌握梯度直通的直觉</h2><div className="equation">xₗ₊₁ = xₗ + F(LN(xₗ))</div><p>LayerNorm 位于 Attention / FFN 分支内，残差主干中的 x 不经过归一化或子层。当分支输出接近 0，该层近似恒等映射；反向传播也保留一条局部导数为 1 的直接路径，因此深层网络通常更容易优化。</p><div className="insight"><span>CURRENT UNDERSTANDING</span><p>已理解 Pre-Norm 的核心直觉：梯度可以沿残差主干跨层传播，不必强制经过每层的 LayerNorm、Attention 或 FFN。</p></div><p><strong>掌握边界：</strong><code>I + J</code>、LayerNorm Jacobian 和多层矩阵连乘只作接触，本日尚未细推；以后结合反向传播专题再专门补强。</p></section>

        <section id="heads"><span className="record-label">04 · SPLIT & MERGE HEADS</span><h2>reshape 拆维，transpose 换语义顺序</h2><pre className="code-block"><code>{headsCode}</code></pre><p>不能直接把原存储 reshape 成 <code>(B,H,T,D)</code>。合并时的 <code>contiguous()</code> 可以在使用 reshape 时省略，因为 reshape 必要时会自动复制；显式写出更清楚。transpose 不能省，它保证 token 与 head 的数据顺序正确。</p></section>

        <section id="review"><span className="record-label">05 · ERROR REVIEW</span><h2>本次实现的关键问题</h2><div className="error-card"><span>HEAD ORDER</span><h3>直接 reshape 为 (B,H,T,D)</h3><p>shape 合法但语义错误。正确路径是先 <code>(B,T,H,D)</code>，再交换 T、H。</p></div><div className="error-card"><span>ATTENTION</span><h3>一度直接返回 softmax 权重</h3><p>权重还是 <code>(B,H,T,T)</code>，还需执行 <code>weights @ V</code>、merge heads 与 out_proj。</p></div><div className="error-card"><span>MASK & SCALE</span><h3>条件判断、扩维和缩放</h3><p>用 <code>attention_mask is not None</code>；扩维后再广播；scores 除以 <code>sqrt(head_dim)</code>。</p></div><div className="error-card"><span>FFN</span><h3>多写 GELU，Dropout 位置错误</h3><p>正确顺序是 <code>Linear → GELU → Dropout → Linear</code>。</p></div></section>

        <section id="gradient"><span className="record-label">06 · GRADIENT DESCENT</span><h2>用 y = x² 看参数更新</h2><p>设 <code>x₀=4</code>、学习率 <code>η=0.1</code>。梯度为 <code>2x</code>，所以 <code>xₜ₊₁=xₜ−η·2xₜ=0.8xₜ</code>。</p><div className="table-wrap"><table><thead><tr><th>STEP</th><th>x</th><th>梯度</th><th>新 x</th><th>新 y</th></tr></thead><tbody><tr><td>0 → 1</td><td>4.000</td><td>8.000</td><td>3.200</td><td>10.240</td></tr><tr><td>1 → 2</td><td>3.200</td><td>6.400</td><td>2.560</td><td>6.554</td></tr><tr><td>2 → 3</td><td>2.560</td><td>5.120</td><td>2.048</td><td>4.194</td></tr></tbody></table></div><p>x 为正时向左移动，为负时向右移动，两侧最终都趋近最低点 <code>x=0</code>。</p></section>

        <section id="status"><span className="record-label">07 · LEARNING STATUS</span><h2>本日掌握边界</h2><ol className="step-list"><li>能串起 Attention、LayerNorm、FFN、Dropout 与两条残差支路。</li><li>理解 Linear 只处理最后一维，FFN 不混合 token。</li><li>掌握 reshape、transpose、contiguous 在 head 拆分与合并中的分工。</li><li>理解 Pre-Norm 梯度直通的直觉，但公式尚未细推。</li><li>参考实现已验证 <code>12 passed</code>；个人实现修正两处 head 变形后完成。</li></ol></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 12 · Transformer Action Policy</h2><Link href="/coding/day12-transformer-action-policy">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 11 · TRANSFORMER ENCODER</span></footer>
  </main>;
}
