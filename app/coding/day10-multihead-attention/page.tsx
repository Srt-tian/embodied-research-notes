import Link from "next/link";
import { SiteHeader } from "../../site-header";

const attentionCode = `def forward(self, x, attention_mask=None):
    q = self._split_heads(self.q_proj(x))
    k = self._split_heads(self.k_proj(x))
    v = self._split_heads(self.v_proj(x))

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(self.head_dim)

    if attention_mask is not None:
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(1)
        scores = scores.masked_fill(~attention_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    context = weights @ v
    context = self._merge_heads(context)
    return self.out_proj(context)`;

const shapeCode = `x                         (B, T, E)
q_proj / k_proj / v_proj (B, T, E)
reshape                   (B, T, H, D)
transpose(1, 2)           (B, H, T, D)
Q @ K.transpose(-2, -1)   (B, H, T, T)
softmax(scores, dim=-1)   (B, H, T, T)
weights @ V               (B, H, T, D)
transpose + reshape       (B, T, E)
out_proj                  (B, T, E)`;

export default function Day10MultiHeadAttention() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 10</span></div>
      <div className="coding-note-title"><span>DAY 10 · 2026-07-22</span><h1>Multi-Head<br />Self-Attention</h1><p>第一次经典网络核心模块复现：不用封装好的 attention API，从 Q、K、V 投影开始，亲手完成多头拆分、缩放点积、mask、归一化、Value 聚合与输出投影。</p></div>
      <div className="coding-score"><span>STATUS · TESTED</span><b>11</b><i>tests passed</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#flow">01 / 张量流</a><a href="#formula">02 / 注意力公式</a><a href="#mask">03 / Causal Mask</a><a href="#code">04 / 最终实现</a><a href="#review">05 / 错误复盘</a><a href="#storage">06 / Tensor 存储</a><a href="#concepts">07 / 概念校正</a></aside>
      <div className="coding-note-content">
        <section id="flow"><span className="record-label">01 · TENSOR FLOW</span><h2>从输入到输出的完整 shape</h2><pre className="code-block"><code>{shapeCode}</code></pre><div className="shape-callout"><span>HEAD SPLIT</span><b>E = H × D</b><p><code>embed_dim</code> 必须能被 <code>num_heads</code> 整除，多个 head 的结果才可以重新拼回原 embedding 维度。</p></div></section>

        <section id="formula"><span className="record-label">02 · SCALED DOT-PRODUCT</span><h2>一个 Query 如何读取上下文</h2><p>每个 Query 与所有 Key 做点积，得到它对各个位置的相关性；沿 Key 维度做 softmax 后，每个 Query 分配给所有 Key 的权重之和为 1，最后用这些权重加权对应的 Value。</p><div className="constraint-grid"><div><b>为什么除以 √D</b><p>head_dim 增大时，点积幅值通常也会增大。缩放可避免 softmax 过度饱和，从而保留更健康的梯度。</p></div><div><b>为什么 dim=-1</b><p><code>scores</code> 最后一个维度枚举 Key。对每个 Query，都应在所有 Key 位置上形成一组概率分布。</p></div></div></section>

        <section id="mask"><span className="record-label">03 · CAUSAL MASK</span><h2>未来 token 仍被计算，但不能被读取</h2><p>causal mask 中 <code>True</code> 表示允许关注，<code>False</code> 表示禁止关注。因此要用 <code>~attention_mask</code> 找到被禁止的位置，把对应 score 替换为负无穷；softmax 后其权重变为 0。</p><pre className="code-block"><code>{`True  False False
True  True  False
True  True  True

scores.masked_fill(~attention_mask, float("-inf"))`}</code></pre><p>未来 token 的 K、V 通常仍在并行前向中生成。被阻止的是“当前 Query 读取未来 Key/Value”的路径，而不是未来 Tensor 根本不存在。</p></section>

        <section id="code"><span className="record-label">04 · FINAL IMPLEMENTATION</span><h2>注意力核心实现</h2><pre className="code-block"><code>{attentionCode}</code></pre><div className="test-result"><span>PYTEST</span><b>11 passed</b><i>shape、梯度、手算对齐与 causal 行为测试</i></div></section>

        <section id="review"><span className="record-label">05 · ERROR REVIEW</span><h2>两次失败对应两条公式细节</h2><div className="error-card"><span>SCALING</span><h3>漏掉 <code>/ sqrt(head_dim)</code></h3><code>scores = q @ k.transpose(-2, -1)</code><p>矩阵形状完全正确，但数值公式不完整，导致单头输出无法与手算结果对齐。修正为 <code>scores / math.sqrt(self.head_dim)</code>。</p></div><div className="error-card"><span>MASK SEMANTICS</span><h3>把允许位置屏蔽了</h3><code>scores.masked_fill(attention_mask, float("-inf"))</code><p>题目约定 True 表示允许，上式却将 True 位置设为负无穷。正确写法是 <code>masked_fill(~attention_mask, ...)</code>，屏蔽 False 位置。</p></div><p>其余实现——QKV 独立投影、多头拆分与合并、矩阵乘法、softmax 维度和输出投影——均正确。</p></section>

        <section id="storage"><span className="record-label">06 · TENSOR STORAGE</span><h2>reshape、transpose 与 contiguous</h2><div className="knowledge-grid"><article><span>TRANSPOSE</span><h3>换 stride，不主动复制</h3><p>返回新的 Tensor 视图，通常与原 Tensor 共享底层存储。它只交换维度解释，结果往往不连续。</p></article><article><span>RESHAPE</span><h3>优先视图，必要时复制</h3><p>内存布局允许时共享存储；不允许时会自动复制。因此不能依赖它一定共享或一定独立。</p></article><article><span>CONTIGUOUS</span><h3>按当前维度顺序重排</h3><p>合并 heads 前调用它，是为了得到连续布局，让后续 reshape 按正确顺序合并维度，并非为了“深拷贝保护”。</p></article><article><span>CLONE</span><h3>明确需要独立数据时使用</h3><p>普通前向计算没有原地修改，通常不需要 clone。只有准备原地改写且不能影响共享存储时才显式复制。</p></article></div></section>

        <section id="concepts"><span className="record-label">07 · CONCEPT CHECK</span><h2>多头不是直接切原始 embedding</h2><ol className="step-list"><li>Q、K、V 各自由完整的 <code>embed_dim → embed_dim</code> 线性层生成，然后才沿输出维度拆成多个 head。</li><li>因此一个 head 虽只拿到投影结果的一段，但其中每个元素理论上都可以依赖当前 token 的全部输入 embedding 维度。</li><li>这里的“完整信息”仅指当前 token 在 embedding 维度上的完整混合信息，不代表已经包含序列中其他 token 的上下文。</li><li>线性投影逐 token 独立执行；跨 token 的全局上下文，要到 <code>softmax(QKᵀ)V</code> 聚合之后才产生。</li><li>工程中也可用一个大线性层一次生成拼接的 QKV，再切成三份；与三个独立投影层在表达形式上等价。</li></ol></section>
      </div>
    </article>

    <section className="next-note shell"><span>NEXT</span><h2>Transformer Block · 即将继续</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 10 · MULTI-HEAD ATTENTION</span></footer>
  </main>;
}
