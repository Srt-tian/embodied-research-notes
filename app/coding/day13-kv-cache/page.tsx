import Link from "next/link";
import { SiteHeader } from "../../site-header";

const cacheFlow = `x_new                         # (B,T_new,E)
q, current_k, current_v       # (B,H,T_new,D)

k = cat([K_past, current_k], dim=2)
v = cat([V_past, current_v], dim=2)
                              # (B,H,T_total,D)

scores = q @ k.transpose(-2,-1) / sqrt(D)
                              # (B,H,T_new,T_total)
weights = softmax(masked_scores, dim=-1)
context = weights @ v         # (B,H,T_new,D)

context = merge_heads(context) # (B,T_new,E)
output = out_proj(context)     # (B,T_new,E)`;

const maskCode = `query_positions = torch.arange(
    past_length,
    past_length + new_length,
    device=x.device,
)[:, None]                    # (T_new,1), absolute positions

key_positions = torch.arange(
    past_length + new_length,
    device=x.device,
)[None, :]                    # (1,T_total)

mask = key_positions <= query_positions
                              # (T_new,T_total)`;

const tokenSlice = `# 错误：整数索引会删除时间维
new_token = token_embeddings[:, timestep, :]
# shape: (B,E)

# 正确方案 1：长度为 1 的切片保留时间维
new_token = token_embeddings[:, timestep:timestep + 1, :]
# shape: (B,1,E)

# 正确方案 2：整数索引后显式补回时间维
new_token = token_embeddings[:, timestep, :].unsqueeze(1)
# shape: (B,1,E)`;

const mergeCode = `context = weights @ v          # (B,H,T_new,D)
context = context.transpose(1, 2).contiguous()
context = context.reshape(B, T_new, H * D)
                                  # (B,T_new,E)
output = out_proj(context)         # Linear(E,E)`;

export default function Day13KVCache() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 13</span></div>
      <div className="coding-note-title"><span>DAY 13 · 2026-07-27</span><h1>Transformer<br />KV Cache</h1><p>实现支持完整序列、分块输入与逐 token 推理的 causal self-attention。重点不是“存一个 cache”，而是保证绝对位置、张量维度和输出投影都严格一致。</p></div>
      <div className="coding-score"><span>IMPLEMENTATION · COMPLETE</span><b>04</b><i>core TODOs</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#flow">01 / 数据流</a><a href="#cache">02 / 缓存内容</a><a href="#mask">03 / 绝对位置 Mask</a><a href="#slice">04 / 保留时间维</a><a href="#heads">05 / 合并多头</a><a href="#tradeoff">06 / 计算与显存</a><a href="#review">07 / 严谨复盘</a></aside>
      <div className="coding-note-content">
        <section id="flow"><span className="record-label">01 · END-TO-END FLOW</span><h2>增量 Attention 的完整数据流</h2><pre className="code-block"><code>{cacheFlow}</code></pre><div className="shape-callout"><span>CACHED I/O</span><b>(B,T_new,E) → (B,T_new,E)</b><p>本轮只为新 token 生成 Q/K/V；历史 K/V 从 cache 复用。输出只对应本轮输入的 T_new 个 query。</p></div></section>

        <section id="cache"><span className="record-label">02 · WHAT IS CACHED</span><h2>为什么缓存 K/V，不缓存历史 Q</h2><p>历史 Q 只用于产生历史 token 的输出，而这些输出已经计算完成。causal attention 中，新 token 出现后，历史 token 不能反向关注它，因此旧输出不会改变。</p><div className="equation">Q_new × [K_past, K_new]ᵀ → attention weights → [V_past, V_new]</div><p>新一步仍需要当前 <code>Q_new</code> 与全部历史 K 做点积；cache 复用的是每一层历史 token 已投影得到的 K/V。</p></section>

        <section id="mask"><span className="record-label">03 · ABSOLUTE POSITION MASK</span><h2>有 cache 时不能只看 chunk 内相对位置</h2><pre className="code-block"><code>{maskCode}</code></pre><p>若 <code>past_length=3</code>、<code>T_new=2</code>，query 的绝对位置是 3、4，key 的绝对位置是 0～4。比较通过广播得到 <code>(T_new,T_total)</code>：每个 query 只允许关注位置不大于自己的 key。</p><div className="error-card"><span>MISLEADING EXAMPLE</span><h3><code>torch.arange(T_new)</code> 仅在无 cache 时可直接作为 query 位置</h3><p>孤立写 <code>torch.arange(2)[:,None]</code> 容易误导为始终使用 0、1。存在历史 cache 时必须加上 <code>past_length</code>，否则新 chunk 无法正确看到历史 key。</p></div></section>

        <section id="slice"><span className="record-label">04 · DIMENSION SAFETY</span><h2>逐 token 切片必须保留时间维</h2><pre className="code-block"><code>{tokenSlice}</code></pre><p><code>forward</code> 约定输入为 <code>(B,T_new,E)</code>。整数索引会消掉被索引的时间轴，而长度为 1 的切片会保留它；也可以在整数索引后用 <code>unsqueeze(1)</code> 补回。</p><div className="insight"><span>TIP 01</span><p><code>token_embeddings[:, timestep:timestep + 1, :]</code> 不是写法偏好，而是在维护模型接口的三维 shape 契约。</p></div></section>

        <section id="heads"><span className="record-label">05 · MULTI-HEAD OUTPUT</span><h2>先合并所有头，再经过最后的 Linear</h2><pre className="code-block"><code>{mergeCode}</code></pre><p>每个 head 得到自己的 <code>D=head_dim</code> 维 context。标准 multi-head attention 先把 H 个头拼回 <code>E=H×D</code>，再用一个 <code>nn.Linear(E,E)</code> 混合来自不同头的信息并映射到残差流所需的 embedding 维度。</p><div className="insight"><span>TIP 02</span><p>顺序是 <code>attention per head → merge heads → out_proj</code>。不能把 <code>out_proj(E,E)</code> 直接作用在仍为 <code>(B,H,T,D)</code> 的张量上，因为最后一维此时是 D，不是 E。</p></div></section>

        <section id="tradeoff"><span className="record-label">06 · COMPUTE–MEMORY TRADEOFF</span><h2>KV Cache 是用显存换计算</h2><div className="table-wrap"><table><thead><tr><th>项目</th><th>无 cache 的逐步生成</th><th>使用 KV Cache</th></tr></thead><tbody><tr><td>历史 K/V 投影</td><td>每一步重复计算</td><td>保存后直接复用</td></tr><tr><td>历史 query 的 attention</td><td>随整段前向重复计算</td><td>不再计算旧 query</td></tr><tr><td>新 query 对历史 K</td><td>需要</td><td>仍然需要</td></tr><tr><td>设备内存</td><td>不持久保存各层历史 K/V</td><td>随层数、batch 与序列长度增长</td></tr></tbody></table></div><p>GPU 推理时 cache 通常占用显存；CPU 推理时占用内存，也可使用 offload。它不会降低显存占用，而是额外保存每一层的历史 K/V，以避免大量重复计算。</p><div className="equation">KV elements ∝ 2 × B × L × T × H × D</div></section>

        <section id="review"><span className="record-label">07 · RIGOROUS REVIEW</span><h2>本次应保留的边界与修正</h2><ol className="step-list"><li>KV Cache 最适合旧输出不会被未来 token 改写的 causal self-attention；固定 encoder 输出的 cross-attention K/V 也可以缓存。</li><li>cache 并没有让新 token 的 attention 变成常数开销：<code>Q_new</code> 仍需读取并匹配全部历史 K，序列越长单步仍会变慢。</li><li>softmax 后变量应命名为 <code>weights</code>，<code>weights @ V</code> 应命名为 <code>context</code>，而不是 logits 或 new_v。</li><li>完整序列、任意合法分块与逐 token cached inference 应输出一致，这是验证 mask 和 cache 更新正确性的核心测试。</li><li>准确结论是“KV Cache 增加显存占用、减少重复计算”，不是“提高速度但不降低缓存占用”。</li></ol></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 14 · RoPE 与 KV Cache 位置对齐</h2><Link href="/coding/day14-rope-cache">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 13 · KV CACHE</span></footer>
  </main>;
}
