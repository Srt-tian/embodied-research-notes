import Link from "next/link";
import { SiteHeader } from "../../site-header";

const rotationCode = `x_even = x[..., 0::2]  # (B,H,T,D/2)
x_odd  = x[..., 1::2]  # (B,H,T,D/2)

rotated_even = x_even * cos - x_odd * sin
rotated_odd  = x_even * sin + x_odd * cos

rotated = torch.stack(
    [rotated_even, rotated_odd],
    dim=-1,
)                           # (B,H,T,D/2,2)
return rotated.flatten(-2)  # (B,H,T,D)`;

const applyCode = `cos = self.cos_table[position_ids][None, None, :, :]
sin = self.sin_table[position_ids][None, None, :, :]
                              # (1,1,T,D/2)

q = self._rotate_pairs(q, cos, sin)
k = self._rotate_pairs(k, cos, sin)
return q, k`;

const positionCode = `past_length = 0 if cache is None else cache.key.shape[2]
new_length = x.shape[1]

position_ids = torch.arange(
    past_length,
    past_length + new_length,
    device=x.device,
)

# 只旋转本轮 Q/K，再把 current_k 写入 cache
q, current_k = self.rope(q, current_k, position_ids)`;

const cacheOrder = `current_q, current_k
        ↓  使用全局绝对位置 [T_past, ..., T_total - 1]
apply_rope(current_q, current_k)
        ↓
rotated current_k
        ↓
cat([cached_rotated_k, rotated_current_k], dim=2)`;

export default function Day14RoPECache() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 14</span></div>
      <div className="coding-note-title"><span>DAY 14 · 2026-07-29</span><h1>RoPE ×<br />KV Cache</h1><p>理解旋转位置编码如何通过 Q/K 点积表达相对位置，并保证完整前向、分块与逐 token cached inference 使用一致的位置坐标系。</p></div>
      <div className="coding-score"><span>CORE CONCEPTS · COMPLETE</span><b>03</b><i>key decisions</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#goal">01 / 训练取舍</a><a href="#principle">02 / RoPE 原理</a><a href="#rotate">03 / 通道旋转</a><a href="#apply">04 / 正确替换 Q/K</a><a href="#cache">05 / Cache 位置对齐</a><a href="#why">06 / 为什么不旋转 V</a><a href="#review">07 / 错误复盘</a></aside>
      <div className="coding-note-content">
        <section id="goal"><span className="record-label">01 · TRAINING SCOPE</span><h2>不是每个底层细节都值得手搓</h2><p>本题最初要求完整实现频率表、相邻通道旋转和 cache 位置对齐。复盘后重新划分训练价值：日常具身工程中，更重要的是能读懂 RoPE 数据流、判断位置是否对齐，并用一致性测试定位 cache 错误，而不是反复默写成熟框架已有的底层实现。</p><div className="table-wrap"><table><thead><tr><th>层级</th><th>本次内容</th><th>要求</th></tr></thead><tbody><tr><td>A · 必须掌握</td><td>RoPE 作用于 Q/K；绝对位置旋转后，点积体现相对位置差</td><td>能独立解释和审计</td></tr><tr><td>A · 值得动手</td><td>cache 存在时从 past_length 构造 position_ids</td><td>能独立实现</td></tr><tr><td>B · 理解会改</td><td>偶/奇通道配对、stack 后 flatten 交错还原</td><td>能阅读、修改、排错</td></tr><tr><td>C · 暂不手搓</td><td>完整频率表、长上下文外推与各种 RoPE 变体</td><td>知道作用与边界</td></tr></tbody></table></div><div className="insight"><span>TRAINING PRINCIPLE</span><p>会直接影响模型正确性、训练推理一致性且高频出现的部分，才作为核心 TODO；成熟实现和样板代码以阅读、修改为主。</p></div></section>

        <section id="principle"><span className="record-label">02 · CORE PRINCIPLE</span><h2>分别使用绝对位置旋转，点积得到相对位置</h2><p>RoPE 把 Q/K 最后一维的相邻通道视为二维向量。位置 m 的 Q 旋转 mθ，位置 n 的 K 旋转 nθ。旋转矩阵的乘法性质会让点积中的绝对角度相消，只留下位置差。</p><div className="equation">[R(mθ)Q]ᵀ[R(nθ)K] = QᵀR((n-m)θ)K</div><p>因此，RoPE 的输入是每个 token 的绝对位置，但最终 attention score 自然包含 query 与 key 的相对距离。不同通道对使用不同频率，从多个尺度表达位置关系。</p></section>

        <section id="rotate"><span className="record-label">03 · PAIRWISE ROTATION</span><h2>偶数、奇数通道配对旋转</h2><pre className="code-block"><code>{rotationCode}</code></pre><p><code>torch.stack(..., dim=-1)</code> 新增一个长度为 2 的维度，把每对旋转后的 even/odd 放在一起；<code>flatten(-2)</code> 再将最后两个维度合并，恢复为交错顺序和原始 head_dim。</p><div className="shape-callout"><span>PAIR FLOW</span><b>D → (D/2, 2) → D</b><p>这部分实现正确：最后的通道顺序仍是 even₀、odd₀、even₁、odd₁……</p></div></section>

        <section id="apply"><span className="record-label">04 · APPLY ROPE</span><h2>旋转结果直接替换 Q/K</h2><pre className="code-block"><code>{applyCode}</code></pre><p><code>cos_table[position_ids]</code> 与 <code>sin_table[position_ids]</code> 先按 token 位置查表，再补出 batch 和 head 两个广播维。<code>_rotate_pairs</code> 返回的已经是完整旋转结果，不是需要加到原向量上的位置增量。</p><div className="error-card"><span>KEY BUG</span><h3>错误：<code>q = q + rotate(q)</code></h3><code>position 0: cos = 1, sin = 0 → rotate(q) = q → q + rotate(q) = 2q</code><p>位置 0 本应不改变向量，却会被放大为两倍。因此正确写法是直接赋值：<code>q = rotate(q)</code>、<code>k = rotate(k)</code>。</p></div></section>

        <section id="cache"><span className="record-label">05 · ABSOLUTE POSITION CONTINUITY</span><h2>新 token 必须延续整条序列的位置</h2><pre className="code-block"><code>{positionCode}</code></pre><p>若 cache 已有位置 0、1、2，新 chunk 的两个 token 就必须使用位置 3、4。若每个 chunk 都重新从 0 编号，它与历史 K 的相对位置会错乱，完整前向与 cached inference 也不再一致。</p><pre className="code-block"><code>{cacheOrder}</code></pre><p>缓存里的 K 已按原始位置旋转过，不能再次旋转。正确顺序是：只旋转本轮 <code>current_k</code>，再与 <code>cache.key</code> 拼接。</p><div className="insight"><span>PYTORCH MODULE CALL</span><p><code>self.rope(q, current_k, position_ids)</code> 会通过 <code>nn.Module.__call__()</code> 执行 <code>RotaryEmbedding.forward()</code>，并保留 hooks、autocast 等模块机制。</p></div></section>

        <section id="why"><span className="record-label">06 · Q/K, NOT V</span><h2>位置要影响“关注谁”，不是改变内容值</h2><div className="equation">scores = rotated_q @ rotated_kᵀ<br />output = softmax(scores) @ v</div><p>Q/K 直接决定 attention score，即关注哪个 token、权重是多少。RoPE 的目标就是让这个分数带有位置关系。V 在权重确定后提供被汇总的内容，不参与 score 计算，因此标准 RoPE 通常不旋转 V。</p></section>

        <section id="review"><span className="record-label">07 · REVIEW</span><h2>本次完成情况与验证标准</h2><ol className="step-list"><li>正确实现相邻通道的二维旋转，并用 <code>stack + flatten</code> 交错还原。</li><li>发现并修正把完整旋转结果误当作增量、再次加回 Q/K 的错误。</li><li>完成 cache 下的绝对 <code>position_ids</code>：范围为 <code>[past_length, past_length + new_length)</code>。</li><li>明确 cached K 已旋转，只有 <code>current_k</code> 在进入 cache 前旋转一次。</li><li>核心验证应包括：位置 0 不改变输入、每对通道范数保持、完整前向与逐 token/任意分块输出一致。</li></ol><div className="test-result"><span>REVIEW STATUS</span><b>CORE LOGIC VERIFIED</b><i>DAY 14 COMPLETE</i></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 15 · Flow Matching Path</h2><Link href="/coding/day15-flow-matching-path">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 14 · ROPE × KV CACHE</span></footer>
  </main>;
}
