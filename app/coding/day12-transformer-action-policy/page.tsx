import Link from "next/link";
import { SiteHeader } from "../../site-header";

const policyFlow = `states                         # (B,T,state_dim)
x = input_projection(states)   # (B,T,E)
x = x + position_embedding[:, :T, :]

for block in blocks:
    x = block(x, attention_mask)

x = final_norm(x)              # (B,T,E)
last_features = x[batch_ids, last_indices]  # (B,E)
actions = action_head(last_features)        # (B,H*A)
actions = actions.reshape(B, H, A)          # (B,H,A)`;

const maskCode = `pairwise_valid = (
    valid_mask[:, :, None]      # (B,T,1): query 是否有效
    & valid_mask[:, None, :]    # (B,1,T): key 是否有效
)                              # (B,T,T)

identity = torch.eye(T, dtype=torch.bool, device=device)[None]
attention_mask = pairwise_valid | (
    ~valid_mask[:, :, None] & identity
)`;

const lastCode = `# 仅在 valid token 从左到右连续时成立
last_indices = valid_mask.sum(dim=1) - 1  # (B,)

last_features = x[
    torch.arange(batch_size, device=x.device),
    last_indices,
]  # (B,E)`;

export default function Day12TransformerActionPolicy() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 12</span></div>
      <div className="coding-note-title"><span>DAY 12 · 2026-07-24</span><h1>Transformer<br />Action Policy</h1><p>接上 Day 11 的 Encoder Block：处理 padded state sequence，安全构造 attention mask，汇聚最后有效 token，并预测固定长度 action chunk。</p></div>
      <div className="coding-score"><span>IMPLEMENTATION · COMPLETE</span><b>05</b><i>core TODOs</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#flow">01 / 数据流</a><a href="#mask">02 / Attention Mask</a><a href="#padding">03 / Padding 数值安全</a><a href="#pooling">04 / 序列汇聚</a><a href="#head">05 / Action Head</a><a href="#review">06 / 错误复盘</a><a href="#status">07 / 掌握边界</a></aside>
      <div className="coding-note-content">
        <section id="flow"><span className="record-label">01 · END-TO-END FLOW</span><h2>从状态序列到动作块</h2><pre className="code-block"><code>{policyFlow}</code></pre><div className="shape-callout"><span>POLICY I/O</span><b>(B,T,state_dim) → (B,H,A)</b><p>Transformer 保留每个 timestep 的表示；policy 从中选择一个序列级特征，再一次性回归 H 个动作。</p></div></section>
        <section id="mask"><span className="record-label">02 · PAIRWISE MASK</span><h2>(B,T) 如何变成 (B,T,T)</h2><pre className="code-block"><code>{maskCode}</code></pre><p><code>attention_mask[b,i,j]</code> 表示第 b 个样本中，query 位置 i 是否允许关注 key 位置 j。第一个 T 是 query 轴，第二个 T 是 key 轴；两个扩维后的 mask 通过广播组合成所有 query-key 配对。</p><div className="insight"><span>MENTAL MODEL</span><p><code>(B,T,T) = (batch, query position, key position)</code>。真实 query 只看真实 key，padding key 不进入真实 token 的信息汇聚。</p></div></section>
        <section id="padding"><span className="record-label">03 · NUMERICAL SAFETY</span><h2>Padding query 必须至少有一个可见位置</h2><p>若 padding query 的 score 整行都被填为 <code>-inf</code>，softmax 内部会遇到 <code>-inf - (-inf)</code>，结果产生 NaN。虽然 padding query 最终不会被用于预测，它仍会经过 Transformer 计算，NaN 可能污染后续结果。</p><p>本实现允许每个 padding query 只关注自己，使每一行至少保留一个有限 score；这是数值安全处理，不是让 padding token 参与有效信息建模。</p></section>
        <section id="pooling"><span className="record-label">04 · SEQUENCE POOLING</span><h2>选择最后一个有效 token</h2><pre className="code-block"><code>{lastCode}</code></pre><p>双向 Self-Attention 中，每个有效 token 都能看到全部有效序列，因此取最后一个 token 是 pooling 设计，而不是信息覆盖上的唯一选择。若改成 causal Attention，最后一个有效 token 才拥有完整历史，选择它会有更强的结构理由。</p><div className="error-card"><span>HIDDEN ASSUMPTION</span><h3><code>sum(mask) - 1</code> 只适用于右侧连续 padding</h3><p>对 <code>[T,F,T,F]</code>，True 数量为 2，计算得到下标 1，但最后一个 True 实际在下标 2。当前 Dataset 必须保证 mask 形如 <code>[T,T,T,F,F]</code>。</p></div></section>
        <section id="head"><span className="record-label">05 · ACTION HEAD</span><h2>一个 Linear 预测完整 chunk</h2><div className="equation">Linear(E, H × A) → reshape(B, H, A)</div><p><code>nn.Linear</code> 只变换最后一维：输入 <code>last_features (B,E)</code>，输出每个样本的 <code>H×A</code> 个数，再拆成 horizon 与 action dimension。每个未来动作都由同一个序列级特征联合预测。</p></section>
        <section id="review"><span className="record-label">06 · ERROR REVIEW</span><h2>本次实现修正</h2><div className="error-card"><span>POSITION EMBEDDING</span><h3>当前 T 与 max_sequence_length 不相等</h3><p><code>(B,5,E) + (1,6,E)</code> 无法广播。位置参数按最大长度创建，但 forward 中必须截取 <code>[:, :sequence_length, :]</code>。</p></div><div className="error-card"><span>MODULELIST</span><h3>将模块容器直接当网络调用</h3><p><code>nn.ModuleList</code> 负责注册并保存模块，不定义 forward 数据流；需要遍历并把 <code>x</code> 与 <code>attention_mask</code> 传给每个 block。</p></div><div className="error-card"><span>ENVIRONMENT</span><h3>PyTorch 无法初始化 NumPy</h3><p><code>No module named numpy</code> 是虚拟环境缺依赖的 warning，与 Transformer 实现无关；涉及 Tensor/NumPy 互转时会成为实际问题。</p></div></section>
        <section id="status"><span className="record-label">07 · LEARNING STATUS</span><h2>本日真正需要带走的内容</h2><ol className="step-list"><li>能串起 state projection、position embedding、Encoder blocks、pooling 与 action head 的完整 shape 流。</li><li>理解 pairwise attention mask 的 query/key 两个 T 维，以及 padding query 的数值安全处理。</li><li>知道双向与 causal Attention 下“最后 token 汇聚”的不同理由。</li><li>识别 <code>sum(mask)-1</code> 对右侧连续 padding 的隐含假设。</li><li><code>expand</code>、高级索引与 <code>gather</code> 作为 API 用法掌握，不再作为后续检验题重点。</li></ol></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 13 · Transformer KV Cache</h2><Link href="/coding/day13-kv-cache">继续阅读 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 12 · TRANSFORMER ACTION POLICY</span></footer>
  </main>;
}
