import Link from "next/link";
import { SiteHeader } from "../../site-header";

const solution = `def create_action_chunks(episodes, horizon, pad_value=0.0):
    if horizon <= 0:
        raise ValueError(f"Invalid horizon: {horizon}. Must be positive.")
    if not episodes:
        return np.empty((0, horizon, 0)), np.empty((0, horizon), dtype=bool)

    validated_episodes = []
    D = None
    for episode in episodes:
        episode = np.asarray(episode)
        validated_episodes.append(episode)
        if episode.ndim != 2:
            raise ValueError(f"Episode must be 2D, got shape {episode.shape}.")
        if D is None:
            D = episode.shape[1]
        if episode.shape[1] != D:
            raise ValueError(
                f"Expected action dimension {D}, got {episode.shape[1]}."
            )

    chunks, masks = [], []
    for episode in validated_episodes:
        T = episode.shape[0]
        for start in range(T):
            end = start + horizon
            if end <= T:
                chunk = episode[start:end]
                mask = np.zeros(horizon, dtype=bool)
                mask[:end - start] = True
            else:
                chunk = np.full((horizon, D), pad_value, dtype=episode.dtype)
                chunk[:T - start] = episode[start:T]
                mask = np.zeros(horizon, dtype=bool)
                mask[:T - start] = True
            chunks.append(chunk)
            masks.append(mask)

    if not chunks:
        return np.empty((0, horizon, D)), np.empty((0, horizon), dtype=bool)
    return np.stack(chunks, axis=0), np.stack(masks, axis=0)`;

const structures = [
  ["list", "[1, 2, 3]", "是", "是", "是", "是", "收集尚未 stack 的 chunks"],
  ["tuple", "(1, 2, 3)", "是", "是", "否", "是", "保存 shape 等固定结构"],
  ["set", "{1, 2, 3}", "否", "否", "是", "否", "去重与成员查询"],
  ["dict", '{"action": x}', "插入顺序", "按 key", "是", "key 不重复", "组织机器人样本字段"],
  ["ndarray", "np.array(...) ", "是", "是", "通常是", "是", "图像、状态与动作计算"],
];

export default function Day01ActionChunks() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 01</span></div>
      <div className="coding-note-title"><span>DAY 01 · 2026-07-14</span><h1>Action Chunk<br />切片器</h1><p>从多个不同长度的机器人 episode 构造固定长度动作窗口，同时处理尾部 padding、valid mask 与 episode 边界。</p></div>
      <div className="coding-score"><span>FINAL SCORE</span><b>9.0</b><i>/ 10</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#task">01 / 题目</a><a href="#idea">02 / 思路</a><a href="#code">03 / 代码</a><a href="#tests">04 / 测试</a><a href="#review">05 / 复盘</a><a href="#knowledge">06 / 知识点</a></aside>
      <div className="coding-note-content">
        <section id="task"><span className="record-label">01 · TASK</span><h2>题目与约束</h2><p>输入多个形状为 <code>(T_i, D)</code> 的 episode。每个时间步都作为窗口起点，输出 <code>(N, horizon, D)</code> 的 chunks 和 <code>(N, horizon)</code> 的布尔 mask。</p><div className="constraint-grid"><div><b>允许</b><p>episode 的时间长度 T_i 不同</p></div><div><b>禁止</b><p>动作维度 D 不同或窗口跨越 episode 边界</p></div></div></section>

        <section id="idea"><span className="record-label">02 · APPROACH</span><h2>解题思路</h2><ol className="step-list"><li>先验证 horizon，再逐个将 episode 转为 ndarray。</li><li>只要求 <code>shape[1]</code> 相同，允许 <code>shape[0]</code> 不同。</li><li>对每个 episode 的每个 start 构造一个局部 chunk。</li><li>真实动作写入 chunk 前部，剩余位置保持 pad_value。</li><li>同步生成 mask，最后沿样本维 stack。</li></ol><div className="shape-callout"><span>OUTPUT SHAPE</span><b>N × H × D</b><p>chunk 样本数 × 动作窗口长度 × 单步动作维度。N 是数据集样本数，不等同于训练时的 batch size。</p></div></section>

        <section id="code"><span className="record-label">03 · IMPLEMENTATION</span><h2>最终代码</h2><pre className="code-block"><code>{solution}</code></pre></section>

        <section id="tests"><span className="record-label">04 · VERIFICATION</span><h2>测试结果</h2><div className="test-result"><span>PYTEST</span><b>8 passed in 0.05s</b><i>100%</i></div><p>在既定测试之外，额外验证了三项边界情况：空输入不能绕过非法 horizon 校验；Python list episode 能被正确转换；零长度 <code>(0, D)</code> episode 能返回正确空 shape。</p><div className="test-result secondary"><span>EXTRA CHECKS</span><b>3 passed</b><i>VERIFIED</i></div></section>

        <section id="review"><span className="record-label">05 · ERROR REVIEW</span><h2>错误复盘</h2><div className="error-card"><span>INDEX SPACE</span><h3>混淆 episode 全局索引与 chunk 局部索引</h3><code>chunk[start:T] = episode[start:T]</code><p>start 属于原 episode 的时间坐标；每个新 chunk 都从局部索引 0 开始。错误写法导致将 shape <code>(2,1)</code> 的数据写入空切片 <code>(0,1)</code>。</p><strong>修正：chunk[:T - start] = episode[start:T]</strong></div><div className="error-card"><span>BOUNDARY</span><h3>输入校验顺序与转换结果保存</h3><p>空输入判断不能放在 horizon 校验之前；<code>np.asarray()</code> 的返回值需要收集到新列表，否则原始 Python list 在后续仍没有 shape 属性。</p></div></section>

        <section id="knowledge"><span className="record-label">06 · KNOWLEDGE</span><h2>知识点总结</h2><h3>Python 与 NumPy 常见数据结构</h3><div className="table-wrap"><table><thead><tr><th>类型</th><th>写法</th><th>有序</th><th>索引</th><th>可修改</th><th>重复</th><th>机器人场景</th></tr></thead><tbody>{structures.map((row) => <tr key={row[0]}>{row.map((cell, index) => <td key={index}><code>{cell}</code></td>)}</tr>)}</tbody></table></div><div className="knowledge-grid"><article><span>SHAPE</span><h3>.shape 返回 tuple</h3><p><code>(N, H, D)</code> 是有序、可索引、不可修改的固定结构，可以直接解包。</p></article><article><span>DTYPE</span><h3>赋值不会改变目标 dtype</h3><p>使用 <code>np.full(..., dtype=episode.dtype)</code>，避免 float32 动作被整数数组截断。</p></article><article><span>MEMORY</span><h3>view 与 copy</h3><p>数组切片通常返回共享内存的 view；切片赋值会把数值写入目标数组的存储空间。</p></article><article><span>PADDING</span><h3>full、zeros 与 empty</h3><p>full 适合 padding，zeros 适合 mask；empty 只用于确定全部元素会被覆盖或返回零尺寸数组。</p></article></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 2 · 轨迹数据结构与在线统计</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 01 · VERIFIED</span></footer>
  </main>;
}
