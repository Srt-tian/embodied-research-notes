import Link from "next/link";
import { SiteHeader } from "../../site-header";

const datasetCode = `@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self):
        self.states = np.asarray(self.states)
        self.actions = np.asarray(self.actions)
        if (
            self.states.ndim != 2
            or self.actions.ndim != 2
            or self.states.shape[0] == 0
            or self.states.shape[0] != self.actions.shape[0]
        ):
            raise ValueError("Invalid episode arrays.")
        self.length = self.states.shape[0]


class ActionChunkDataset(Dataset):
    def __init__(self, episodes, horizon, pad_value=0.0):
        if not episodes:
            raise ValueError("Episodes cannot be empty.")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")

        self.episodes = episodes
        self.horizon = horizon
        self.pad_value = pad_value
        self.state_dim = episodes[0].states.shape[1]
        self.action_dim = episodes[0].actions.shape[1]

        for episode in episodes:
            if (episode.states.shape[1] != self.state_dim
                    or episode.actions.shape[1] != self.action_dim):
                raise ValueError("Inconsistent dimensions.")

        self._index = [
            (episode_index, start)
            for episode_index, episode in enumerate(episodes)
            for start in range(episode.length)
        ]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        episode_index, start = self._index[index]
        episode = self.episodes[episode_index]
        state = torch.as_tensor(episode.states[start], dtype=torch.float32)
        actions = torch.full(
            (self.horizon, self.action_dim),
            self.pad_value,
            dtype=torch.float32,
        )
        mask = torch.zeros(self.horizon, dtype=torch.bool)
        valid_length = min(self.horizon, episode.length - start)
        actions[:valid_length] = torch.as_tensor(
            episode.actions[start:start + valid_length],
            dtype=torch.float32,
        )
        mask[:valid_length] = True
        return {"state": state, "actions": actions, "mask": mask}`;

const trainingCode = `def masked_mse_loss(predictions, targets, mask):
    if predictions.ndim != 3:
        raise ValueError("Predictions must be 3D.")
    if predictions.shape != targets.shape:
        raise ValueError("Prediction/target shapes must match.")
    if mask.shape != predictions.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Invalid mask.")
    if not torch.any(mask):
        raise ValueError("Mask needs at least one valid timestep.")

    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask.unsqueeze(-1)
    valid_elements = mask.sum() * predictions.shape[-1]
    return masked_error.sum() / valid_elements


def train_step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad()
    predictions = model(batch["state"])
    loss = masked_mse_loss(
        predictions, batch["actions"], batch["mask"]
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_batch(model, batch):
    model.eval()
    with torch.no_grad():
        predictions = model(batch["state"])
        loss = masked_mse_loss(
            predictions, batch["actions"], batch["mask"]
        )
    return loss.item()`;

export default function Day06ActionPolicy() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 06</span></div>
      <div className="coding-note-title"><span>DAY 06 · 2026-07-17</span><h1>端到端<br />Action Policy</h1><p>第一周综合项目：从可变长度机器人 episode 构造 Dataset，经 DataLoader 形成 batch，训练 state-conditioned action chunk policy。</p></div>
      <div className="coding-score"><span>FINAL SCORE · LEVEL A</span><b>9.0</b><i>/ 10</i></div>
    </section>
    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#task">01 / 题目</a><a href="#idea">02 / 思路</a><a href="#code">03 / 代码</a><a href="#tests">04 / 测试</a><a href="#review">05 / 复盘</a><a href="#knowledge">06 / 总结</a></aside>
      <div className="coding-note-content">
        <section id="task"><span className="record-label">01 · TASK · LEVEL A</span><h2>题目与约束</h2><p>实现完整训练链路：Episode 校验、全局样本索引、action chunk padding、valid mask、MLP、masked MSE、train/eval step。</p><div className="constraint-grid"><div><b>输入</b><p>多个形状为 (T_i,S) 与 (T_i,A) 的独立 episode。</p></div><div><b>输出</b><p>单样本 (S)、(H,A)、(H)，batch 后得到 (B,S)、(B,H,A)、(B,H)。</p></div></div></section>
        <section id="idea"><span className="record-label">02 · DATA FLOW</span><h2>解题思路</h2><ol className="step-list"><li>每个 episode 的每个 timestep 都作为一个样本起点，因此 Dataset 长度为 ΣT_i。</li><li>索引表保存 (episode_index,start)，同时定位轨迹和局部时间。</li><li>尾部剩余动作不足 H 时只在当前 episode 内取值，其余 padding。</li><li>DataLoader 对字典中同名 Tensor 沿新第 0 维 stack。</li><li>Masked MSE 的分母只计算有效 timestep × action_dim。</li></ol><div className="shape-callout"><span>PIPELINE</span><b>(T_i,S/A) → (B,H,A)</b><p>Episode → Dataset sample → DataLoader batch → ActionChunkMLP → masked loss。</p></div></section>
        <section id="code"><span className="record-label">03 · IMPLEMENTATION</span><h2>最终代码</h2><h3>Episode 与 Dataset</h3><pre className="code-block"><code>{datasetCode}</code></pre><h3>Masked loss 与训练</h3><pre className="code-block"><code>{trainingCode}</code></pre></section>
        <section id="tests"><span className="record-label">04 · VERIFICATION</span><h2>测试结果</h2><div className="test-result"><span>REFERENCE SUITE</span><b>24 passed in 8.54s</b><i>100%</i></div><p>覆盖空 episode、非法 horizon、跨 episode 泄漏、tail padding、dtype、DataLoader shape、模型输入、masked loss 和参数更新。</p></section>
        <section id="review"><span className="record-label">05 · ERROR REVIEW</span><h2>错误复盘</h2><div className="error-card"><span>VALIDATION</span><h3>压缩注释导致边界校验遗漏</h3><p>最初只检查 states/actions 长度相等，遗漏 T&gt;0；forward 和 masked loss 也未按测试契约主动验证 shape、dtype 与非空 mask。后续题目会把每项校验拆成清晰 checklist。</p></div><div className="error-card"><span>MASK</span><h3>完整 chunk 的 mask 未设为 True</h3><p>原实现仅在 padding 分支更新 mask，导致完整 chunk 全为 False。统一创建 padding buffer，再用 valid_length 同时填 actions 和 mask，消除了分支遗漏。</p></div><div className="error-card"><span>BOUNDARY</span><h3>Tail chunk 不能跨 episode</h3><p>episode 之间通常存在任务结束、环境 reset 和状态动作跳变。跨边界读取会制造不存在的连续轨迹，属于数据泄漏。</p></div></section>
        <section id="knowledge"><span className="record-label">06 · KNOWLEDGE</span><h2>知识点总结</h2><div className="knowledge-grid"><article><span>DATASET</span><h3>len 与 getitem</h3><p><code>__len__</code> 给出样本数；<code>__getitem__</code> 定义一条样本如何构造。</p></article><article><span>INDEX</span><h3>二维坐标映射</h3><p>(episode,start) 将扁平 Dataset 索引映射回独立轨迹与局部时间。</p></article><article><span>COLLATE</span><h3>自动增加 batch 维</h3><p>默认 collate 对同名、同 shape Tensor 执行 stack，新增第 0 维 B。</p></article><article><span>MASKED LOSS</span><h3>Padding 不进分母</h3><p>乘 mask 只清零分子；必须除以有效 timestep × A，或用布尔索引后 mean。</p></article><article><span>COPY</span><h3>Episode 只增加引用</h3><p><code>episode=self.episodes[i]</code> 不发生拷贝，只增加对同一 Episode 对象的引用。</p></article><article><span>VALIDATION</span><h3>逐项检查契约</h3><p>ndim、shape、非空、dtype、有效元素需要分别检查，避免一句注释掩盖遗漏。</p></article></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 7 · 完整 Epoch 训练与评估</h2><Link href="/coding/day07-epoch-trainer">进入 Day 7 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 06 · LEVEL A · VERIFIED</span></footer>
  </main>;
}
