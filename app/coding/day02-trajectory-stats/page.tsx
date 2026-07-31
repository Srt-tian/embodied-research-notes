import Link from "next/link";
import { SiteHeader } from "../../site-header";

const episodeCode = `@dataclass
class Episode:
    timestamps: np.ndarray
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps)
        self.states = np.asarray(self.states)
        self.actions = np.asarray(self.actions)

        if self.timestamps.ndim != 1:
            raise ValueError("timestamps != 1D")
        if self.states.ndim != 2:
            raise ValueError("states != 2D")
        if self.actions.ndim != 2:
            raise ValueError("actions != 2D")
        if not (len(self.timestamps) == len(self.states) == len(self.actions)):
            raise ValueError("All arrays must have the same length.")
        if not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("Timestamps must be strictly increasing.")`;

const statsCode = `class RunningStats:
    def __init__(self, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.count = 0
        self._mean = np.zeros(feature_dim, dtype=np.float64)
        self._m2 = np.zeros(feature_dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"Expected (B, {self.feature_dim}), got {values.shape}")
        if values.shape[0] == 0:
            return

        batch_count = values.shape[0]
        batch_mean = np.mean(values, axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)

        if self.count == 0:
            self.count, self._mean, self._m2 = batch_count, batch_mean, batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self._mean
        self._mean += delta * (batch_count / total_count)
        self._m2 += batch_m2 + delta ** 2 * (
            self.count * batch_count / total_count
        )
        self.count = total_count

    def _require_data(self) -> None:
        if self.count == 0:
            raise RuntimeError("No samples have been observed yet.")

    @property
    def mean(self):
        self._require_data()
        return self._mean.copy()

    @property
    def variance(self):
        self._require_data()
        return self._m2 / self.count

    @property
    def std(self):
        return np.sqrt(self.variance)`;

export default function Day02TrajectoryStats() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 02</span></div>
      <div className="coding-note-title"><span>DAY 02 · 2026-07-15</span><h1>轨迹结构<br />与流式统计</h1><p>用 dataclass 建模时间对齐的机器人 episode，并通过批量 Welford 合并算法稳定地累计逐特征均值、方差与标准差。</p></div>
      <div className="coding-score"><span>FINAL SCORE</span><b>9.2</b><i>/ 10</i></div>
    </section>

    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#task">01 / 题目</a><a href="#idea">02 / 思路</a><a href="#code">03 / 代码</a><a href="#tests">04 / 测试</a><a href="#review">05 / 复盘</a><a href="#knowledge">06 / 知识点</a></aside>
      <div className="coding-note-content">
        <section id="task"><span className="record-label">01 · TASK</span><h2>题目与约束</h2><p><code>Episode</code> 接收 timestamps、states、actions，验证维度、时间长度和时间戳顺序；<code>RunningStats</code> 接收任意数量的二维 batch，在不保存历史样本的前提下维护逐特征总体统计量。</p><div className="constraint-grid"><div><b>时间对齐</b><p>三个数组的第 0 维都必须是同一个 T；states 与 actions 的特征维可以不同。</p></div><div><b>统计约定</b><p>输入为 (B, D)，沿 axis=0 聚合，输出保留 D；方差使用总体方差 ddof=0。</p></div></div></section>

        <section id="idea"><span className="record-label">02 · APPROACH</span><h2>解题思路</h2><ol className="step-list"><li>让 dataclass 自动生成构造函数，在 <code>__post_init__</code> 中统一转换和校验。</li><li>时间戳是一维；状态和动作是二维；三个数组逐时间步一一对齐。</li><li>每批先计算 count、mean 与离均差平方和 M2。</li><li>用两批均值之差 delta 修正跨批次差异，再合并全局 mean 与 M2。</li><li>属性访问前检查是否已有数据，并对 mean 返回 defensive copy。</li></ol><div className="shape-callout"><span>STATISTICS</span><b>(B, D) → (D,)</b><p>消去的是 batch/sample 轴；每个特征各自拥有一个均值与方差，不是在特征之间求统计量。</p></div></section>

        <section id="code"><span className="record-label">03 · IMPLEMENTATION</span><h2>最终代码</h2><h3>Episode：结构与校验</h3><pre className="code-block"><code>{episodeCode}</code></pre><h3>RunningStats：流式合并</h3><pre className="code-block"><code>{statsCode}</code></pre></section>

        <section id="tests"><span className="record-label">04 · VERIFICATION</span><h2>测试结果</h2><div className="test-result"><span>PYTEST</span><b>22 passed in 0.07s</b><i>100%</i></div><p>覆盖 dataclass 初始化、数组维度与长度校验、严格递增时间戳、空 batch、首批初始化、多批合并、总体方差、标准差以及 mean 的防御性复制。</p></section>

        <section id="review"><span className="record-label">05 · ERROR REVIEW</span><h2>错误复盘</h2><div className="error-card"><span>DATACLASS LIFECYCLE</span><h3>手写 __init__ 导致 __post_init__ 不再自动执行</h3><p><code>@dataclass</code> 生成的 <code>__init__</code> 会在字段赋值后调用 <code>self.__post_init__()</code>。覆盖它以后，这条生命周期链路被切断，转换与验证全部失效。</p><strong>修正：删除自定义 __init__，让 __post_init__(self) 专注后处理。</strong></div><div className="error-card"><span>CONCEPT CALIBRATION</span><h3>对齐的是时间步，不是 parquet 文件</h3><p>同一 episode 中，timestamps[t]、states[t]、actions[t] 描述同一时刻；存储格式可以是 Parquet，也可以是 NumPy、HDF5 等，它不是长度一致的根本原因。</p></div></section>

        <section id="knowledge"><span className="record-label">06 · KNOWLEDGE</span><h2>知识点总结</h2><div className="knowledge-grid"><article><span>DATACLASS</span><h3>声明字段，自动构造</h3><p><code>__post_init__</code> 适合类型转换、派生字段和跨字段校验，不需要再次接收构造参数。</p></article><article><span>AXIS</span><h3>axis=0 消去样本轴</h3><p>对 (B, D) 输入得到 (D,)，表示为 D 个特征分别统计，而不是混合不同特征。</p></article><article><span>BROADCASTING</span><h3>(B, D) − (D,)</h3><p>batch_mean 沿 B 轴广播到每一行，然后计算每个样本相对本特征均值的偏差。</p></article><article><span>STREAMING</span><h3>固定内存与数值稳定性</h3><p>无需保存全部历史数据；批量 Welford 合并比直接累计 sum 与 sum-of-squares 更能减少大数相减造成的精度损失。</p></article><article><span>DIFF / ALL</span><h3>严格递增检查</h3><p><code>np.diff(timestamps) &gt; 0</code> 产生逐间隔布尔数组，<code>np.all</code> 要求所有间隔都为正。</p></article><article><span>COPY</span><h3>保护内部状态</h3><p>数值 ndarray 的 <code>copy()</code> 创建独立数据；调用方修改返回值不会污染统计器内部均值。</p></article></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 3 · 待训练表现评估后确定</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 02 · VERIFIED</span></footer>
  </main>;
}
