"use client";

import Link from "next/link";
import { useState } from "react";
import { SiteHeader } from "../../site-header";

const chapters = [
  { id: "overview", label: "01 核心结论" },
  { id: "fastweights", label: "02 Fast Weights" },
  { id: "architecture", label: "03 网络结构" },
  { id: "training", label: "04 双层优化" },
  { id: "sequence", label: "05 长序列训练" },
  { id: "context", label: "06 Context 学习" },
  { id: "inference", label: "07 在线推理" },
  { id: "results", label: "08 实验结果" },
  { id: "audit", label: "09 评价与边界" },
];

export default function RoboTTTNote() {
  const [active, setActive] = useState("overview");

  return (
    <main>
      <SiteHeader />
      <section className="paper-head shell">
        <div className="paper-breadcrumb"><Link href="/notes">NOTES</Link><span>/</span><span>RoboTTT</span></div>
        <div className="paper-title masquerade-title"><span>NOTE 005 · LONG-CONTEXT / TTT</span><h1>RoboTTT</h1><p>Context Scaling for<br />Robot Policies</p></div>
        <div className="paper-meta"><div><span>VERSION</span><b>ARXIV V1 · JUL 2026</b></div><div><span>BACKBONE</span><b>GR00T N1.7</b></div><div><span>STATUS</span><b>ANALYZED</b></div></div>
      </section>

      <section className="paper-body shell">
        <aside className="paper-toc"><span>CONTENTS</span>{chapters.map(chapter => <button className={active === chapter.id ? "active" : ""} onClick={() => setActive(chapter.id)} key={chapter.id}>{chapter.label}</button>)}</aside>
        <article className="paper-content">
          {active === "overview" && <>
            <div className="evidence-label">论文明确说明 · Figures 1-2 / Sections 1-3</div>
            <p className="paper-lead">RoboTTT 在 GR00T N1.7 的 DiT action head 中插入可在线更新的 TTT-MLP，把最长 8K timestep 的视觉、状态和动作历史压入 fast weights。它不是部署时拿任务 loss 微调整个 VLA，而是用当前 token 自己构造的 K→V 回归任务持续更新临时参数记忆。</p>
            <div className="ling-flow"><div><span>OBSERVE</span><b>当前真实观测</b><p>四路 RGB、proprioception 与语言进入 VLM/DiT。</p></div><i>→</i><div><span>WRITE</span><b>更新 Fast Weights</b><p>自监督 K→V loss 将当前上下文写入两层 MLP。</p></div><i>→</i><div><span>ACT</span><b>生成 Action Chunk</b><p>更新后的 MLP 读取 Q，辅助 flow-matching action head。</p></div></div>
            <div className="insight"><span>我们的核心判断</span><p>真正的新意不是单独使用 TTT，而是把参数级 memory、长序列 flow matching 和 context/target 非对称 masking 组合起来：历史可以影响动作，同时不要求历史中的每个时间步都有正确动作标签。</p></div>
          </>}

          {active === "fastweights" && <>
            <div className="evidence-label">论文明确说明 · Equations 1-3</div><h2>Fast weights 是递推状态，不是永久模型参数</h2>
            <div className="stream-grid"><article><span>SLOW WEIGHTS</span><h3>训练后保存</h3><p>包括 Q/K/V 投影、初始 W<sub>0</sub>、inner learning rate、gate 与主干参数。推理时保持冻结。</p></article><article><span>FAST WEIGHTS</span><h3>当前 rollout 临时存在</h3><p>W<sub>1</sub>、W<sub>2</sub>…由 W<sub>0</sub> 和当前 context 逐步计算产生，rollout 结束后不写回 checkpoint。</p></article></div>
            <div className="loss-master"><span>INNER UPDATE · WRITE</span><div className="equation">W<sub>t</sub> = W<sub>t−1</sub> − η∇<sub>W</sub> ‖f<sub>Wt−1</sub>(K<sub>t</sub>) − V<sub>t</sub>‖²</div></div>
            <div className="loss-master"><span>APPLY · READ</span><div className="equation">O<sub>t</sub> = f<sub>Wt</sub>(Q<sub>t</sub>)</div></div>
            <p>W<sub>t</sub> 的下标表示已经处理到第 t 个 context timestep 后的 fast-weight 状态，而不是不同网络层。16 个 DiT block 各自维护一套 W<sup>(ℓ)</sup><sub>t</sub>。</p>
            <div className="mechanism-grid"><div><b>K：写入索引</b><p>描述当前信息应该以什么键进入参数记忆。</p></div><div><b>V：自监督目标</b><p>描述 fast model 在这个键上应该保存什么内容。</p></div><div><b>Q：读取请求</b><p>更新完成后，从 fast model 查询当前动作预测需要的信息。</p></div><div><b>W<sub>0</sub>：可学习初始化</b><p>不是空白内存，而是经 outer action loss 元学习出的通用起点。</p></div></div>
          </>}

          {active === "architecture" && <>
            <div className="evidence-label">论文明确说明 · Figure 2 / Section 3.1 / Appendix A.1</div><h2>Attention 管单步，TTT 管跨时间</h2>
            <div className="stage-detail"><article><span>BACKBONE</span><h3>Eagle VLM + 16-layer DiT</h3><p>预训练 GR00T N1.7 提供视觉语言编码和连续 action flow-matching head；原始 DiT 约 538M 参数。</p></article><article><span>SEQUENCE MODULE</span><h3>16 个 TTT-MLP</h3><p>每个 DiT block 插入一个两层 GeLU fast model，每层约增加 10M 参数，完整 action head 约 690M。</p></article></div>
            <div className="equation compact">X<sub>t</sub> = [R<sub>t</sub>, q<sub>t</sub>, Ã<sub>t</sub>]　　Q<sub>t</sub>=X<sub>t</sub>θ<sub>Q</sub>　K<sub>t</sub>=X<sub>t</sub>θ<sub>K</sub>　V<sub>t</sub>=X<sub>t</sub>θ<sub>V</sub></div>
            <ol className="training-steps"><li><b>VLM encoding</b><p>当前图像和语言编码为 Φ<sub>t</sub>。视觉语言 tokens 不直接穿过 TTT。</p></li><li><b>Single-step attention</b><p>当前 timestep 的 register、proprioception 和带噪 action tokens 做 self-attention，并 cross-attend 到 Φ<sub>t</sub>。</p></li><li><b>Register compression</b><p>N=16 个 learned register tokens 汇聚视觉语言信息，降低跨时间处理视觉 tokens 的成本。</p></li><li><b>TTT recurrence</b><p>各 timestep 的 hidden tokens 沿时间送入 TTT，通过 W<sub>t−1</sub>→W<sub>t</sub>形成参数级 recurrent state。</p></li><li><b>Gated residual</b><p>O = O<sub>attn</sub> + tanh(α)⊙O<sub>TTT</sub>，α 从 0.001 附近开始，避免新模块破坏预训练能力。</p></li></ol>
            <div className="gradient-note"><span>直接输入输出</span><p>对 K 个 timestep，一个 TTT layer 可写作 TTT(X<sub>1:K</sub>, W<sub>0</sub>) → (O<sub>1:K</sub>, W<sub>K</sub>)。O 是与输入同形状的 contextualized hidden features，不是最终 action；W<sub>K</sub>则压缩了全部历史。</p></div>
          </>}

          {active === "training" && <>
            <div className="evidence-label">论文明确说明 · Equations 4-5 / Section 3.2</div><h2>Inner loss 写记忆，Outer loss 教它怎样写才有用</h2>
            <div className="loss-grid"><article><span>INNER · SELF-SUPERVISED</span><h3>L<sub>FW</sub></h3><p>V<sub>t</sub> 来自当前输入自身的可学习投影，不需要人工标签。它在训练和推理时都能更新临时 W<sub>t</sub>。</p></article><article><span>OUTER · ACTION SUPERVISION</span><h3>L<sub>fm</sub></h3><p>真实 A<sub>t</sub> 来自离线 robot trajectory，只在训练期更新 W<sub>0</sub>、Q/K/V 投影、η、gate 与可训练主干。</p></article></div>
            <div className="equation compact">A<sub>t</sub><sup>τ</sup> = τ<sub>t</sub>A<sub>t</sub> + (1−τ<sub>t</sub>)ε<sub>t</sub>　　target velocity = A<sub>t</sub> − ε<sub>t</sub></div>
            <div className="equation compact">L<sub>fm</sub> = 1/T · Σ<sub>t</sub> ‖v<sub>θ</sub>(Φ<sub>t</sub>, A<sub>t</sub><sup>τt</sup>, q<sub>t</sub>; W<sub>t−1</sub>) − (A<sub>t</sub>−ε<sub>t</sub>)‖²</div>
            <p>Action label 的构造没有改变：每个当前观测 (o<sub>t</sub>,q<sub>t</sub>) 仍然对应从 t 开始的未来 action chunk A<sub>t</sub>=[a<sub>t</sub>,…,a<sub>t+H−1</sub>]。RoboTTT 改变的是 condition：预测 A<sub>t</sub> 时额外带入压缩过去历史的 W<sub>t−1</sub>。</p>
            <div className="gradient-note"><span>为什么最终更新 W₀ 而不是 W₂？</span><p>W<sub>1</sub>、W<sub>2</sub>是由 W<sub>0</sub>经可微 inner updates 生成的 non-leaf 中间变量。Outer loss 的梯度先经过 W<sub>2</sub>、W<sub>1</sub>，最终落到 optimizer 管理的 W<sub>0</sub>及投影参数。由于 W<sub>1</sub>=W<sub>0</sub>−η∇L<sub>FW</sub>，这里包含 gradient-through-gradient 的二阶项。</p></div>
            <div className="insight"><span>最重要的理解</span><p>Inner loop 中数值上只有 W<sub>t</sub>逐步变化；θ<sub>Q/K/V</sub>和 η 不会随每帧临时改写。但 outer optimizer 会训练这些 slow parameters，使未来产生的 K、V、Q 和 ΔW 最终服务于正确 action。</p></div>
          </>}

          {active === "sequence" && <>
            <div className="evidence-label">论文明确说明 · Section 3.2 / Figure 4 / Figure 12</div><h2>8K Context 靠两项训练配方撑起来</h2>
            <div className="stage-detail"><article><span>SEQUENCE ACTION FORCING</span><h3>每个 chunk 独立采样噪声</h3><p>相邻 action chunks 可以高度重叠，但各自独立采样 τ<sub>t</sub>和 ε<sub>t</sub>，避免整条 sequence 同时全部很易或全部很难。</p></article><article><span>TBPTT</span><h3>传状态，截梯度</h3><p>长序列切成 segments；W 的数值跨段继续传递，但在边界 stop-gradient，使显存取决于 segment length 而非完整 context。</p></article></div>
            <div className="equation compact">τ<sub>t</sub> = 0.999(1−u<sub>t</sub>)　　u<sub>t</sub> ~ Beta(1.5,1)</div>
            <div className="equation compact">W<sub>end</sub><sup>next segment</sup> = stopgrad(W<sub>end</sub><sup>previous segment</sup>)</div>
            <p>TTT 状态更新在逻辑上是串行的，因为 W<sub>t</sub>依赖 W<sub>t−1</sub>。视觉编码、Q/K/V 投影、batch 内不同轨迹和单 timestep 内 token 计算可以并行，但不能把所有 W<sub>t</sub>当作彼此独立的 Transformer tokens 一次算完。</p>
            <div className="gradient-note"><span>TBPTT 的准确边界</span><p>它只解决 activation memory，不消除前向递推。后续 segment 的 outer loss不能跨 detach 边界回到 W<sub>0</sub>；因此 8K 训练是“长状态传播 + 局部 meta-gradient”，不是完整 8K-step BPTT。</p></div>
          </>}

          {active === "context" && <>
            <div className="evidence-label">论文明确说明 · Section 3.3 / Figures 5-6</div><h2>Context 可以更新记忆，但不必提供 Action target</h2>
            <div className="stream-grid"><article><span>HUMAN VIDEO</span><h3>Video as Context</h3><p>同一 circuit configuration 的 human video 与 robot trajectory 拼成一条 sequence。Video 部分更新 W，但 flow-matching loss 被 mask；后续 robot action loss训练模型从视频记忆中读取配置与顺序。</p></article><article><span>DAGGER</span><h3>Failure as Context</h3><p>次优 robot actions 写入 W，但不作模仿目标；只有 human corrections 计算 action loss，从而学习 failure→correction 的映射。</p></article></div>
            <div className="equation compact">Human video → W<sub>video</sub> → Robot trajectory action loss</div>
            <div className="equation compact">Robot failure history → W<sub>failure</sub> → Human correction target</div>
            <div className="mechanism-grid"><div><b>Robot demonstration</b><p>更新 fast weights；有 action-chunk loss。</p></div><div><b>Human-video frame</b><p>更新 fast weights；action loss mask。</p></div><div><b>DAgger failure</b><p>更新 fast weights；action loss mask。</p></div><div><b>Human correction</b><p>更新 fast weights；提供 action supervision。</p></div></div>
            <div className="insight"><span>可迁移原则</span><p>错误动作或无动作标签视频不适合直接当 BC target，但仍可作为“为什么后续动作应该这样做”的 context。这里最有价值的是 supervision 与 memory update 的解耦。</p></div>
          </>}

          {active === "inference" && <>
            <div className="evidence-label">论文明确说明 · Figure 2 / Appendix A.3；未公开项单独标注</div><h2>每次只输入当前观测与上一时刻的 W</h2>
            <ol className="training-steps"><li><b>初始化 rollout</b><p>16 个 TTT layers 分别从训练得到的 W<sub>0</sub><sup>(ℓ)</sup>开始；独立 episode 默认不继承上一条 rollout 的临时 W。</p></li><li><b>读取真实观测</b><p>四路 480p RGB、proprioception 和语言经过 Eagle VLM 与当前 DiT block。</p></li><li><b>在线写入</b><p>当前 token 产生 K、V，自监督 inner loss把 W<sub>t−1</sub>更新成 W<sub>t</sub>；推理时没有真实 action label或 outer loss。</p></li><li><b>生成动作</b><p>更新后的 fast model用 Q 读出历史特征，DiT 从高斯噪声进行 k-step flow sampling，得到 H-step action chunk。</p></li><li><b>闭环推进</b><p>控制器执行 chunk 的一部分或全部，环境返回新观测；模型保留 W<sub>t</sub>，无需重新输入过去 8K 帧。</p></li></ol>
            <div className="equation compact">(o<sub>t</sub>, q<sub>t</sub>, l, W<sub>t−1</sub>) → (Â<sub>t</sub>, W<sub>t</sub>)</div>
            <div className="mechanism-grid"><div><b>普通 rollout</b><p>每个新控制/观测 timestep 推进一次逻辑 fast-weight state；8K steps / 30 Hz ≈ 4.4 分钟。</p></div><div><b>Human-video ICL</b><p>先逐帧形成 W<sub>video</sub>；场景 reset 时不重置 W，然后继续让机器人执行历史写入同一状态。</p></div></div>
            <div className="gradient-note"><span>论文尚未写清</span><p>Action horizon H、每次实际执行步数、temporal ensembling、flow solver 的 k，以及一次真实控制步内 fast weights 如何跨多次 DiT solver evaluation commit，正文与附录均不足以精确复现。不能据此断言每个 chunk 完整执行或相邻观测跨越 H 个动作。</p></div>
          </>}

          {active === "results" && <>
            <div className="evidence-label">论文明确说明 · Tables 1-3 / Figures 7-12</div><h2>最强证据是 Context Scaling，而非单项 ICL 展示</h2>
            <div className="metric-grid"><article><span>MAIN SCORE</span><b>79%</b><p>GR00T single-step 42% · GDN 56%</p></article><article><span>8K SCALING</span><b>71.5%</b><p>同模型 1K 为 43.9%，相对提升约 63%</p></article><article><span>VIDEO ICL</span><b>6 / 10</b><p>GDN 0 / 10；completion 65% vs 33%</p></article></div>
            <div className="stage-detail"><article><span>FULL SUCCESS</span><h3>9/20 · 13/20 · 2/10</h3><p>分别对应 Pup Go Car、Circuit、Gear Bot；RoboTTT 是唯一完整完成五分钟 Gear Bot 的方法，但成功率仍只有 20%。</p></article><article><span>DAGGER DISTILLATION</span><h3>+36%</h3><p>相同 100 条 DAgger trajectories 下，RoboTTT 的 context-target 非对称训练优于标准 DAgger；错误动作直接作为 BC target 没有收益。</p></article></div>
            <div className="mechanism-grid"><div><b>MLP vs Linear</b><p>Linear fast model 比两层 MLP 低约 27%，非线性参数记忆很重要。</p></div><div><b>加入 Action tokens</b><p>相对提升约 23%，说明过去执行过什么有助于理解环境变化。</p></div><div><b>加入 Registers</b><p>进一步提升约 18%；单独给 GR00T 加相同 registers 无收益。</p></div><div><b>外部扰动</b><p>Roof 恢复 15/20；Tire 恢复 18/20，但 GDN 在 Tire 上同为 18/20。</p></div></div>
          </>}

          {active === "audit" && <>
            <div className="evidence-label">证据边界审计</div><h2>创新、限制与复现判断</h2>
            <div className="audit-stack"><div><span>01</span><h3>不是“整个 VLA 在线微调”</h3><p>推理时只有 TTT-MLP fast weights 用自监督 K→V loss更新；slow weights 与主任务参数冻结。</p></div><div><span>02</span><h3>Constant latency 不等于零代价</h3><p>它表示单步计算不随累计 context length 增长；16 个约 10M 参数的 TTT-MLP及在线梯度更新仍增加绝对开销。</p></div><div><span>03</span><h3>Human-video ICL 范围有限</h3><p>只在 Circuit task family 中测试 unseen configurations，共 10 次 trials；不能外推为开放世界任意新技能学习。</p></div><div><span>04</span><h3>Context scaling 证据最扎实</h3><p>同一 RoboTTT 从 1K 到 8K 持续提升，而 matched GDN 没有同趋势，支持 gradient-based fast weights是更强的长流压缩器。</p></div><div><span>05</span><h3>训练成本高</h3><p>预训练 16×GB200、30K steps；task post-training 8 GPUs、20K steps。完整复现还需要 YAM 双臂、长时序数据、human-video pairs 与 DAgger corrections。</p></div><div><span>06</span><h3>关键部署细节缺失</h3><p>H、k、execution horizon、TBPTT segment length和 solver 内 fast-state commit均未完整报告；目前也未发现作者官方训练代码。</p></div><div><span>07</span><h3>与 HOST 的差异</h3><p>HOST 显式预测 task progress 和 future robot observation；RoboTTT 不显式建模未来世界，而是把 human video 隐式压进 parameter-space memory。</p></div></div>
            <div className="source-strip"><b>原始资料</b><p><a href="https://arxiv.org/abs/2607.15275" target="_blank" rel="noreferrer">arXiv 论文 ↗</a>　·　<a href="https://research.nvidia.com/labs/gear/robottt/" target="_blank" rel="noreferrer">项目主页 ↗</a><br />Yunfan Jiang et al. · NVIDIA / Stanford / UT Austin</p></div>
          </>}
        </article>
        <aside className="paper-aside"><span>TAKEAWAYS</span><ol><li>W<sub>t</sub>是临时参数记忆</li><li>Inner K→V 写，Q 负责读</li><li>Outer action loss 学更新规则</li><li>Action label 仍是未来 chunk</li><li>Context 可无 action target</li><li>8K = 长状态 + 截断梯度</li></ol></aside>
      </section>
      <section className="next-note shell"><span>NEXT NOTE</span><h2>继续 In-Context Robot Learning。</h2><Link href="/notes">查看全部笔记 ↗</Link></section>
      <footer className="footer shell"><span>不凡天 · NOTE 005</span><Link href="/notes">返回笔记库 ↑</Link></footer>
    </main>
  );
}
