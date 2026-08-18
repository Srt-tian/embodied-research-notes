"use client";

import Link from "next/link";
import { useState } from "react";
import { SiteHeader } from "../../site-header";

const chapters = [
  { id: "overview", label: "01 核心结论" },
  { id: "gap", label: "02 动作语义 Gap" },
  { id: "architecture", label: "03 网络结构" },
  { id: "training", label: "04 三阶段训练" },
  { id: "router", label: "05 Router" },
  { id: "inference", label: "06 在线推理" },
  { id: "results", label: "07 实验结果" },
  { id: "audit", label: "08 评价与边界" },
];

export default function BridgeNote() {
  const [active, setActive] = useState("overview");

  return (
    <main>
      <SiteHeader />
      <section className="paper-head shell">
        <div className="paper-breadcrumb"><Link href="/notes">NOTES</Link><span>/</span><span>BRIDGE</span></div>
        <div className="paper-title masquerade-title"><span>NOTE 004 · UMI / CONTACT-RICH</span><h1>BRIDGE</h1><p>State-Gated Experts for<br />Observed and Desired Actions</p></div>
        <div className="paper-meta"><div><span>VERSION</span><b>ARXIV V1 · JUN 2026</b></div><div><span>ROBOT</span><b>FRANKA FR3</b></div><div><span>STATUS</span><b>ANALYZED</b></div></div>
      </section>

      <section className="paper-body shell">
        <aside className="paper-toc"><span>CONTENTS</span>{chapters.map(chapter => <button className={active === chapter.id ? "active" : ""} onClick={() => setActive(chapter.id)} key={chapter.id}>{chapter.label}</button>)}</aside>
        <article className="paper-content">
          {active === "overview" && <>
            <div className="evidence-label">论文明确说明 · Figure 2-3 / Section 3</div>
            <p className="paper-lead">BRIDGE 的关键不是简单增加少量真机数据，而是承认手持 UMI 与遥操作数据拥有不同的动作语义：UMI 记录实际实现的 observed action，遥操作还能记录发给控制器的 desired action。两种标签只在自由空间近似一致，接触阶段不能直接混为一种监督。</p>
            <div className="ling-flow"><div><span>01 · BASE</span><b>手持 UMI</b><p>大量 observed-action 数据学习任务主体与自由空间运动。</p></div><i>→</i><div><span>02 · SUPPORT</span><b>局部遥操作</b><p>只在 Base 失败的接触片段收集 desired-action 数据。</p></div><i>→</i><div><span>03 · ROUTE</span><b>状态门控</b><p>推理时逐动作步硬切换 Base 与 Support expert。</p></div></div>
            <div className="insight"><span>我们的核心判断</span><p>这篇论文最有价值的地方，是把“跨数据源 supervision 是否有效”提升为首要问题。异构数据不一定只是分布不同，也可能在同一个动作字段中表达不同物理含义；此时 naive co-training 会直接制造标签冲突。</p></div>
          </>}

          {active === "gap" && <>
            <div className="evidence-label">论文明确说明 · Figure 1 / Appendix E</div><h2>Observed action 为什么不是 Desired action？</h2>
            <div className="stream-grid"><article><span>HANDHELD UMI</span><h3>Observed action</h3><p>记录手持夹爪实际到达的末端位姿 x。人的肌肉施力、柔顺性以及“还想继续压多深”的控制意图没有被记录。</p></article><article><span>TELEOPERATION</span><h3>Desired action</h3><p>记录主端映射给机器人控制器的目标位姿 x<sub>d</sub>；同时还能读取从臂受到接触约束后实际达到的 x。</p></article></div>
            <div className="loss-master"><span>CARTESIAN IMPEDANCE</span><div className="equation">F = K(x<sub>d</sub> - x) = KΔ</div></div>
            <p>自由空间中外力很小，控制器能跟上命令，因此 x<sub>d</sub>≈x，UMI 实际轨迹可以近似作为控制目标。接触后，环境阻止从臂完全到达目标，持续存在 Δ=x<sub>d</sub>-x；这个位置误差正是阻抗控制器产生接触力的来源。</p>
            <div className="mechanism-grid"><div><b>弹簧例子</b><p>若机器人刚度 K 与弹簧刚度 k<sub>s</sub> 相等，期望压入 10 mm 时实际只能达到 5 mm。要让从臂实际达到 10 mm，主端需要命令约 20 mm。</p></div><div><b>UMI 缺失的信息</b><p>人手可以靠肌肉把工具实际压到 10 mm，但 UMI 只保存“实际 10 mm”。把它直接作为机器人目标后，机器人可能压不到同样深度。</p></div></div>
            <div className="equation compact">K(y<sub>d</sub>−y)=k<sub>s</sub>y　⇒　y=K/(K+k<sub>s</sub>)·y<sub>d</sub></div>
            <div className="gradient-note"><span>与 IK 的关系</span><p>论文没有把 UMI 轨迹离线 IK 成关节标签。策略预测相对笛卡尔末端位姿，交给 Cartesian impedance controller。即使做 IK，也只能找到满足 FK(q)=x 的关节姿态，无法从实际 x 反推出隐藏的 x<sub>d</sub> 或接触力。</p></div>
          </>}

          {active === "architecture" && <>
            <div className="evidence-label">论文明确说明 · Figure 3 / Section 3</div><h2>共享视觉编码器，两套完整 Expert</h2>
            <div className="stage-detail"><article><span>SHARED</span><h3>DINOv2 ViT-S/14</h3><p>输入 518×518 图像，保留 1369 个空间 patch tokens；维度为 384。Stage 2 中共享视觉编码器被冻结。</p></article><article><span>EXPERT-SPECIFIC</span><h3>Adapter + Diffusion Head</h3><p>Base 与 Support 各自拥有 latent adapter 和 temporal diffusion U-Net，而不是只共享一个 latent 后接两个 Linear 输出层。</p></article></div>
            <div className="loss-grid"><article><span>BASE EXPERT</span><h3>φ<sub>b</sub> + π<sub>b</sub></h3><p>φ<sub>b</sub> 使用 Perceiver-style queries 聚合 patch tokens，并把末端状态投影后通过 cross-attention 融合；π<sub>b</sub> 预测 observed-action chunk。</p></article><article><span>SUPPORT EXPERT</span><h3>φ<sub>s</sub> + π<sub>s</sub></h3><p>φ<sub>s</sub> 生成 support-specific latent；π<sub>s</sub> 预测 desired-action chunk。Support U-Net 比 Base U-Net 更小。</p></article></div>
            <div className="equation compact">Base expert = φ<sub>b</sub> + π<sub>b</sub>　　Support expert = φ<sub>s</sub> + π<sub>s</sub></div>
            <ol className="training-steps"><li><b>视觉输入</b><p>DINOv2 输出完整空间 tokens，不只使用 CLS token；Perceiver IO 将 1369 个 token 聚合为较少任务相关 latent。</p></li><li><b>状态输入</b><p>状态包含 SE(3) 末端位姿、相对姿态与夹爪宽度，经线性层投影至 latent dimension。</p></li><li><b>模态融合</b><p>状态特征与视觉 tokens 通过 cross-attention 形成 expert-specific conditioning latent。</p></li><li><b>动作生成</b><p>Temporal U-Net 通过 FiLM 注入 384×4 conditioning，生成 H=24 的动作块。</p></li></ol>
          </>}

          {active === "training" && <>
            <div className="evidence-label">论文明确说明 · Section 3 / Table 5</div><h2>三阶段顺序训练</h2>
            <ol className="training-steps"><li><b>Stage 1 · Base policy</b><p>使用完整手持数据集训练 DINOv2、Base latent adapter φ<sub>b</sub> 与 Base Diffusion head π<sub>b</sub>；监督目标是未来 observed-action chunk。</p></li><li><b>识别失败区域</b><p>将 Base policy 部署到机器人，经验性识别插入、受力、精确对准等失败阶段。</p></li><li><b>采集 Support 数据</b><p>DM-UMI 切换到 teleoperation mode，只示范失败附近的局部片段，记录机器人控制器的 desired trajectory。</p></li><li><b>Stage 2 · Support expert</b><p>冻结 Base expert 与共享 DINOv2，只用 Support 数据独立训练 φ<sub>s</sub> 和 π<sub>s</sub>。</p></li><li><b>Stage 3 · Router</b><p>冻结策略模块，从 Base/Support 样本提取 Base latent，构造 KNN 伪标签，再用 BCE 蒸馏到轻量 MLP gate。</p></li></ol>
            <div className="mechanism-grid"><div><b>Base U-Net</b><p>Down dims [128, 256, 512]；学习任务 scaffold 与更广的自由空间状态分布。</p></div><div><b>Support U-Net</b><p>Down dims [96, 192, 384]；只覆盖空间上稀疏的接触瓶颈。</p></div></div>
            <div className="gradient-note"><span>Diffusion 配置</span><p>Action horizon 24；训练 50 个 diffusion timesteps，推理使用 16 步；Squared Cosine Capped v2 β schedule。论文只表述为标准 Diffusion Policy loss，没有在本文重新展开 target parameterization。</p></div>
          </>}

          {active === "router" && <>
            <div className="evidence-label">论文明确说明 · Equations 1-2 / Appendix G</div><h2>Router 没有逐帧人工标签</h2>
            <p className="paper-lead">作者先用非参数 KNN 判断一个 Base latent 更靠近 Base manifold 还是 Support manifold，再把这个规则蒸馏成部署时使用的 MLP。</p>
            <ol className="training-steps"><li><b>建立 latent banks</b><p>Base 和 Support 两类样本都经过冻结的 Base conditioning path，分别得到 B<sub>b</sub> 与 B<sub>s</sub>。</p></li><li><b>删除歧义样本</b><p>移除位于 Support latent ε-neighborhood 内的 Base latent，避免边界附近出现互相冲突的伪标签。</p></li><li><b>KNN teacher</b><p>在两个 bank 中分别找 k=16 个邻居，计算平均 cosine similarity σ<sub>+</sub>(z) 与 σ<sub>−</sub>(z)。</p></li><li><b>生成伪标签</b><p>当 σ<sub>+</sub>(z)-σ<sub>−</sub>(z)&gt;η<sub>ρ</sub> 时标为 Support，否则标为 Base。</p></li><li><b>MLP distillation</b><p>用 binary cross-entropy 训练 residual MLP Router G<sub>ψ</sub>，部署时不再查询 KNN banks。</p></li></ol>
            <div className="equation compact">ρ(z)=𝟙[σ<sub>+</sub>(z)−σ<sub>−</sub>(z)&gt;η<sub>ρ</sub>]　　k=16</div>
            <div className="insight"><span>重要区别</span><p>KNN 在这里不是用来配对 UMI 与遥操作轨迹，也没有 DTW 或 OT loss；它只负责划定“什么状态应该切到 Support expert”的局部状态区域。</p></div>
          </>}

          {active === "inference" && <>
            <div className="evidence-label">论文明确说明 · Equations 3-4 / Appendix A</div><h2>两个 Expert 并行出动作，Router 硬选择</h2>
            <div className="ling-flow"><div><span>OBSERVE</span><b>图像 + State</b><p>共享 DINOv2 编码当前真实观测。</p></div><i>→</i><div><span>PREDICT</span><b>双 Action Chunk</b><p>Base 与 Support 分别执行 16-step diffusion sampling。</p></div><i>→</i><div><span>SELECT</span><b>Hard Switch</b><p>Router 对 action chunk 中每一步产生选择 mask。</p></div></div>
            <div className="equation compact">m<sub>t+j</sub>=𝟙[g<sub>t+j</sub>&gt;η]　　â<sub>t+j</sub>=(1−m<sub>t+j</sub>)â<sup>b</sup><sub>t+j</sub>+m<sub>t+j</sub>â<sup>s</sup><sub>t+j</sub></div>
            <p>由于 m∈{`{0,1}`}, 上式不是 action blending，而是逐步选择。作者认为 observed 与 desired action 可能对应两个不同的行为模式，直接平均可能得到两边都不成立的中间命令。</p>
            <div className="mechanism-grid"><div><b>防止误切换</b><p>Support probability 必须连续 0.25 s 高于 η=0.5 才激活 Support expert。</p></div><div><b>缓解控制跳变</b><p>选中的动作经 trajectory generator 在 100 Hz 插值，再流入 1 kHz Cartesian impedance controller，并进行 torque smoothing。</p></div></div>
          </>}

          {active === "results" && <>
            <div className="evidence-label">论文明确说明 · Tables 1-4</div><h2>三项接触任务均明显提升</h2>
            <div className="metric-grid"><article><span>NIST PULLEY</span><b>76%</b><p>Handheld 44% · Full teleop 84%</p></article><article><span>PIPE INSERTION</span><b>50%</b><p>Handheld 13.3% · Full teleop 63.3%</p></article><article><span>BATTERY</span><b>33.3%</b><p>Handheld 10% · Full teleop 40%</p></article></div>
            <div className="stage-detail"><article><span>NAIVE MIX</span><h3>0% / 6.7% / 0%</h3><p>直接将 observed 与 desired labels 混进同一个 policy，在三项任务上都比 BRIDGE 差，甚至比 Handheld-only 更差。</p></article><article><span>SPARSE SUPPORT</span><h3>&lt;15% path length</h3><p>Support demonstrations 的中位路径长度占比仅为 Pulley 7.4%、Pipe 3.4%、Battery 14.9%，却恢复了大部分 full teleoperation 性能。</p></article></div>
            <div className="gradient-note"><span>Router 结果</span><p>在 Pipe Insertion 的 held-out 手工标注数据上，Router recall 为 99%，precision 为 69%。作者刻意偏向高 recall，宁可稍早进入 Support，也尽量避免 Base 错误处理插入阶段。</p></div>
          </>}

          {active === "audit" && <>
            <div className="evidence-label">证据边界审计</div><h2>价值、限制与可迁移原则</h2>
            <div className="audit-stack"><div><span>01</span><h3>最强贡献是动作语义拆分</h3><p>Observed/desired mismatch 比一般的视觉 domain gap 更根本；错误动作语义无法只靠增加数据量解决。</p></div><div><span>02</span><h3>不是通用机器人 MoE</h3><p>两个 expert 按 supervision validity 与任务阶段划分，而不是按机器人 identity 或任务类别划分。</p></div><div><span>03</span><h3>IK 无法补回控制意图</h3><p>IK 只能重建关节构型，不能恢复 UMI 未记录的 desired setpoint、刚度选择或接触力。</p></div><div><span>04</span><h3>支持区域依赖人工发现</h3><p>当前流程需要先部署 Base，再由研究者经验性识别失败阶段并收集局部 teleoperation。</p></div><div><span>05</span><h3>Hard switch 有边界风险</h3><p>论文承认硬切换可能产生控制不连续；debounce 与下游 smoothing 是工程缓解，不是严格稳定性保证。</p></div><div><span>06</span><h3>扩展性尚未证明</h3><p>只验证一台 FR3 和三个接触任务。若长任务存在很多接触机制，为每个区域增加 refinement head 可能迅速膨胀。</p></div><div><span>07</span><h3>控制器是隐含前提</h3><p>论文结论建立在有限刚度 Cartesian impedance tracking 上；不同 action interface、力控策略或高精度刚性跟踪下，mismatch 大小可能不同。</p></div></div>
            <div className="source-strip"><b>原始资料</b><p><a href="https://arxiv.org/abs/2606.26603" target="_blank" rel="noreferrer">arXiv 论文 ↗</a>　·　<a href="https://nperi-rai.github.io/bridge/" target="_blank" rel="noreferrer">项目主页 ↗</a><br />Vidullan Surendran, Neehar Peri, David Watkins · RAI Institute / Carnegie Mellon University</p></div>
          </>}
        </article>
        <aside className="paper-aside"><span>TAKEAWAYS</span><ol><li>UMI 记录 observed action</li><li>Teleop 提供 desired action</li><li>Support expert = φs + πs</li><li>KNN 只负责 Router 伪标签</li><li>IK 不能恢复接触意图</li></ol></aside>
      </section>
      <section className="next-note shell"><span>NEXT NOTE</span><h2>继续 UMI-to-Robot。</h2><Link href="/notes">查看全部笔记 ↗</Link></section>
      <footer className="footer shell"><span>不凡天 · NOTE 004</span><Link href="/notes">返回笔记库 ↑</Link></footer>
    </main>
  );
}
