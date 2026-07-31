"use client";

import Link from "next/link";
import { useState } from "react";
import { SiteHeader } from "../../site-header";

const chapters = [
  { id: "overview", label: "01 核心结论" },
  { id: "architecture", label: "02 架构与对齐" },
  { id: "training", label: "03 完整训练" },
  { id: "loss", label: "04 三类 Loss" },
  { id: "inference", label: "05 闭环推理" },
  { id: "results", label: "06 实验与复现" },
  { id: "audit", label: "07 关键审计" },
];

export default function LingBotVANote() {
  const [active, setActive] = useState("overview");
  return (
    <main>
      <SiteHeader />
      <section className="paper-head shell">
        <div className="paper-breadcrumb"><Link href="/notes">NOTES</Link><span>/</span><span>LINGBOT-VA</span></div>
        <div className="paper-title lingbot-title"><span>NOTE 002 · WORLD MODEL</span><h1>LingBot<br />VA</h1><p>Causal World Modeling<br />for Robot Control</p></div>
        <div className="paper-meta"><div><span>VERSION</span><b>ARXIV V2 · MAR 2026</b></div><div><span>MODEL</span><b>5.3B</b></div><div><span>STATUS</span><b>ANALYZED</b></div></div>
      </section>

      <section className="paper-body shell">
        <aside className="paper-toc"><span>CONTENTS</span>{chapters.map(c => <button className={active === c.id ? "active" : ""} onClick={() => setActive(c.id)} key={c.id}>{c.label}</button>)}</aside>
        <article className="paper-content">
          {active === "overview" && <>
            <div className="evidence-label">论文明确说明 · Figure 2 / Equations 6–9</div>
            <p className="paper-lead">LingBot-VA 先在 latent space 中“想象”未来视觉，再用 inverse dynamics 根据视觉变化反推动作；视频与动作由双流 MoT 联合建模。</p>
            <div className="ling-flow"><div><span>01</span><b>真实观测</b><p>相机图像经 causal VAE 编码为 video latent。</p></div><i>→</i><div><span>02</span><b>未来想象</b><p>Video Stream 用 Flow Matching 生成未来 latent。</p></div><i>→</i><div><span>03</span><b>动作反推</b><p>Action Stream 根据当前/未来视觉生成 action chunk。</p></div></div>
            <div className="equation compact">o≤t, a&lt;t → ẑt+1:t+K → at:t+K−1</div>
            <div className="insight"><span>准确定位</span><p>概念上是 Visual Dynamics + Inverse Dynamics；实现上不是两个完全独立的阶段，而是在每层通过 MoT attention 交换信息并端到端联合训练。</p></div>
          </>}

          {active === "architecture" && <>
            <div className="evidence-label">论文明确说明 · Section 3.3 / Section 4.2</div><h2>双流 MoT 架构</h2>
            <div className="stream-grid"><article><span>VIDEO STREAM</span><h3>Wan2.2-5B</h3><p>30层，hidden 3072；预测未来视频 latent velocity。</p></article><article><span>ACTION STREAM</span><h3>≈350M</h3><p>30层，hidden 768；输出连续动作 velocity。</p></article></div>
            <p>两个流使用独立 QKV。Action token 先投影到 video dimension，参加联合 attention，再投影回 action dimension并保留 residual。指令由冻结 T5 编码，经 cross-attention 注入。</p>
            <h3 className="section-title">时间对齐</h3>
            <div className="alignment-strip"><b>z<sub>t</sub></b><span>a<sub>t,1</sub></span><span>a<sub>t,2</sub></span><span>a<sub>t,3</sub></span><span>a<sub>t,4</sub></span><b>z<sub>t+1</sub></b></div>
            <p>视频按时间下采样 τ=4，所以一个稀疏视频位置对应4个高频动作。预测 K 个视频位置，理论上对应 4K 个动作；RoboTwin 中视频为12.5Hz、动作仍为50Hz。</p>
            <div className="insight"><span>必须区分</span><p>video timestep、原始 action timestep 和 action chunk index 不是同一个时间变量。</p></div>
          </>}

          {active === "training" && <>
            <div className="evidence-label">论文明确说明 · Figure 3 / Equations 10–12</div><h2>一次完整训练 iteration</h2>
            <ol className="training-steps"><li><b>采样数据</b><p>从约16K小时机器人数据中取 episode/window；各来源均匀采样，训练/验证按90%/10%划分。</p></li><li><b>统一表示</b><p>双臂动作统一为30维，并做 per-dimension quantile normalization；视频经 causal VAE、patchify 后每帧约192个空间 token。</p></li><li><b>随机 Chunk</b><p>实际训练随机采样 K∈[1,4]，部署默认 K=4。</p></li><li><b>Flow 加噪</b><p>分别给 future video latent 与 action target 采样高斯噪声和 flow time，构造线性插值。</p></li><li><b>历史增强</b><p>以50%概率给历史视频 latent 加入 s<sub>aug</sub>∈[0.5,1] 的噪声。</p></li><li><b>Teacher Forcing</b><p>历史 context 使用数据集真值；causal mask 阻止查看未来 chunk，并隔离 packed episode。</p></li><li><b>MoT Forward</b><p>一次前向同时输出 video velocity 与 action velocity。</p></li><li><b>反向更新</b><p>两个 Flow MSE 相加后反传；更新视频流、动作流与融合参数，T5 冻结。</p></li></ol>
            <div className="equation compact">x(s) = (1−s)ε + sx₁　　target = dx(s)/ds = x₁−ε</div>
            <div className="gradient-note"><span>训练—推理差异</span><p>训练时 Action Stream 主要看由 GT 视频编码、但可能加噪的 future latent；推理时看 Video Stream 自己生成的 future latent。Noisy History Augmentation 只能部分缓解 exposure bias。</p></div>
          </>}

          {active === "loss" && <>
            <div className="evidence-label">论文明确说明 · Equations 11–13</div><h2>三类 Flow Matching Loss</h2>
            <div className="loss-cards"><article><span>L DYN</span><h3>视觉动力学</h3><div className="mini-equation">‖vθ(z(s), s | z̃≤t, a&lt;t, c) − (z−εz)‖²</div><p><b>输入：</b>带噪未来视频、历史视频/动作、指令。<br /><b>输出：</b>video latent velocity。<br /><b>目的：</b>学习世界如何变化。</p></article><article><span>L INV</span><h3>逆动力学</h3><div className="mini-equation">‖vψ(a(s), s | z̃≤t+1, a&lt;t, c) − (a−εa)‖²</div><p><b>输入：</b>带噪动作、当前/未来视觉、历史动作、指令。<br /><b>输出：</b>action velocity。<br /><b>目的：</b>根据期望视觉变化反推动作。</p></article><article><span>L FDM · POST-TRAIN</span><h3>异步校准</h3><div className="mini-equation">zt(real), at → ẑt+1(grounded)</div><p><b>输入：</b>最近真实状态与正在执行的动作。<br /><b>输出：</b>执行后的视觉 velocity。<br /><b>目的：</b>替换异步系统中的 stale prediction。</p></article></div>
            <div className="equation compact">L = L<sub>dyn</sub> + λL<sub>inv</sub>，λ = 1　　Post-training: + λ<sub>fdm</sub>L<sub>fdm</sub></div>
            <div className="insight"><span>论文未说明</span><p>λfdm、具体 loss reduction、episode padding mask，以及跨本体无效 action dimension 是否参与 loss 均未完整给出。</p></div>
          </>}

          {active === "inference" && <>
            <div className="evidence-label">论文明确说明 · Algorithms 1–2 / Figure 4</div><h2>什么时候取真实观测？</h2>
            <p className="paper-lead">相机在每个 action chunk 执行期间持续采集观测并写入 ObsQueue；下一轮推理读取上一 chunk 的真实反馈，用它重新校准缓存。</p>
            <div className="async-table"><div><span>正在执行</span><span>最新真实观测</span><span>FDM 补偿</span><span>提前生成</span></div><div><b>A⁰</b><b>O⁰</b><b>Ô¹</b><b>A¹</b></div><div><b>A¹</b><b>O¹</b><b>Ô²</b><b>A²</b></div><div><b>A²</b><b>O²</b><b>Ô³</b><b>A³</b></div></div>
            <div className="equation compact">zᶦ(real) + Aᶦ → FDM → ẑᶦ⁺¹(grounded) → Aᶦ⁺¹</div>
            <p>因此预测视频负责提前规划，真实视频负责闭环纠错。系统不是无限递归预测，而是“预测一个 chunk → 获得真实反馈 → 重新锚定 → 再预测”。异步速度的代价是反馈通常落后当前执行动作一个 chunk。</p>
            <div className="mechanism-grid"><div><b>Video</b><p>Euler 3步，只积分到 s=0.6；动作不需要像素级清晰的未来视频。</p></div><div><b>Action</b><p>Euler 10步，完整积分到 s=1.0；当前动作执行与下一段预测并行。</p></div></div>
          </>}

          {active === "results" && <>
            <div className="evidence-label">论文明确说明 · Tables 1–3 / Figures 8–9</div><h2>结果与复现条件</h2>
            <div className="metric-grid"><article><span>ROBOTWIN EASY</span><b>92.9%</b><p>π0.5：82.7%</p></article><article><span>ROBOTWIN HARD</span><b>91.6%</b><p>π0.5：76.8%</p></article><article><span>LIBERO AVG</span><b>98.5%</b><p>Long：98.5%</p></article></div>
            <p>Horizon=3 的优势达到约8–9个百分点。异步消融中，Naive Async 总体仅74.3%，Horizon=3 更降至32.9%，说明旧预测造成的漂移非常严重。</p>
            <div className="stream-grid"><article><span>FULL PRE-TRAIN</span><h3>成本极高</h3><p>5.3B模型、约16K小时机器人数据、1.4T tokens；AdamW，peak LR 1e−4，bf16，gradient clipping 2.0。</p></article><article><span>RECOMMENDED</span><h3>Checkpoint Post-train</h3><p>使用作者 checkpoint；约50条 demonstrations，LR 1e−5，训练3K steps，再在 RoboTwin/LIBERO 验证。</p></article></div>
          </>}

          {active === "audit" && <>
            <div className="evidence-label">证据边界审计</div><h2>关键审计</h2>
            <div className="audit-stack"><div><span>01</span><h3>Chunk-level causal</h3><p>chunk 之间因果，但同一 chunk 内仍有 bidirectional attention，不能描述成严格逐 frame/action 因果。</p></div><div><span>02</span><h3>时间符号冲突</h3><p>正文说 K 个视频对应 τK 个动作，但公式与算法经常仍写 K 个动作。</p></div><div><span>03</span><h3>配置不一致</h3><p>方法举例写 K∈[1,8]，实现细节写实际使用 K∈[1,4]。</p></div><div><span>04</span><h3>符号问题</h3><p>公式(13)使用 vψ 预测视频，与前文 ψ 表示 Action Model 的约定冲突。</p></div><div><span>05</span><h3>未说明</h3><p>VAE 冻结状态、padding/action mask、λfdm 以及 FDM 参数共享关系需以代码为准。</p></div><div><span>06</span><h3>创新证据边界</h3><p>整体系统有效，但缺少有/无视频预测 loss、不同视频质量、动作流对未来视频依赖度等关键消融。</p></div></div>
            <div className="source-strip"><b>证据位置</b><p>方法与公式：第4–9页 · 训练细节：第10–11页 · 主实验与消融：第13–16页</p></div>
          </>}
        </article>
        <aside className="paper-aside"><span>TAKEAWAYS</span><ol><li>先想象未来，再反推动作</li><li>双流 MoT 联合训练</li><li>视频与动作都是 Flow Matching</li><li>一个视频位置对应4个动作</li><li>真实反馈 + FDM 抑制漂移</li></ol></aside>
      </section>
      <section className="next-note shell"><span>NEXT NOTE</span><h2>继续生长中。</h2><Link href="/notes">查看全部笔记 ↗</Link></section>
      <footer className="footer shell"><span>不凡天 · NOTE 002</span><Link href="/notes">返回笔记库 ↑</Link></footer>
    </main>
  );
}
