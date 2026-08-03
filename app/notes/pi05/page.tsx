"use client";

import Link from "next/link";
import { useState } from "react";
import { SiteHeader } from "../../site-header";
import { assetPath } from "../../site-paths";

const chapters = [
  { id:"overview", label:"01 核心结论" },
  { id:"architecture", label:"02 完整架构" },
  { id:"fast", label:"03 FAST 动作编码" },
  { id:"training", label:"04 两阶段与 Loss" },
  { id:"attention", label:"05 Attention Mask" },
  { id:"audit", label:"06 关键审计" },
];

export default function Pi05Note() {
  const [active,setActive]=useState("overview");
  return (
    <main>
      <SiteHeader />
      <section className="paper-head shell">
        <div className="paper-breadcrumb"><Link href="/notes">NOTES</Link><span>/</span><span>π0.5</span></div>
        <div className="paper-title"><span>NOTE 001 · VLA</span><h1>π0.5</h1><p>A Vision-Language-Action Model with<br />Open-World Generalization</p></div>
        <div className="paper-meta"><div><span>PUBLISHED</span><b>APR 2025</b></div><div><span>READING</span><b>12 MIN</b></div><div><span>STATUS</span><b>ANALYZED</b></div></div>
      </section>
      <section className="paper-body shell">
        <aside className="paper-toc"><span>CONTENTS</span>{chapters.map(c=><button className={active===c.id?"active":""} onClick={()=>setActive(c.id)} key={c.id}>{c.label}</button>)}<div className="toc-files"><a href={assetPath("/downloads/pi05-visual-paper-analysis.pdf")} download>PDF ↓</a><a href={assetPath("/downloads/pi05-visual-paper-analysis.docx")} download>WORD ↓</a></div></aside>
        <article className="paper-content">
          {active==="overview" && <><div className="evidence-label">论文明确说明 · Section IV-A</div><p className="paper-lead">π0.5 的核心不是单个新网络层，而是一套训练配方：FAST 离散预训练吸收异构知识，Flow Matching Action Expert 负责连续动作实时生成。</p><div className="equation">π(a, ℓ̂ | o, ℓ) = π(a | o, ℓ̂) · π(ℓ̂ | o, ℓ)</div><p>同一 VLA 先生成高层语义子任务，再以该子任务为条件生成低层连续动作块。高层输出既是可解释的中间计划，也是低层动作的条件瓶颈。</p><div className="insight"><span>分析结论</span><p>创新重心是“异构数据共训练 + 高低层语义分解 + 混合动作表示”，而不只是 Action Expert。</p></div></>}
          {active==="architecture" && <><div className="evidence-label">论文明确说明 · Figure 3 / Appendix E</div><h2>完整模型架构</h2><figure><img src={assetPath("/notes/pi05-model-overview.png")} alt="π0.5 模型总览" /><figcaption>Figure 3 · 由原论文 PDF 以 300 DPI 重新渲染</figcaption></figure><p>图像经 SigLIP 编码，文本与离散化本体状态进入 PaliGemma；FAST 动作由 VLM 自回归预测，连续噪声动作则由约 300M 参数的 Action Expert 处理。</p><div className="insight"><span>根据架构推断</span><p>Action Expert 更接近按模态固定分配 Transformer 参数组，而不是传统稀疏 MoE 的动态路由；论文未展示 router。</p></div></>}
          {active==="fast" && <><div className="evidence-label">补充机制 · π0.5 Section IV-B + FAST 原论文</div><h2>FAST 如何把动作变成 token</h2><p className="paper-lead">FAST 不是逐时间步独立分桶，而是先压缩整个动作块的时间结构，再把压缩结果交给 VLM 做 next-token prediction。</p><div className="fast-flow"><div><span>01</span><b>Action chunk</b><p>A∈ℝ<sup>H×D</sup>，一次处理 H 步、每步 D 维动作。</p></div><div><span>02</span><b>Normalize</b><p>按动作维归一化，避免不同关节或控制量尺度主导量化。</p></div><div><span>03</span><b>DCT</b><p>沿时间轴逐动作维变换到频域；平滑轨迹主要集中于低频系数。</p></div><div><span>04</span><b>Quantize</b><p>将连续 DCT 系数舍入为离散整数；这里产生有损近似。</p></div><div><span>05</span><b>BPE</b><p>展平整数序列，并把高频共现片段合并成更短的 token 序列；BPE 编解码本身可逆。</p></div></div><div className="equation compact">A → Normalize → DCT → Quantize → BPE → z<sub>1:L</sub></div><h3 className="section-title">训练与解码</h3><div className="mechanism-grid"><div><b>训练</b><p>VLM 自回归预测 z<sub>i</sub>，使用 teacher forcing 的 token 交叉熵。FAST 让机器人动作与文本共享“离散序列预测”训练范式。</p></div><div><b>推理 / 还原</b><p>预测 token → BPE 解码 → 反量化 → IDCT → 反归一化，恢复连续动作块。</p></div></div><div className="insight"><span>在 π0.5 中的准确角色</span><p>FAST 主要承担预训练中的离散动作表示，以高效吸收异构机器人数据；最终实时低层控制改由 Flow Matching Action Expert 生成。π0.5 正文没有重新给出 FAST 的全部量化细节，因此本节机制来自其引用的 FAST 原论文。</p></div></>}
          {active==="training" && <><div className="evidence-label">论文明确说明 · Equation (1) / Section IV-C / IV-D</div><h2>两阶段训练与 Loss</h2><div className="loss-master"><span>联合目标</span><div className="equation">L = H(x<sub>1:M</sub>, y<sup>ℓ</sup><sub>1:M</sub>) + α ‖(ω − a<sub>t:t+H</sub>) − f<sup>a</sup><sub>θ</sub>(a<sup>τ,ω</sup><sub>t:t+H</sub>, o<sub>t</sub>, ℓ)‖²</div></div><div className="loss-grid"><article><span>TERM 01 · TOKEN CE</span><h3>离散序列损失</h3><p>目标：真实文本 token；在机器人离散动作任务中也包括 FAST action token。模型用 teacher forcing 预测下一个 token。</p><p><b>不直接监督：</b>图像 patch、作为条件输入的本体状态，以及没有对应输出目标的位置。论文明确指出并非所有 N 个输出都关联 loss。</p></article><article><span>TERM 02 · FLOW MSE</span><h3>连续动作向量场损失</h3><p>先采样真实动作块 a、噪声 ω 和流时间 τ，构造 a<sup>τ,ω</sup>=τa+(1−τ)ω；Action Expert 接收带噪动作块及条件，回归论文写出的目标向量 ω−a。</p><p><b>监督范围：</b>整个连续动作 chunk；以平方 L2 误差训练，不是动作重建 CE。</p></article></div><div className="stage-detail"><article><span>STAGE 01 · 280K STEPS</span><h3>Pre-training：只开离散目标</h3><p><b>数据：</b>MM + ME + CE + HL + WD。</p><p><b>权重：</b>α=0，因此联合式退化为 token CE。机器人动作先经 FAST 编码，动作预测与文本/高层子任务预测都变成 next-token prediction。</p><p><b>作用：</b>让 VLM 在统一离散接口上吸收跨本体机器人数据、语义任务和 web 知识；此时没有 Flow Loss 训练 Action Expert。</p></article><article><span>STAGE 02 · 80K STEPS</span><h3>Post-training：CE 与 Flow 联合</h3><p><b>数据：</b>成功且长度达标的 MM、ME，匹配多环境数据的 HL 子集，WD，以及新增 VI；CE 被移除。</p><p><b>权重：</b>α=10。Action Expert 在该阶段随机初始化；next-token CE 保留文本/高层预测能力，Flow MSE 学习连续动作块。</p><p><b>推理对应：</b>先自回归生成高层文本子任务，再以其为条件进行 10 次 Flow 积分得到 50 步动作 horizon。</p></article></div><div className="gradient-note"><span>梯度链路审计</span><p>CE 明确训练离散 token 预测路径；Flow Loss 明确训练 Action Expert。由于 Expert 通过 attention 读取 VLM prefix，且论文没有报告 stop-gradient、冻结 VLM 或独立优化器，Flow 梯度原则上也可能回传到提供条件表示的共享/VLM 路径；但具体参数冻结、梯度裁剪和 optimizer 分组，论文未说明，不能写成已证实实现。</p></div><p>第一阶段 97.6% 的训练样本不是目标移动机器人家务数据。这个比例支撑作者的核心判断：开放世界能力来自异构预训练与后训练配方，而不是仅扩大目标平台数据。</p></>}
          {active==="attention" && <><div className="evidence-label">论文明确说明 · Figure 18 / Appendix E</div><h2>Attention Mask</h2><figure className="mask-figure"><img src={assetPath("/notes/pi05-attention-mask.png")} alt="π0.5 Attention Mask" /><figcaption>Figure 18 · 由原论文 PDF 以 300 DPI 重新渲染</figcaption></figure><div className="attention-lines"><p><b>Prefix</b><span>图像、Prompt、State 采用 full prefix mask，彼此可见。</span></p><p><b>FAST</b><span>读取 Prefix，并仅自回归读取此前 FAST token。</span></p><p><b>Expert</b><span>读取 Prefix 与 Action Expert token，但不读取 FAST token。</span></p></div></>}
          {active==="audit" && <><div className="evidence-label">证据边界审计</div><h2>关键审计</h2><div className="audit-stack"><div><span>01</span><h3>可能存在矛盾</h3><p>按论文给定路径 aᵗ=τa+(1−τ)ω 对 τ 求导得到 a−ω，但正文与 Loss 的 target 写作 ω−a。仅凭论文无法消解，需要结合 ODE 积分方向或官方代码约定核实。</p></div><div><span>02</span><h3>论文未说明</h3><p>Action horizon 为 50，但论文未充分说明在线执行时每轮消费多少步、何时重规划以及高层子任务刷新频率。</p></div><div><span>03</span><h3>实现边界</h3><p>论文系统包含 HL、WD、VI 与显式层次推理；公开 OpenPI 的常规动作微调不能默认等价于完整论文系统。</p></div></div><div className="source-strip"><span>证据位置</span><p>架构与公式：第 5 页 · 数据配方：第 6–7 页 · Attention Mask：第 19 页</p></div></>}
        </article>
        <aside className="paper-aside"><span>TAKEAWAYS</span><ol><li>核心是训练配方</li><li>FAST：DCT + 量化 + BPE</li><li>预训练仅 CE</li><li>后训练 CE + 10×Flow MSE</li><li>公开代码并非完整复现</li></ol></aside>
      </section>
      <section className="next-note shell"><span>NEXT NOTE</span><h2>继续生长中。</h2><Link href="/notes">查看全部笔记 ↗</Link></section>
      <footer className="footer shell"><span>不凡天 · NOTE 001</span><Link href="/notes">返回笔记库 ↑</Link></footer>
    </main>
  );
}
