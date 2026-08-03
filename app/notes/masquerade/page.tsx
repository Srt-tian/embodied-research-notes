"use client";

import Link from "next/link";
import { useState } from "react";
import { SiteHeader } from "../../site-header";

const chapters = [
  { id: "overview", label: "01 核心结论" },
  { id: "editing", label: "02 视频机器人化" },
  { id: "supervision", label: "03 人类视频监督" },
  { id: "cotraining", label: "04 Co-training" },
  { id: "training", label: "05 完整训练" },
  { id: "results", label: "06 实验结果" },
  { id: "audit", label: "07 评价与边界" },
];

export default function MasqueradeNote() {
  const [active, setActive] = useState("overview");

  return (
    <main>
      <SiteHeader />
      <section className="paper-head shell">
        <div className="paper-breadcrumb"><Link href="/notes">NOTES</Link><span>/</span><span>MASQUERADE</span></div>
        <div className="paper-title masquerade-title"><span>NOTE 003 · CROSS-EMBODIMENT</span><h1>Masquerade</h1><p>Learning from In-the-wild<br />Human Videos using Data-Editing</p></div>
        <div className="paper-meta"><div><span>VERSION</span><b>ARXIV V1 · AUG 2025</b></div><div><span>ROBOT DATA</span><b>50 DEMOS / TASK</b></div><div><span>STATUS</span><b>ANALYZED</b></div></div>
      </section>

      <section className="paper-body shell">
        <aside className="paper-toc"><span>CONTENTS</span>{chapters.map(chapter => <button className={active === chapter.id ? "active" : ""} onClick={() => setActive(chapter.id)} key={chapter.id}>{chapter.label}</button>)}</aside>
        <article className="paper-content">
          {active === "overview" && <>
            <div className="evidence-label">论文明确说明 · Figure 2 / Section III-C</div>
            <p className="paper-lead">Masquerade 的真正重点不是把人类视频变成机器人 action 数据，而是让编辑视频在策略训练期间持续承担未来二维轨迹任务，以 co-training 保住并增强共享 ViT 的视觉能力。</p>
            <figure><img src="/notes/masquerade-overview.png" alt="Masquerade 数据编辑、视觉预训练与共同训练流程" /><figcaption>Figure 2 · 从原论文 PDF 高清裁切；实线为机器人策略分支，虚线为编辑视频辅助分支</figcaption></figure>
            <div className="ling-flow"><div><span>01 · EDIT</span><b>视觉本体对齐</b><p>擦除人臂，叠加虚拟双臂机器人。</p></div><i>→</i><div><span>02 · PRETRAIN</span><b>未来运动表征</b><p>ViT 从当前帧预测未来二维末端轨迹。</p></div><i>→</i><div><span>03 · COTRAIN</span><b>持续共享主干</b><p>编辑视频辅助 loss 与真机 policy loss 同时更新 ViT。</p></div></div>
            <div className="insight"><span>我们的核心判断</span><p>论文表面的记忆点是 robot overlay，实验真正揭示的关键则是：完成视觉本体对齐之后，必须让人类视频持续参与 co-training；简单的“预训练后仅用机器人数据微调”会显著丢失 OOD 泛化。</p></div>
          </>}

          {active === "editing" && <>
            <div className="evidence-label">论文明确说明 · Section III-B / Appendix C</div><h2>如何把人类视频“机器人化”</h2>
            <ol className="training-steps"><li><b>手部定位</b><p>利用 EPIC-KITCHENS 标注定位左右手，并用 HaMeR 为每只手估计 21 个三维关键点。</p></li><li><b>动作重定向</b><p>将关键点映射为末端位姿 P<sub>t</sub>=(p<sub>t</sub>, R<sub>t</sub>, g<sub>t</sub>)，包含位置、姿态和归一化夹爪开度，再做时间平滑。</p></li><li><b>擦除人体</b><p>Detectron2 与 SAM2 分割手臂，E2FGVI 对被移除区域进行视频修复。</p></li><li><b>叠加机器人</b><p>利用已知相机内外参渲染虚拟双臂机器人，使末端跟随恢复轨迹，并合成回原视频。</p></li><li><b>过滤坏帧</b><p>相邻时刻估计相机运动超过 5 cm 或 0.5 rad 时丢弃；双手都缺失的帧也会被移除。</p></li></ol>
            <div className="mechanism-grid"><div><b>编辑解决什么</b><p>缩小人手、人臂与机器人夹爪之间的视觉域差异，让视觉主干更容易把人类操作经验迁移到机器人画面。</p></div><div><b>编辑没有解决什么</b><p>单目视频仍缺少可靠绝对深度，遮挡关系和灵巧手到平行夹爪的映射也不准确，因此不能把恢复轨迹直接视作可执行 action。</p></div></div>
            <div className="insight"><span>值得注意</span><p>overlay 即使存在抓取不自然、机器人像素错误覆盖物体等瑕疵，仍明显优于不叠加机器人。这说明跨本体迁移中，粗粒度视觉一致性可能比照片级真实性更重要。</p></div>
          </>}

          {active === "supervision" && <>
            <div className="evidence-label">论文明确说明 · Equation 1 / Section III-B.2</div><h2>编辑视频到底监督什么？</h2>
            <p className="paper-lead">它不监督机器人三维 action，只监督当前画面中的未来二维末端轨迹。</p>
            <div className="equation compact">p<sub>t:t+H,2D</sub> = (p<sub>t,2D</sub>, p<sub>t+1,2D</sub>, …, p<sub>t+H,2D</sub>)</div>
            <p>HaMeR 的单目输入无法可靠恢复绝对三维位置，因此作者把平滑后的末端位置投影到图像平面，得到二维 waypoint。由于头戴相机本身会运动，未来关键点还要通过 homography 统一 warp 回当前帧 t 的视角。</p>
            <div className="stream-grid"><article><span>INPUT</span><h3>当前编辑帧 + 语言</h3><p>224×224 RGB；每段视频的描述由 DistilBERT 预先编码，并通过 FiLM 调制 ViT 特征。</p></article><article><span>TARGET</span><h3>未来 H 步二维轨迹</h3><p>左右末端在当前图像坐标系中的未来 waypoint 序列；论文没有报告 H 的具体值。</p></article></div>
            <div className="gradient-note"><span>与世界模型的关系</span><p>它确实用“未来会怎样运动”监督当前表征，具有 predictive representation / world-model flavor；但它不预测未来图像、latent 或完整状态，不能递归 rollout，因此不是完整世界模型。</p></div>
          </>}

          {active === "cotraining" && <>
            <div className="evidence-label">论文明确说明 · Section III-C / Figure 5</div><h2>真正关键的 Co-training</h2>
            <div className="loss-master"><span>JOINT OBJECTIVE</span><div className="equation">L = L<sub>2D</sub> + λL<sub>policy</sub>　　λ = 10</div></div>
            <div className="loss-grid"><article><span>EDITED HUMAN VIDEO</span><h3>未来二维轨迹 Loss</h3><div className="mini-equation">L₂D = ‖h(f(x, zₓ)) − pₜ:ₜ₊H,2D‖²</div><p>只训练视觉表征与辅助 MLP；不使用、也没有真实机器人 action target。</p></article><article><span>REAL ROBOT DEMO</span><h3>Diffusion Policy Loss</h3><div className="mini-equation">Lpolicy = policy loss on P(r)</div><p>真实笛卡尔末端动作只来自每个任务 50 条双臂机器人示范。正文以简式表示，实际动作头采用 DDPM Diffusion Policy。</p></article></div>
            <div className="stage-detail"><article><span>SHARED</span><h3>ViT-Base f</h3><p>同时接收 L<sub>2D</sub> 与 L<sub>policy</sub> 的梯度，是两类数据发生知识迁移的唯一共享主干。</p></article><article><span>SEPARATE HEADS</span><h3>MLP h / Policy g</h3><p>辅助 MLP 只接收二维轨迹梯度；Diffusion Policy 只接收真实机器人 imitation 梯度。</p></article></div>
            <div className="insight"><span>为什么不是普通“预训练 → 微调”</span><p>若策略阶段停止 L<sub>2D</sub>，少量、单场景机器人数据会让 ViT 遗忘从多样人类视频中学到的场景、物体、可供性和未来运动表征。持续 co-training 相当于任务相关的视觉正则化。</p></div>
          </>}

          {active === "training" && <>
            <div className="evidence-label">论文明确说明 · Appendix A-B</div><h2>训练配置与完整数据流</h2>
            <ol className="training-steps"><li><b>人类视频数据</b><p>EPIC-KITCHENS 约 10K clips，过滤后共 675,713 帧；ViT-Base 以 ImageNet 权重初始化。</p></li><li><b>视觉预训练</b><p>输入 224×224，batch size 160，AdamW，LR 1e-4，训练 150K steps，用未来二维轨迹 MSE 更新 ViT 与 MLP。</p></li><li><b>机器人示范</b><p>三个任务各收集 50 条双臂示范；双 Kinova Gen3、Robotiq 2F-85，固定 ZED Mini 主视角。</p></li><li><b>策略阶段</b><p>Diffusion Policy 输入 224×224，batch size 64，AdamW，LR 1e-4，cosine scheduler 与 500-step warmup，共 40K steps。</p></li><li><b>共同训练</b><p>编辑视频 batch 继续计算 L<sub>2D</sub>；机器人 batch 计算 L<sub>policy</sub>；两者按 λ=10 联合优化共享 ViT。</p></li><li><b>动作采样</b><p>策略使用 DDPM scheduler，训练与推理均为 100 个去噪步；论文未报告实际控制延迟。</p></li></ol>
            <div className="gradient-note"><span>关键实现边界</span><p>正文把 L<sub>policy</sub> 简写为 g(f(y)) 与真实动作的平方误差，但附录明确动作头是 Diffusion Policy。不能据此把系统误读成普通单步 action regression；标准 DDPM 的具体加噪变量、噪声 target 与 reduction 在文中没有展开。</p></div>
          </>}

          {active === "results" && <>
            <div className="evidence-label">论文明确说明 · Figures 4-6 / Appendix F</div><h2>结果：强，但要看清指标</h2>
            <div className="metric-grid"><article><span>BASELINE AVG</span><b>12%</b><p>ImageNet、DINOv2、HRP 等视觉主干对比</p></article><article><span>MASQUERADE AVG</span><b>74%</b><p>三个任务、三个 OOD 场景的平均进度</p></article><article><span>ABSOLUTE GAIN</span><b>+62</b><p>percentage points，约 5-6×</p></article></div>
            <p>任务为 Stack Pots、Scrape Potato 与 Sweep Chilis。每个任务在三个未见场景各运行 10 次，共 30 次；训练示范只来自一个场景。图中的 “Success” 并非只统计完整成功：每个长任务拆成三个子任务，每完成一个获得 1/3 分数，因此 74% 更接近平均完成进度。</p>
            <h3 className="section-title">最有说服力的两组消融</h3>
            <div className="mechanism-grid"><div><b>去掉 Overlay</b><p>保持相同人类视频与 co-training，只使用原始人手画面，性能陡降，证明视觉本体对齐本身不可替代。</p></div><div><b>去掉 Co-training</b><p>先在编辑视频上预训练，随后只用 policy loss 微调，性能同样大幅下降，证明持续辅助任务是核心环节。</p></div></div>
            <div className="equation compact">编辑视频比例：0% → 10% → 50% → 100%　　Stack Pots：2% → 26% → 47% → 68%</div>
            <p>数据规模实验只在 Stack Pots 的一个 OOD 场景进行，每档 25 次 rollout；趋势单调，但尚不能证明跨任务或更大规模下仍保持同样 scaling law。</p>
          </>}

          {active === "audit" && <>
            <div className="evidence-label">证据边界审计</div><h2>为什么它是一篇好文章</h2>
            <div className="audit-stack"><div><span>01</span><h3>问题拆得准</h3><p>不把跨本体差异全部甩给网络隐式学习，而是在数据层先显式缩小视觉 embodiment gap。</p></div><div><span>02</span><h3>Co-training 结论扎实</h3><p>论文不仅做“预训练有效”的常规对比，还直接证明停止辅助任务会导致明显退化，给出了可迁移的训练原则。</p></div><div><span>03</span><h3>不是 Action 迁移</h3><p>编辑视频从未进入机器人 action loss；它提升的是共享 ViT 的视觉与未来运动表征，不能称为把人类视频转换成完整机器人轨迹。</p></div><div><span>04</span><h3>不是完整世界模型</h3><p>二维 waypoint 预测使用未来监督，但没有未来视觉生成、latent dynamics 或递归 rollout。</p></div><div><span>05</span><h3>“零样本”边界</h3><p>零样本指未见测试场景，而不是新任务；每个任务仍有 50 条同任务机器人示范。</p></div><div><span>06</span><h3>证据范围有限</h3><p>只有三个厨房任务、一种双臂平台；动作维度、action horizon、控制频率、观察历史和两类 batch 采样比例均未充分报告。</p></div><div><span>07</span><h3>部署成本未回答</h3><p>DDPM 推理使用 100 步，但没有给出端到端策略延迟和在线执行频率。</p></div></div>
            <div className="source-strip"><b>原始资料</b><p><a href="https://arxiv.org/abs/2508.09976" target="_blank" rel="noreferrer">arXiv 论文 ↗</a>　·　<a href="https://masquerade-robot.github.io/" target="_blank" rel="noreferrer">项目主页 ↗</a><br />作者：Marion Lepert, Jiaying Fang, Jeannette Bohg · Stanford University</p></div>
          </>}
        </article>
        <aside className="paper-aside"><span>TAKEAWAYS</span><ol><li>核心是持续 co-training</li><li>编辑视频不参与 action loss</li><li>Overlay 负责视觉本体对齐</li><li>未来 2D 轨迹增强 ViT</li><li>像世界模型，但不是世界模型</li></ol></aside>
      </section>
      <section className="next-note shell"><span>NEXT NOTE</span><h2>继续生长中。</h2><Link href="/notes">查看全部笔记 ↗</Link></section>
      <footer className="footer shell"><span>不凡天 · NOTE 003</span><Link href="/notes">返回笔记库 ↑</Link></footer>
    </main>
  );
}
