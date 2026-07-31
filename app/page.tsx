import Link from "next/link";
import { SiteHeader } from "./site-header";
import { expansionModules } from "./content";

export default function Home() {
  return (
    <main>
      <SiteHeader />
      <section className="home-hero shell">
        <div className="hero-meta"><span>不凡天 / BFT</span><span>SINGAPORE · 2026</span></div>
        <h1>研究具身智能，<br />也研究机器如何理解世界。</h1>
        <div className="hero-bottom">
          <p>聚焦 VLA、世界模型、Diffusion Policy 与机器人学习。这里保存的不只是结论，还有数据流、梯度链路与工程判断。</p>
          <a href="#gateway" className="circle-action" aria-label="查看内容入口"><span>探索内容</span><b>↓</b></a>
        </div>
      </section>

      <section className="home-gateway shell" id="gateway">
        <div className="eyebrow-row"><span>KNOWLEDGE SYSTEM</span><span>03 ENTRANCES</span></div>
        <div className="gateway-grid">
          <Link href="/notes" className="gateway-card notes-gateway">
            <div className="gateway-top"><span>01 · RESEARCH NOTES</span><b>↗</b></div>
            <h2>论文<br />笔记</h2>
            <p>VLA、WAM、视频世界模型与机器人学习。完整记录数据流、网络结构、Loss、时间对齐、训练推理和证据审计。</p>
            <div className="gateway-foot"><span>进入笔记库</span><i>持续更新</i></div>
          </Link>
          <Link href="/coding" className="gateway-card coding-gateway">
            <div className="gateway-top"><span>02 · CODING PRACTICE</span><b>↗</b></div>
            <h2>Coding<br />训练</h2>
            <p>从 NumPy、机器人数据管线到 Diffusion Policy 与 VLA。保存题目、思路、代码、测试、错误复盘与知识点。</p>
            <div className="gateway-foot"><span>进入训练档案</span><i>每日练习</i></div>
          </Link>
          <Link href="/coding-skill" className="gateway-card skill-gateway">
            <div className="gateway-top"><span>03 · CODING SKILL</span><b>↗</b></div>
            <h2>Coding<br />Skill</h2>
            <p>蒸馏顶尖 coder 的工具、调试技巧与工程习惯。每篇聚焦一个能立即复用的方法，并给出最小模板与适用边界。</p>
            <div className="gateway-foot"><span>进入技巧库</span><i>快速蒸馏</i></div>
          </Link>
        </div>
      </section>

      <section className="home-map shell" id="research">
        <div className="eyebrow-row"><span>RESEARCH INDEX</span><span>04 FIELDS</span></div>
        <div className="map-grid">
          <article><span>01</span><h3>Vision-Language-Action</h3><p>多模态融合、高低层推理与动作生成</p></article>
          <article><span>02</span><h3>World & Action Models</h3><p>视频世界模型、WAM 与长时序预测</p></article>
          <article><span>03</span><h3>Robot Learning</h3><p>Flow Matching、Diffusion Policy 与 RL</p></article>
          <article><span>04</span><h3>Real-world Systems</h3><p>OpenPI、UR5e、UAV 与 Sim-to-Real</p></article>
        </div>
      </section>

      <section className="archive-preview shell" id="archive">
        <div className="archive-intro"><span>THE ARCHIVE</span><h2>一个持续生长的<br />研究系统。</h2><p>每类内容都有独立入口，以后添加论文、实验或工程日志时，不需要重新设计网站。</p></div>
        <div className="archive-modules">
          {expansionModules.map((module) => (
            <article key={module.code}><span>{module.code}</span><div><h3>{module.title}</h3><p>{module.description}</p></div><i>＋</i></article>
          ))}
        </div>
      </section>

      <footer className="footer shell"><span>不凡天 · PERSONAL RESEARCH ARCHIVE</span><span>持续更新</span></footer>
    </main>
  );
}
