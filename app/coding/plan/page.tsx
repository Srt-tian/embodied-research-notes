import Link from "next/link";
import { SiteHeader } from "../../site-header";

const phases = [
  ["01–02", "独立编码地基", "Python、NumPy、接口设计、测试与调试"],
  ["03–04", "机器人数据管线", "序列切片、padding、mask 与时间对齐"],
  ["05–06", "深度学习实现", "PyTorch 训练、梯度、checkpoint 与验证"],
  ["07–08", "Diffusion / Flow", "条件动作生成、目标函数与采样路径"],
  ["09–10", "Transformer / VLA", "注意力掩码、多模态 token 与 action expert"],
  ["11–12", "闭环综合项目", "异步执行、日志、安全、延迟与部署"],
];

export default function CodingPlan() {
  return <main><SiteHeader />
    <section className="plan-head shell"><div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>训练计划</span></div><span>CURRICULUM · V1.0 · 2026-07-14</span><h1>12 周具身智能<br />Coding 训练计划</h1><p>目标不是少用 Codex，而是改变使用顺序：先独立分析与实现，再接受分级提示，最后对照参考方案并复盘。</p></section>
    <section className="plan-body shell"><div className="plan-principle"><span>TRAINING LOOP</span><h2>独立实现 → 测试验证 → 分级提示 → 自主修正 → 错误复测</h2></div><div className="phase-list">{phases.map(([week,title,description]) => <article key={week}><span>WEEK {week}</span><h3>{title}</h3><p>{description}</p></article>)}</div><div className="scoring-note"><span>DAILY REVIEW</span><h2>每天按 10 分制复盘</h2><p>正确性 3 分、独立性 2 分、代码质量 2 分、验证能力 2 分、解释能力 1 分。连续高分自动升级；同类错误出现三次则进入错误模式清单并安排无提示复测。</p></div></section>
    <section className="next-note shell"><span>NEXT</span><h2>Day 1 · Action Chunk 切片器</h2><Link href="/coding">返回训练档案 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>PLAN V1.0</span></footer>
  </main>;
}
