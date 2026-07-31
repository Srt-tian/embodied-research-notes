import Link from "next/link";
import { SiteHeader } from "../site-header";
import { codingEntries } from "../content";

export default function CodingIndex() {
  const published = codingEntries.filter((entry) => entry.status === "published");
  return (
    <main>
      <SiteHeader />
      <section className="coding-index-hero shell">
        <span>CODING TRAINING / 2026</span>
        <div className="coding-index-title"><h1>从会用，<br />到真正掌握。</h1><p>围绕具身智能持续训练 Python、机器人数据管线、PyTorch、Diffusion / Flow Matching、Transformer / VLA 与闭环部署。每篇记录都经过代码运行与测试检查。</p></div>
        <div className="coding-stats"><div><span>RECORDS</span><b>{String(published.length).padStart(2, "0")}</b></div><div><span>CURRICULUM</span><b>12 WEEKS</b></div><div><span>FOCUS</span><b>EMBODIED AI</b></div></div>
      </section>
      <section className="coding-archive shell">
        <div className="coding-archive-head"><div><span>TRAINING LOG</span><h2>练习档案</h2></div><Link href="/coding/plan">查看训练计划 ↗</Link></div>
        {published.length > 0 ? <div className="coding-list">{published.map((entry) => <Link href={`/coding/${entry.slug}`} className="coding-row" key={entry.slug}><span>DAY {String(entry.day).padStart(2, "0")}</span><div><h3>{entry.title}</h3><p>{entry.subtitle}</p></div><div className="row-tags">{entry.tags.map((tag) => <span key={tag}>{tag}</span>)}</div><time>{entry.date}</time><b>↗</b></Link>)}</div> : <div className="coding-empty"><span>00</span><div><h3>第一篇练习即将开始</h3><p>完成 Day 1 Action Chunk 切片器并通过测试后，记录会出现在这里。</p></div></div>}
      </section>
      <section className="record-standard shell"><span>RECORD STANDARD</span><div><h2>每次练习，留下完整证据。</h2><ol><li>题目与约束</li><li>解题思路</li><li>可运行代码</li><li>测试结果</li><li>错误复盘</li><li>知识点总结</li></ol></div></section>
      <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>持续迭代</span></footer>
    </main>
  );
}
