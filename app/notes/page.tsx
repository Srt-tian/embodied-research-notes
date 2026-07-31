import Link from "next/link";
import { SiteHeader } from "../site-header";
import { researchEntries } from "../content";

export default function NotesPage() {
  return (
    <main>
      <SiteHeader />
      <section className="index-hero shell">
        <span>RESEARCH NOTES / 研究笔记</span>
        <h1>把论文读成<br />可以复用的知识。</h1>
        <p>每篇笔记以数据流为起点，继续审计网络结构、Loss、时间对齐、梯度更新和训练—推理一致性。</p>
      </section>
      <section className="note-index shell">
        <div className="index-toolbar"><span>ALL NOTES</span><span>{String(researchEntries.length).padStart(2,"0")} ITEMS</span></div>
        {researchEntries.map((entry, index) => (
          <Link className="note-index-row" href={`/notes/${entry.slug.startsWith("pi05") ? "pi05" : entry.slug}`} key={entry.slug}>
            <span className="row-no">{String(index + 1).padStart(2,"0")}</span>
            <div className="row-title"><span>{entry.kind.toUpperCase()}</span><h2>{entry.title}</h2><p>{entry.subtitle}</p></div>
            <div className="row-tags">{entry.tags.map(tag => <span key={tag}>{tag}</span>)}</div>
            <time>{entry.date}</time><b>↗</b>
          </Link>
        ))}
        <div className="empty-row"><span>{String(researchEntries.length + 1).padStart(2,"0")}</span><p>下一篇论文解析会出现在这里</p><i>COMING NEXT</i></div>
      </section>
      <section className="index-categories shell">
        <h2>Browse by field</h2>
        <div><span>VLA</span><span>WORLD MODELS</span><span>DIFFUSION POLICY</span><span>ROBOT RL</span><span>ENGINEERING</span></div>
      </section>
      <footer className="footer shell"><span>不凡天 · NOTES INDEX</span><Link href="/">返回首页 ↑</Link></footer>
    </main>
  );
}
