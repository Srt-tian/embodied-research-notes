import Link from "next/link";
import { SiteHeader } from "../site-header";

const projects = [
  {
    slug: "herdr",
    category: "AGENT INFRASTRUCTURE",
    title: "Herdr",
    description: "面向 Coding Agent 的终端 runtime / multiplexer，用于长期运行、观察和编排多个 Codex、Claude 等 CLI Agent。",
    tags: ["Codex", "Multi-Agent", "Terminal", "Orchestration"],
    status: "推荐",
  },
];

export default function ProjectsPage() {
  return (
    <main>
      <SiteHeader />
      <section className="index-hero shell">
        <span>RECOMMENDED PROJECTS / 推荐项目</span>
        <h1>值得真正用起来的<br />开源项目与工具。</h1>
        <p>这里不做单纯收藏。只记录值得进入实际研究与开发工作流的项目，并说明它解决什么问题、怎么用、适用边界以及是否值得长期保留。</p>
      </section>

      <section className="note-index shell">
        <div className="index-toolbar"><span>PROJECT INDEX</span><span>{String(projects.length).padStart(2, "0")} ITEMS</span></div>
        {projects.map((project, index) => (
          <Link className="note-index-row" href={`/projects/${project.slug}`} key={project.slug}>
            <span className="row-no">{String(index + 1).padStart(2, "0")}</span>
            <div className="row-title">
              <span>{project.category}</span>
              <h2>{project.title}</h2>
              <p>{project.description}</p>
            </div>
            <div className="row-tags">{project.tags.map(tag => <span key={tag}>{tag}</span>)}</div>
            <time>{project.status}</time><b>↗</b>
          </Link>
        ))}
        <div className="empty-row"><span>{String(projects.length + 1).padStart(2, "0")}</span><p>后续发现的高价值项目会继续放在这里</p><i>KEEP CURATING</i></div>
      </section>

      <section className="index-categories shell">
        <h2>Browse by field</h2>
        <div><span>AGENT</span><span>CODING</span><span>RESEARCH</span><span>ROBOTICS</span><span>INFRASTRUCTURE</span></div>
      </section>
      <footer className="footer shell"><span>不凡天 · PROJECT INDEX</span><Link href="/">返回首页 ↑</Link></footer>
    </main>
  );
}
