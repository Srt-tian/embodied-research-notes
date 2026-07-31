import Link from "next/link";
import { SiteHeader } from "../site-header";

export default function CodingSkillIndex() {
  return <main><SiteHeader /><section className="coding-index-hero shell"><span>CODING SKILL / TOOLBOX</span><div className="coding-index-title"><h1>更快定位，<br />更稳解决。</h1><p>把顶尖 coder 常用的工具、调试方法与工程判断，蒸馏成可以马上复制到项目里的短技巧。这里不按课程推进，只沉淀高复用能力。</p></div><div className="coding-stats"><div><span>SKILLS</span><b>01</b></div><div><span>FORMAT</span><b>QUICK NOTES</b></div><div><span>FOCUS</span><b>REAL WORK</b></div></div></section><section className="coding-archive shell"><div className="coding-archive-head"><div><span>SKILL LIBRARY</span><h2>技巧档案</h2></div></div><div className="coding-list"><Link href="/coding-skill/pdb-post-mortem" className="coding-row skill-row"><span>SKILL 01</span><div><h3>让 PDB 在异常后直接进入现场</h3><p>使用 post_mortem 检查报错瞬间的调用栈与局部变量</p></div><div className="row-tags"><span>Python</span><span>PDB</span><span>Debug</span></div><time>2026-07-20</time><b>↗</b></Link></div></section><footer className="footer shell"><span>不凡天 · CODING SKILL</span><span>持续蒸馏</span></footer></main>;
}
