import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="site-header shell">
      <Link className="site-mark" href="/"><b>不凡天</b><span>BU FAN TIAN</span></Link>
      <nav aria-label="主导航"><Link href="/">首页</Link><Link href="/notes">笔记</Link><Link href="/coding">Coding 训练</Link><Link href="/coding-skill">Coding Skill</Link><Link href="/#research">研究</Link></nav>
      <span className="availability">● RESEARCHING</span>
    </header>
  );
}
