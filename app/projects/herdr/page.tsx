import Link from "next/link";
import { SiteHeader } from "../../site-header";

export default function HerdrProjectPage() {
  return (
    <main>
      <SiteHeader />
      <article className="shell" style={{ maxWidth: 980, paddingTop: "5rem", paddingBottom: "6rem" }}>
        <p>RECOMMENDED PROJECT · AGENT INFRASTRUCTURE</p>
        <h1>Herdr</h1>
        <p style={{ fontSize: "1.25rem", maxWidth: 760 }}>
          面向 Coding Agent 的终端运行时与 multiplexer：把多个 Codex、Claude 等 CLI Agent 放进可持续、可观察、可编排的工作区。
        </p>

        <h2>为什么值得记录</h2>
        <p>
          Herdr 不替代 Codex 或 Cursor，也不负责提升单个模型的智能。它解决的是多 Agent 工程层的问题：长期保持 Agent 会话、统一查看状态、管理多个 pane / workspace，并通过 CLI 自动启动、提示、等待和读取子 Agent。
        </p>

        <h2>心智模型</h2>
        <pre>{`你的 Agent Workflow\n        │\n        ▼\n      Herdr\n   ┌────┼────┐\n   ▼    ▼    ▼\n Codex Codex Claude\n  impl  test  review`}</pre>

        <h2>适合的使用方式</h2>
        <p>
          第一阶段把它当作“为 Coding Agent 优化的 tmux”：一个项目一个 workspace，主开发、Repo/论文阅读、测试日志分别放在不同 pane。熟悉之后，再让主 Agent 通过 Herdr 的自动化接口创建和管理 researcher、implementer、reviewer、tester 等子 Agent。
        </p>

        <h2>快速开始</h2>
        <pre>{`# macOS\nbrew install herdr\n\ncd ~/your-project\nherdr\n\n# 进入 Herdr 后直接启动\ncodex`}</pre>
        <p>默认 prefix 为 Ctrl+B；可通过分屏和 workspace 管理多个长期运行的终端任务。</p>

        <h2>和现有工具的关系</h2>
        <p><strong>Codex / Claude / Cursor：</strong>负责完成具体 coding / research 任务。</p>
        <p><strong>Herdr：</strong>负责 Agent 的 terminal runtime、会话组织和自动化编排。</p>
        <p><strong>更上层的 workflow / harness：</strong>负责角色、任务拆分、流程规则和最终汇总。</p>

        <h2>推荐指数</h2>
        <p><strong>高。</strong>尤其适合同时运行多个 Codex CLI、测试进程和研究 Agent 的工作流。建议先从 2–3 个 pane 的人工管理开始，再逐步引入自动化编排。</p>

        <h2>Links</h2>
        <p><a href="https://herdr.dev/">Official Website ↗</a></p>
        <p><a href="https://herdr.dev/docs/">Documentation ↗</a></p>
        <p><a href="https://github.com/herdrdev/herdr">GitHub ↗</a></p>

        <p style={{ marginTop: "4rem" }}><Link href="/projects">← 返回推荐项目</Link></p>
      </article>
    </main>
  );
}
