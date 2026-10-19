import Link from "next/link";
import { SiteHeader } from "../../site-header";

const sections = [
  {title:"01 · Online Coding / Skill Agent",label:"EmbodiedEvo",intro:"Agent 在 Episode 中读图、决定子任务、生成或调用控制代码。Code 是动态 Skill；Python/Runtime 执行多个 Native Actions。",items:["Agent 决定抓取顺序、机械臂选择、目标和反馈后的改动。","Code / Skill 负责循环、插值、分支及底层 step(action)；无需每步调用 LLM。","Skills 与 Lessons 跨场景继承；程序执行次数不等于 LLM 推理次数。"]},
  {title:"02 · Recursive Self-Harness",label:"PhysicalRSI",intro:"System 1 在既有 Harness 下执行；System 2 主要跨 Rollout 分析失败，生成候选修改，再验证和继承。",items:["Harness 包含 Memory、Skills、Tools、Routing 和控制逻辑。","反思 Agent 并非逐个动作调用；可有 Episode 初始路由，但不等于逐步决策。","Vary → Evaluate → Select → Inherit，纯 Harness 更新不必训练 VLA 权重。"]},
  {title:"03 · Joint Agent–Harness–Model RSI",label:"WorldScape-R²SI",intro:"部分 Rollout 由 Inner LLM Agent 做在线高层决策，部分可直接由 WSP/Simate-beta 等反应式模型执行；Outer Agent 负责跨轮次演化。",items:["Reactive 表示直接由策略输出动作的执行方式，不排斥记忆或内部规划。","WSP 属于 WAM；Simate-beta 的具体 VLA/WAM 架构需谨慎核实。","模型更新需要已验证的数据、明确的训练监督和独立评估；不能只看 Harness 改动后的总分。"]},
];
export default function RoboDojoAgentParadigms() {
return <main><SiteHeader/>
<section className="paper-head shell"><div className="paper-breadcrumb"><Link href="/notes">NOTES</Link><span>/</span><span>RoboDojo Agent Paradigms</span></div>
<div className="paper-title masquerade-title"><span>RESEARCH NOTE · OCT 2026</span><h1>RoboDojo Agent</h1><p>三种技术范式：在线决策、Harness 反思与模型进化</p></div>
<div className="paper-meta"><div><span>TOPIC</span><b>AGENT / ROBOT RSI</b></div><div><span>DATE</span><b>2026-10-19</b></div><div><span>TYPE</span><b>RESEARCH SUMMARY</b></div></div></section>
<section className="paper-body shell"><article className="paper-content" style={{maxWidth:900,margin:"0 auto"}}>
<div className="evidence-label">核心理解 · 控制方式与更新对象是不同的分类轴</div>
<p className="paper-lead">EmbodiedEvo 研究 Agent 如何在线组织行为；PhysicalRSI 研究 Agent 如何反思并改进执行 Harness；WorldScape-R²SI 的联合路径进一步探索如何把执行经验转成 Skill 与模型能力。这三类是便于比较的系统范式，并不是彼此互斥的模型类别。</p>
<div className="insight"><span>最重要的边界</span><p>Code ≈ 动态 Skill；PhysicalRSI 的 System 2 主要负责 Rollout 后反思，不是每一步动作的 LLM；R²SI 的部分 Rollout 则确实有 Inner LLM Agent 在线参与。</p></div>
{sections.map(s=><section key={s.label} style={{marginTop:36}}><div className="evidence-label">{s.label}</div><h2>{s.title}</h2><p>{s.intro}</p><ol className="training-steps">{s.items.map(i=><li key={i}><p>{i}</p></li>)}</ol></section>)}
<h2 style={{marginTop:40}}>三范式对照</h2>
<div style={{overflowX:"auto"}}><table style={{borderCollapse:"collapse",width:"100%",fontSize:14}}><thead><tr>{["问题","EmbodiedEvo","PhysicalRSI","WorldScape-R²SI"].map(x=><th key={x} style={{textAlign:"left",padding:12,borderBottom:"1px solid #8885"}}>{x}</th>)}</tr></thead><tbody>
{[
["Rollout 中 LLM 决策","在线高层分段决策","System 2 不逐步参与","Agent 路径有；Direct Reactive 路径无外部 LLM"],
["谁执行机器人动作","Python Code / Skill","System 1 Harness / VLA","Skills 或 Reactive Policy"],
["Agent 更新什么","代码、Skills、Lessons","Harness 候选与版本","Skills/Harness，部分路线训练模型"],
["参数更新是否必须","否","否","仅 Model RSI 路线需要"]
].map(row=><tr key={row[0]}>{row.map((v,i)=><td key={i} style={{verticalAlign:"top",padding:12,borderBottom:"1px solid #8883"}}>{v}</td>)}</tr>)}</tbody></table></div>
<h2 style={{marginTop:40}}>研究启示</h2><p>针对双臂 Piper：可让在线 Agent 只负责目标、手臂选择与技能调用；System 1 用受约束 EEF Delta → IK → Qpos 控制；System 2 汇总真实状态、指令和视觉反馈并迭代 Harness；最后使用验证过的数据后训练 VLA/WAM。应逐层消融，而不是同时引入所有模块。</p>
<div className="source-strip"><b>完整可编辑研究文档</b><p><a href="https://github.com/Srt-tian/embodied-research-notes/blob/main/research/robodojo-agent-paradigms-2026-10.md" target="_blank" rel="noreferrer">Markdown 原文 ↗</a>　·　<a href="https://github.com/Einsia/EmbodiedEvo" target="_blank" rel="noreferrer">EmbodiedEvo ↗</a>　·　<a href="https://github.com/PhysicalRSI/PhysicalRSI" target="_blank" rel="noreferrer">PhysicalRSI ↗</a>　·　<a href="https://github.com/manifoldai-research/WorldScape-Policy" target="_blank" rel="noreferrer">WorldScape Policy ↗</a></p></div>
</article></section></main>
}