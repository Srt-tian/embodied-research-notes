import Link from "next/link";
import { SiteHeader } from "../../site-header";

const cases = [
 {id:"coding",num:"01",short:"在线决策",name:"Coding-as-Policy",project:"EmbodiedEvo",lead:"Agent 负责任务级决策与动态编写 Skill，代码负责把意图落实为连续机器人动作。",agent:"Episode 内，根据观测决定目标、臂选择、Skill 及代码修订",control:"Python / Skill → Native step(action)",update:"Skills + Lessons，不必更新模型权重",flow:["Observation","LLM Agent","Code / Skill","Robot Actions","Feedback → Agent"],points:["Code 本质是动态 Skill，不是一个新的低层策略网络。","一次代码执行可以包含许多 Native Actions；不等于一次 LLM forward。","高层语义变化或失败后，Agent 重新读取观测并修订程序。"]},
 {id:"harness",num:"02",short:"离线反思",name:"Recursive Self-Harness",project:"PhysicalRSI",lead:"System 1 执行已经确定的 Harness；System 2 在 Rollout 之后分析失败、修改并验证 Harness。",agent:"主要跨 Rollout 反思、诊断和生成新 Harness 候选",control:"System 1 Harness → π0.5 / Memory / Code Policy",update:"Routing、Memory、Skills、Tools 和控制代码",flow:["Harness Hₖ","Rollout + Evidence","Reflection Agent","Evaluate / Select","Harness Hₖ₊₁"],points:["反思 Agent 不需要逐个 timestep 在线调用。","候选必须经过独立验证，不能以 LLM 的主观判断代替成功率。","纯 Self-Harness 迭代通常保持底层模型参数冻结。"]},
 {id:"model",num:"03",short:"联合进化",name:"Agent–Harness–Model RSI",project:"WorldScape-R²SI",lead:"兼有在线 Inner Agent 与跨 Rollout Outer Recursion Agent；部分任务也可直接调用反应式模型。",agent:"Inner Agent 在部分 Rollout 中在线决策；Outer Agent 负责反思和迭代",control:"Agent + Skill，或 Direct Reactive Policy（WSP / Simate）",update:"Skill / Harness；有条件时进一步更新 Reactive Model",flow:["Inner Agent / Reactive","Rollout","Outer Agent + Verifier","Skill / Model Update","Re-evaluation"],points:["不是所有 Rollout 都调用外部 LLM；Direct Reactive 路径可直接输出动作。","Reactive 是执行方式：WSP 属于 WAM；Simate-beta 完整架构待核实。","Model RSI 必须用可验证轨迹训练并与仅更新 Harness 的效果做消融。"]}
];
const comparisons=[
 ["Rollout 内 LLM","参与高层分段决策","System 2 不逐步参与","部分路径参与"],
 ["低层动作执行","Code / Skill","System 1 Harness","Skills 或 Reactive Model"],
 ["更新对象","Skills / Lessons","Harness","Harness + 可选模型参数"],
 ["权重更新","不必","不必","Model RSI 路线需要"]
];
export default function RoboDojoAgentParadigms(){
 return <main className="rda-page"><SiteHeader/>
 <header className="rda-hero shell"><div className="paper-breadcrumb"><Link href="/notes">NOTES</Link><span>/</span><span>AGENT SYSTEMS</span></div>
 <div className="rda-kicker">RESEARCH NOTE / 2026.10.19</div><h1>RoboDojo<br/><em>Agent Paradigms</em></h1><p className="rda-subtitle">三种技术范式：在线决策、Harness 反思与模型进化</p>
 <div className="rda-summary"><strong>核心区分</strong><p>两个正交问题：<b>谁在 Rollout 中做决策？</b>以及<b>经验最终更新什么？</b> 三种范式是系统设计的归纳，不是互斥的模型类别。</p></div></header>
 <div className="rda-layout shell"><nav className="rda-nav" aria-label="文章目录"><span>CONTENTS</span>{cases.map(x=><a key={x.id} href={"#"+x.id}>{x.num}　{x.short}</a>)}<a href="#compare">04　横向比较</a><a href="#piper">05　研究启示</a><a className="rda-nav-source" href="https://github.com/Srt-tian/embodied-research-notes/blob/main/research/robodojo-agent-paradigms-2026-10.md">Markdown ↗</a></nav>
 <article className="rda-article"><section className="rda-intro"><span className="rda-overline">SYSTEMS AT A GLANCE</span><div className="rda-overview">{cases.map(c=><a href={"#"+c.id} key={c.id}><span>{c.num} / {c.project}</span><b>{c.short}</b><p>{c.lead}</p><i>↗</i></a>)}</div></section>
 {cases.map(c=><section className="rda-section" id={c.id} key={c.id}><div className="rda-section-head"><span>{c.num} / {c.project}</span><h2>{c.name}</h2><p>{c.lead}</p></div>
 <div className="rda-flow" aria-label={c.project+"执行流程"}>{c.flow.map((s,i)=><div className="rda-flow-item" key={s}><span>{String(i+1).padStart(2,"0")}</span><b>{s}</b>{i!==c.flow.length-1&&<span className="rda-flow-arrow">↓</span>}</div>)}</div>
 <div className="rda-facts"><div><span>AGENT ROLE</span><p>{c.agent}</p></div><div><span>ACTION EXECUTOR</span><p>{c.control}</p></div><div><span>IMPROVEMENT TARGET</span><p>{c.update}</p></div></div>
 <div className="rda-reading"><h3>关键理解</h3>{c.points.map((p,i)=><p key={p}><span>{String(i+1).padStart(2,"0")}</span>{p}</p>)}</div></section>)}
 <section id="compare" className="rda-section"><div className="rda-section-head"><span>04 / COMPARISON</span><h2>核心区别</h2><p>比较 Agent 介入时机、执行器与更新对象，比简单按模型名称分类更准确。</p></div><div className="rda-table-wrap"><table><thead><tr><th>维度</th><th>EmbodiedEvo</th><th>PhysicalRSI</th><th>WorldScape-R²SI</th></tr></thead><tbody>{comparisons.map(row=><tr key={row[0]}>{row.map((v,i)=><td key={i}>{v}</td>)}</tr>)}</tbody></table></div></section>
 <section id="piper" className="rda-section"><div className="rda-section-head"><span>05 / TAKEAWAYS</span><h2>双臂 Piper 的研究启示</h2></div><div className="rda-reading"><h3>分层，而不是让 LLM 控制每一帧</h3><p><span>01</span>Online Agent：按需决定子目标、左右臂与 Skill。</p><p><span>02</span>System 1：受约束的 EEF Delta → IK → Qpos，保持 30 Hz 底层控制与安全约束。</p><p><span>03</span>System 2：记录真实状态、命令和视觉反馈，跨 Rollout 修改 Harness 并独立验证。</p><p><span>04</span>Model RSI：积累高质量标注与验证轨迹后，再引入 VLA/WAM 后训练。</p></div></section>
 <div className="rda-sources"><span>FURTHER READING</span><a href="https://github.com/Einsia/EmbodiedEvo">EmbodiedEvo ↗</a><a href="https://github.com/PhysicalRSI/PhysicalRSI">PhysicalRSI ↗</a><a href="https://github.com/manifoldai-research/WorldScape-Policy">WorldScape Policy ↗</a><a href="https://github.com/Srt-tian/embodied-research-notes/blob/main/research/robodojo-agent-paradigms-2026-10.md">完整 Markdown ↗</a></div>
 </article></div></main>;
}