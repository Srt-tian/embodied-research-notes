import Link from "next/link";
import { SiteHeader } from "../../site-header";

const hookCode = `class ActivationRecorder:
    def __init__(self, model):
        self.records = {}
        self._handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            value = output.detach()
            self.records[name] = ActivationStats(
                shape=tuple(value.shape),
                mean=value.mean().item(),
                std=value.std(unbiased=False).item(),
            )
        return hook

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def attach_gradient_norm_hook(tensor, store, key):
    if not tensor.requires_grad:
        raise ValueError("Tensor must require gradients.")

    def hook(grad):
        store[key] = grad.norm(2).item()
        return grad

    return tensor.register_hook(hook)`;

export default function Day05PyTorchHooks() {
  return <main><SiteHeader />
    <section className="coding-note-head shell">
      <div className="paper-breadcrumb"><Link href="/coding">CODING 训练</Link><span>/</span><span>DAY 05</span></div>
      <div className="coding-note-title"><span>DAY 05 · 2026-07-16</span><h1>PyTorch<br />Hook 导读</h1><p>了解不修改 forward 即可观察中间激活和梯度的调试接口。本节重新定级为 C：看到源码能识别，需要时会查文档即可。</p></div>
      <div className="coding-score"><span>TRAINING LEVEL</span><b>C</b><i>了解即可</i></div>
    </section>
    <article className="coding-note-body shell">
      <aside className="coding-note-nav"><span>CONTENTS</span><a href="#task">01 / 定位</a><a href="#idea">02 / 原理</a><a href="#code">03 / 代码</a><a href="#tests">04 / 验证</a><a href="#review">05 / 复盘</a><a href="#knowledge">06 / 总结</a></aside>
      <div className="coding-note-content">
        <section id="task"><span className="record-label">01 · POSITIONING</span><h2>训练定位</h2><p>原任务要求遍历模块树、注册 forward hook 和 Tensor gradient hook。复盘后确认：Hook 是实用调试接口，但不是当前具身训练主干，不要求从零实现 Recorder。</p><div className="constraint-grid"><div><b>需要知道</b><p>Hook 的触发时机、输入输出，以及使用后需要 remove。</p></div><div><b>不要求默写</b><p>闭包封装、handle 生命周期管理和通用 Hook 框架。</p></div></div></section>
        <section id="idea"><span className="record-label">02 · MECHANISM</span><h2>核心原理</h2><ol className="step-list"><li>forward hook 注册在 Module 上，模块前向结束后收到 module、inputs、output。</li><li>Tensor hook 注册在 Tensor 上，反向传播经过该节点时收到 grad。</li><li>梯度存在于计算图的每个相关中间节点，不只存在于模型参数。</li><li>非叶子 Tensor 的梯度会被计算，但默认不长期保存在 .grad。</li></ol></section>
        <section id="code"><span className="record-label">03 · REFERENCE</span><h2>参考代码</h2><pre className="code-block"><code>{hookCode}</code></pre></section>
        <section id="tests"><span className="record-label">04 · VERIFICATION</span><h2>参考验证</h2><div className="test-result"><span>REFERENCE TESTS</span><b>13 passed in 9.79s</b><i>VERIFIED</i></div><p>验证模块遍历、参数统计、激活记录、闭包层名绑定、梯度 norm 观测和 handle 移除。</p></section>
        <section id="review"><span className="record-label">05 · REVIEW</span><h2>学习策略复盘</h2><div className="error-card"><span>PRIORITY</span><h3>覆盖 API 不等于掌握主干能力</h3><p>Hook 练习一度投入过深。后续训练明确采用 A/B/C 分级，把独立编码时间集中到 Dataset、mask、loss 和训练闭环。</p></div></section>
        <section id="knowledge"><span className="record-label">06 · KNOWLEDGE</span><h2>只记住三点</h2><div className="knowledge-grid"><article><span>FORWARD</span><h3>Module hook</h3><p><code>register_forward_hook</code> 用于观察层的输入或输出。</p></article><article><span>BACKWARD</span><h3>Tensor hook</h3><p><code>tensor.register_hook</code> 用于观察或修改流经 Tensor 的梯度。</p></article><article><span>LIFECYCLE</span><h3>及时移除</h3><p>注册返回 handle；使用完成后调用 <code>handle.remove()</code>。</p></article><article><span>CLOSURE</span><h3>闭包记住层名</h3><p>内部 hook 函数可保留外层 name；理解 LEGB 与 late binding 即可。</p></article></div></section>
      </div>
    </article>
    <section className="next-note shell"><span>NEXT</span><h2>Day 6 · 端到端 Action Chunk Policy</h2><Link href="/coding/day06-action-policy">进入 Day 6 ↗</Link></section>
    <footer className="footer shell"><span>不凡天 · CODING TRAINING</span><span>DAY 05 · LEVEL C</span></footer>
  </main>;
}
