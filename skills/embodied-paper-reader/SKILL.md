---
name: embodied-paper-reader
description: Deep-read embodied AI / robotics papers, project pages, and repos with emphasis on manipulation, VLA, world models, diffusion/flow policies, UMI/Ego data, cross-embodiment transfer, and reproducibility. Use when the user asks to read, explain, double-check, compare, reproduce, or trace a robotics paper or project.
---

# Embodied Paper Reader

## Goal
Produce an evidence-backed, implementation-oriented reading of embodied intelligence papers. Prioritize what is actually implemented and evaluated over abstract-level claims.

## Default workflow
1. Identify the canonical paper, project page, code repository, released checkpoints/datasets, and publication status.
2. Read the paper/project page first, then inspect source code/configs when claims depend on implementation details.
3. Separate three evidence levels explicitly:
   - **Paper-proven**: supported by experiments in the paper/whitepaper.
   - **Code-supported**: implemented in released code/configs but not necessarily evaluated in the paper.
   - **Author-claimed / planned**: stated as a capability or future work without sufficient evidence.
4. Explain the method from data flow rather than only prose:
   - input observations
   - state representation
   - action representation
   - coordinate frames
   - temporal sampling / action chunks
   - model backbone and conditioning
   - losses
   - training stages
   - inference loop
5. For every important claim, verify against original text or code. If evidence is ambiguous, say so.
6. End with reproduction cost, open-source status, practical limitations, and transferability to the user's setup.

## Mandatory technical checklist
When applicable, answer all of the following.

### Data
- What datasets are used? Human/robot/sim/UMI/Ego?
- Single-arm or bimanual?
- Camera count and placement: head/ego, workspace, wrist, third-person.
- Does the data include body pose, hand pose, IMU/VIO/SLAM, depth, point cloud, force, gripper width?
- How are trajectories synchronized and segmented?

### State and action
- Exact state vector and dimension if available.
- Exact action vector and dimension if available.
- Joint space vs Cartesian EEF/TCP.
- Absolute pose vs delta pose vs relative-to-current anchoring.
- Coordinate frame for every pose/action: robot base, table/world, camera, episode-local, controller, TCP.
- Whether different episodes have different frames and whether that introduces ambiguity.
- How gripper commands are represented.

### Retargeting / embodiment alignment
- Whether IK is used.
- Whether there is base search, workspace normalization, trajectory scaling, kinematic retargeting, learned retargeting, or no retargeting.
- Whether bimanual relative geometry is preserved.
- Whether human hand/UMI reachable-space mismatch is normalized.
- What happens to orientation and singular/unreachable poses.
- Whether the method transfers data offline or changes observations/actions online at inference.

### Model
- Inputs/outputs of each stage.
- Backbone type and tensor/token flow.
- Which parameters are shared across embodiments and which are embodiment-specific.
- How camera streams are fused and how missing/extra camera slots are handled.
- For diffusion/flow models: denoising variable, timestep conditioning, target (noise/velocity/flow), solver, action horizon.
- For TTT/memory/in-context methods: fast vs slow weights, inner loss, update frequency, reset behavior.

### Training
- Pretraining / co-training / post-training / finetuning stages.
- Exact losses and which data source contributes to each loss.
- Sampling ratios, paired/unpaired construction, DTW/OT/MixUp/alignment details when present.
- Whether robot data is required and how much.

### Inference and deployment
- Runtime inputs and outputs.
- Control frequency / model frequency if available.
- Whether IK/point-cloud reconstruction/retargeting is still required online.
- Sim-only vs real-robot evidence.
- Zero-shot vs few-shot vs finetuned transfer.

## UMI / Ego special mode
For UMI/Ego papers, always build the chain:
`human/controller measurement -> reconstructed trajectory -> canonical coordinate representation -> robot retargeting (if any) -> policy training target -> real robot command`.

Always check:
- head camera trajectory vs fixed camera assumption
- single vs dual hand/controller tracking
- controller pose frame and TCP calibration
- table/world calibration
- episode-to-episode frame consistency
- whether raw UMI/Ego data remains embodiment-independent
- whether conversion to joint targets makes the dataset embodiment-specific

## Cross-embodiment comparison mode
When comparing methods, use a compact table with at least:
- source embodiment/data
- target embodiment
- paired data requirement
- representation
- alignment mechanism
- target robot data amount
- true zero/few-shot status
- real-world evidence
- open source / reproducibility

## Reproduction grading
Rate separately:
- **Code availability**
- **Checkpoint availability**
- **Dataset availability**
- **Hardware dependence**
- **Calibration burden**
- **Compute burden**
- **Missing implementation details**

Do not call something reproducible merely because a repository exists.

## Response style
- Chinese by default.
- Start with a one-paragraph thesis: what the paper is fundamentally doing.
- Prefer diagrams in text form for pipelines.
- Explain difficult concepts from concrete tensor/data examples.
- Distinguish facts from interpretation.
- Give paper/project/repo links when available.
- If the user asks a narrow follow-up, answer that directly instead of re-summarizing the whole paper.

## User-specific evaluation lens
When relevant, explicitly assess compatibility with a **bimanual Piper setup using the official camera configuration**, especially:
- number and placement of camera streams
- bimanual geometry
- action/state coordinate system
- availability of Piper retargeting or joint conversion
- whether extra robot demonstrations are still required
- whether the released code can be adapted directly rather than re-implemented.
