---
layout: post
title: VLAs and World-Action Models for Autonomous Driving in 2026
---

#### Introduction:

For a long time, autonomous driving stacks were built as a chain of separate modules: perception, prediction, then planning. End-to-end models changed this by learning a direct mapping from sensor data to a driving plan. Waymo's [EMMA](https://arxiv.org/abs/2410.23262) is a well-known example. It feeds camera images into Gemini and has the model write out the future trajectory, the detected objects and the road graph, all as text.

End-to-end driving still has a weak spot, though. The [Alpamayo-R1](https://arxiv.org/abs/2511.00088) paper from NVIDIA puts it plainly: models trained only by imitating human driving remain brittle in rare, safety-critical situations. These are exactly the cases where supervision is sparse and the model has little causal understanding to fall back on.

In 2025 and 2026, two families of foundation-model policies emerged as answers to this problem:

- **Vision-Language-Action models (VLAs)** start from a vision-language model (VLM) and adapt it to output a trajectory.
- **World-Action Models (WAMs)** start from a video generation or world-model backbone. They learn to predict both how the scene will evolve and what the car should do.

This post looks at how each family works in driving, which systems exist today, what the evidence says so far, and what is still unsolved. Every model and number links to its source. Where the driving literature does not yet answer a question, I say so rather than guess.

#### Why Foundation Models for Driving:

The two families bet on different kinds of prior knowledge.

A VLA borrows the knowledge of a large VLM. EMMA's authors argue that this lets the model draw on general world knowledge and reasoning when it plans. Alpamayo-R1 goes further and trains its model to write out a short causal explanation before choosing a trajectory. One example is "obstacle blocking lane, oncoming lane clear, safe to cross the line."

A WAM borrows something different. The [DriveWAM](https://arxiv.org/abs/2605.28544) authors point out that VLMs are pretrained mostly on static image-text pairs. Driving, however, depends on motion, object persistence and how the scene is likely to change over the next few seconds. Video generation models are pretrained on exactly that kind of data, so a policy built on one starts with those temporal priors already in place.

There is also a data argument that applies to both families. The [DriveVLA-W0](https://arxiv.org/abs/2510.12796) paper describes what it calls a "supervision deficit." A trajectory is only a handful of numbers, so a large model trained to output trajectories alone gets very little learning signal per example. Asking the model to also predict future images gives it far more to learn from.

#### VLA, World Model, WAM: The Vocabulary:

These three terms are easy to mix up.

- **VLA.** Starts from a vision-language model and is fine-tuned to output actions. In driving, the action is usually a future ego trajectory.
- **World model.** Predicts what the scene will look like next, given the current state and an action or prompt. On its own it does not drive the car; in driving it is often used as a simulator or data generator.
- **WAM.** Reuses a pretrained video or world-model backbone inside the driving policy. One network then models both the scene's evolution and the vehicle's actions.

I use the definitions from Moritz Reuss's [overview of WAMs for robotics](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/) on the NVIDIA Technical Blog, since the driving papers mostly adopt the same terms.

<figure style="text-align:center;">
  <img src="/images/driving-vla-wam-concept-map.png" alt="Concept map of VLA, world model, WAM and hybrid approaches for driving" />
  <p class="img-caption">VLA, world model, WAM and hybrids, with driving examples for each (Source: terminology from the <a href="https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/">NVIDIA Technical Blog</a>)</p>
</figure>

In practice the boundaries are already fuzzy. DriveVLA-W0 is a VLA with an added world-model loss. DriveWAM is a WAM that takes guidance from a VLM.

#### VLAs for Driving:

[EMMA](https://arxiv.org/abs/2410.23262) treats driving as a visual question-answering problem. Every output (the trajectory, the 3D boxes, the road graph) is written as plain text. One Gemini-based model handles all of these tasks through task-specific prompts. The authors report state-of-the-art planning on nuScenes, and find that training on all three tasks together improves each of them.

[Alpamayo-R1](https://arxiv.org/abs/2511.00088) pairs NVIDIA's Cosmos-Reason VLM with a diffusion-based trajectory decoder. It is trained on a new dataset of "Chain of Causation" reasoning traces tied to actual driving decisions. After supervised fine-tuning, the team applies reinforcement learning with two goals: improving the reasoning itself, and keeping the reasoning consistent with the trajectory the model outputs. They report a 45% gain in reasoning quality and a 37% gain in reasoning-action consistency from this RL stage. Model weights are public.

[AutoVLA](https://arxiv.org/abs/2506.13757) takes a different route to the action side. It discretises continuous trajectories into a vocabulary of feasible action primitives, so the language model can predict them like ordinary tokens. It learns two modes, a fast one that outputs a trajectory directly and a slower one that reasons first. It then uses GRPO-based reinforcement fine-tuning so the model only reasons when the scene calls for it.

[DriveVLA-W0](https://arxiv.org/abs/2510.12796) is a VLA with a world-model objective added on. Alongside the trajectory, the model is trained to predict future images. On a 70M-frame in-house dataset, the authors find that this extra objective makes performance improve faster as training data grows.

#### WAMs for Driving:

Driving WAMs differ mainly in one choice: whether the car has to generate future video before it can plan. Three designs are common:

- **Inverse dynamics.** The model first imagines the future, then works out the action that leads there. [DriveWAM](https://arxiv.org/abs/2605.28544) works this way. It adapts the Wan2.2-5B video diffusion transformer to generate the next few seconds of video latents, then decodes ego actions conditioned on that imagined future. A frozen Qwen3-VL-8B model supplies short text guidance for each chunk, such as yielding or merging. This adds the high-level intent that a video model lacks.
- **Joint prediction.** Future frames and the trajectory are predicted together. [Epona](https://arxiv.org/abs/2506.24113) is an autoregressive diffusion world model with twin diffusion transformers, one for the next frame and one for the trajectory. [UNIVERSE](https://arxiv.org/abs/2607.05133) goes further and puts both into a single diffusion transformer with shared parameters.
- **Representation-only.** Video prediction is used during training and switched off at test time. UNIVERSE supports this mode as well and reports a 4.3x speedup over joint rollout at comparable planning accuracy. [SimWAM](https://arxiv.org/abs/2608.07468) is built around the same idea. It uses future-video prediction to learn a motion prior, then predicts trajectories directly without generating frames.

<figure style="text-align:center;">
  <img src="/images/driving-four-routes.png" alt="Four pipelines from camera frames to a trajectory: VLA, inverse-dynamics WAM, joint-prediction WAM and representation-only WAM" />
  <p class="img-caption">Four routes from camera frames to a trajectory. The orange steps generate future video at test time. (Source: summarised from <a href="https://arxiv.org/abs/2605.28544">DriveWAM</a>, <a href="https://arxiv.org/abs/2506.24113">Epona</a>, <a href="https://arxiv.org/abs/2607.05133">UNIVERSE</a> and <a href="https://arxiv.org/abs/2608.07468">SimWAM</a>)</p>
</figure>

Two earlier systems show where this line of work started. Valeo's [VaVAM](https://arxiv.org/abs/2502.15672) treats driving as autoregressive video modelling on discrete VQ-VAE tokens with a GPT-style transformer, then adds an action expert for trajectories. [DriveLaW](https://arxiv.org/abs/2512.23421) keeps the video generator and the planner separate: a diffusion planner reads the generator's latent features.

Across these papers you can see a progression. The first systems built their own driving world models (VaVAM, Epona). Later ones adopted large open video backbones (DriveWAM on Wan). The most recent ones try to avoid generating video at inference altogether (UNIVERSE, SimWAM). Robotics went through the same shift, with papers like [Fast-WAM](https://arxiv.org/abs/2603.16666) asking whether test-time imagination is needed at all.

#### The Landscape at a Glance:

The table below collects the systems discussed above. It is a selection rather than a complete list: new driving WAM papers came out almost every month through mid-2026.

| Model | Source | Family | Core design | What it shows |
| --- | --- | --- | --- | --- |
| [EMMA](https://arxiv.org/abs/2410.23262) | Waymo, 2024 | VLA | Gemini MLLM; trajectories, 3D objects and road graph all output as text | State-of-the-art nuScenes planning; co-training the three tasks helps all three |
| [Alpamayo-R1](https://arxiv.org/abs/2511.00088) | NVIDIA, 2025 | Reasoning VLA | Cosmos-Reason VLM with a diffusion trajectory decoder; Chain of Causation data | RL improves reasoning quality by 45% and reasoning-action consistency by 37%; weights released |
| [AutoVLA](https://arxiv.org/abs/2506.13757) | NeurIPS 2025 | VLA | Trajectories as discrete action tokens; GRPO fine-tuning | Adaptive fast/slow reasoning for driving |
| [DriveVLA-W0](https://arxiv.org/abs/2510.12796) | ICLR 2026 | VLA with world-model loss | VLA trained to also predict future images | Better data scaling on a 70M-frame dataset |
| [VaVAM](https://arxiv.org/abs/2502.15672) | Valeo, 2025 | WAM | GPT-style video model on VQ-VAE tokens with an action expert | Early driving video-action model |
| [Epona](https://arxiv.org/abs/2506.24113) | ICCV 2025 | WAM (joint) | Autoregressive diffusion with twin DiTs for frame and trajectory | Joint video and trajectory prediction |
| [DriveLaW](https://arxiv.org/abs/2512.23421) | CVPR 2026 | WAM (hierarchical) | Diffusion planner on video-generator latents | Video latents as planner input |
| [DriveWAM](https://arxiv.org/abs/2605.28544) | CUHK-SZ / Didi, 2026 | WAM (inverse dynamics) | Wan2.2-5B video DiT with VLM guidance per chunk | 90.1 PDMS on NAVSIM v1 from one front camera |
| [UNIVERSE](https://arxiv.org/abs/2607.05133) | 2026 | WAM (joint, unified) | One mask-modulated DiT for video and trajectory | 4.3x faster trajectory-only inference |
| [SimWAM](https://arxiv.org/abs/2608.07468) | 2026 | WAM (representation-only) | Video prediction only during training | Plans without generating future frames |

<figure style="text-align:center;">
  <img src="/images/driving-vla-wam-timeline.png" alt="Timeline of driving VLAs and WAMs placed by arXiv month from October 2024 to August 2026" />
  <p class="img-caption">Driving VLAs (above the line) and WAMs (below), placed by the month in each arXiv ID (Source: arXiv listings linked in the table above)</p>
</figure>

#### How These Models Are Trained:

Most current driving models are trained in three stages, and the two families differ mainly in the first one.

1. **Pretraining.** A VLA inherits a VLM, such as Gemini in EMMA or Cosmos-Reason in Alpamayo-R1. A WAM inherits a video model; DriveWAM, for instance, starts from Wan2.2-TI2V-5B. This choice largely decides what the policy knows before it sees any driving data.
2. **Imitation learning on logged driving.** Both families learn to map camera input to ego trajectories, usually with an extra training signal on top. Alpamayo-R1 adds reasoning traces. DriveVLA-W0 adds future-image prediction. DriveWAM keeps its video loss alongside the action loss.
3. **Reinforcement learning.** Alpamayo-R1 uses RL to improve its reasoning and to keep that reasoning consistent with its actions. AutoVLA uses GRPO with verifiable planning rewards so that it reasons only when needed. Among the papers covered here, only the VLAs report this stage.

<figure style="text-align:center;">
  <img src="/images/driving-training-recipe.png" alt="Three training stages on the VLA and WAM paths, with DriveVLA-W0 shown as a hybrid" />
  <p class="img-caption">The three training stages for VLAs and WAMs in driving (Source: <a href="https://arxiv.org/abs/2511.00088">Alpamayo-R1</a>, <a href="https://arxiv.org/abs/2506.13757">AutoVLA</a>, <a href="https://arxiv.org/abs/2510.12796">DriveVLA-W0</a>, <a href="https://arxiv.org/abs/2605.28544">DriveWAM</a>)</p>
</figure>

#### What the Evidence Shows So Far:

The most useful single experiment I found is an ablation in the DriveWAM paper on the PhysicalAI-Autonomous-Vehicles benchmark:

| DriveWAM variant | ADE@4s (m, lower is better) | FDE@4s (m) |
| --- | --- | --- |
| Pretrained video backbone, joint video loss | 0.83 | 2.47 |
| From scratch, joint video loss | 1.10 | 3.26 |
| Pretrained backbone, action loss only | 1.23 | 3.79 |

The surprising part is that removing the video loss hurt more than dropping the pretrained weights entirely. It is not enough to start from a video model; the model has to keep predicting video during driving training. DriveVLA-W0 arrives at a similar conclusion from the VLA side.

On NAVSIM v1, the two families currently score very close to each other. DriveWAM reports a PDMS of 90.1 from a single front camera. The best DriveVLA-W0 variant reaches 90.2, also with a single camera. AutoVLA scores 89.1 and Epona 86.2. All four figures come from DriveWAM's comparison table. On this benchmark, at least, neither family is clearly ahead.

Speed is also closer than you might expect. DriveWAM reports about 871 ms per 4-second planning chunk on one NVIDIA H20 GPU, including video generation. NVIDIA's Alpamayo-1.5 takes about 900 ms on the same setup. Most of Alpamayo-1.5's time goes to the VLM, while most of DriveWAM's goes to video generation and action denoising.

These numbers need some caution. DriveWAM's PhysicalAI-AV results use a 1,000-clip test subset that the authors curated themselves. Its comparison with Alpamayo-1.5 is single-camera only, while Alpamayo-1.5 was trained on roughly 80,000 hours of data. Treat the figures as within-paper comparisons, not a leaderboard.

#### Open Problems:

So far, neither family has been shown to be safe enough to deploy on its own. For driving, I think five problems matter most.

The first is latency on real hardware. Both latency figures above were measured on datacenter GPUs, and we do not yet know how either approach would perform on in-vehicle compute. For WAMs, the main lever is to stop generating video at test time, which is why UNIVERSE and SimWAM matter.

The second is evaluation. Most results come from NAVSIM, which is [non-reactive by design](https://arxiv.org/abs/2406.15349), or from ADE/FDE against logged trajectories. Neither shows how a policy recovers when other road users react to what it does.

The third is that results are hard to compare across papers. They differ in cameras, data scale, test splits and backbones. A matched VLA-versus-WAM comparison exists for robot manipulation ([Zhang et al.](https://arxiv.org/abs/2603.22078)), but I could not find one for driving.

The fourth is that an imagined future can be wrong. Reuss's post shows a video model turning a robot gripper into a four-fingered hand. A WAM that plans on generated frames will carry errors like this into its trajectory, and there is no established way yet to detect or bound them.

The fifth applies to VLAs: a model's stated reasoning may not match what it actually does. This is why Alpamayo-R1 treats reasoning-action consistency as an explicit training target.

#### Conclusion:

I don't expect driving to settle on one of these approaches. VLAs bring semantic knowledge and the ability to reason about unusual situations. WAMs bring an understanding of motion and how a scene evolves. On current benchmarks their results are close, and some of the strongest systems already combine the two: DriveWAM takes guidance from a VLM, and DriveVLA-W0 adds a world-model loss to a VLA.

The clearest lesson so far is that predicting the future during training helps a driving policy, whichever family it belongs to. Whether the car also needs to generate that future at test time is still open, and the most recent papers suggest it often does not.

Over the next year I will be watching for three things:

- closed-loop, reactive evaluation of these models;
- latency measured on automotive hardware;
- a fair comparison of VLAs and WAMs trained on the same driving data.

#### References:

1. Hwang, J.-J., et al. "EMMA: End-to-End Multimodal Model for Autonomous Driving." arXiv 2024. [arXiv:2410.23262](https://arxiv.org/abs/2410.23262)
2. Wang, Y., et al. "Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail." arXiv 2025. [arXiv:2511.00088](https://arxiv.org/abs/2511.00088)
3. Zhou, Z., et al. "AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning." NeurIPS 2025. [arXiv:2506.13757](https://arxiv.org/abs/2506.13757)
4. Li, Y., et al. "DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving." ICLR 2026. [arXiv:2510.12796](https://arxiv.org/abs/2510.12796)
5. Bartoccioni, F., et al. "VaViM and VaVAM: Autonomous Driving through Video Generative Modeling." arXiv 2025. [arXiv:2502.15672](https://arxiv.org/abs/2502.15672)
6. Zhang, K., et al. "Epona: Autoregressive Diffusion World Model for Autonomous Driving." ICCV 2025. [arXiv:2506.24113](https://arxiv.org/abs/2506.24113)
7. Xia, T., et al. "DriveLaW: Unifying Planning and Video Generation in a Latent Driving World." CVPR 2026. [arXiv:2512.23421](https://arxiv.org/abs/2512.23421)
8. Shi, C., Xu, J., et al. "DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving." arXiv 2026. [arXiv:2605.28544](https://arxiv.org/abs/2605.28544)
9. "UNIVERSE: Unified Video Action Models for Autonomous Driving with Flexible Mask-Modulated Modality Generation." arXiv 2026. [arXiv:2607.05133](https://arxiv.org/abs/2607.05133)
10. "SimWAM: A Simple World Action Model for End-to-End Autonomous Driving." arXiv 2026. [arXiv:2608.07468](https://arxiv.org/abs/2608.07468)
11. Dauner, D., et al. "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking." NeurIPS 2024. [arXiv:2406.15349](https://arxiv.org/abs/2406.15349)
12. Yuan, T., et al. "Fast-WAM: Do World Action Models Need Test-time Future Imagination?" arXiv 2026. [arXiv:2603.16666](https://arxiv.org/abs/2603.16666)
13. Zhang, Z., et al. "Do World Action Models Generalize Better than VLAs? A Robustness Study." arXiv 2026. [arXiv:2603.22078](https://arxiv.org/abs/2603.22078)
14. Reuss, M. "Pretrained to Imagine, Fine-Tuned to Act: The Rise of World-Action Models." NVIDIA Technical Blog, June 2026. [developer.nvidia.com](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
