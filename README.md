# Tune-A-Video-ControlNet
This repository contains the pytorch implementation of [ControlNet](https://arxiv.org/abs/2302.05543)-attached [Tune-A-Video](https://arxiv.org/abs/2212.11565).<br>
This ControlNet-Attached Tune-A-Video, so called <b>"Tune-A-Video-ControlNet"</b> is used as one of comparison baselines in a [Ground-A-Video paper](https://arxiv.org/abs/2310.01107).


## Notes
1.  We apply the same Model Inflation Logic of Tune-A-Video on ControlNet, converting ControlNet2D to <b>ControlNet3D</b>.<br>
In specific, Self-Attentions are extended to Sparse-Causal Attentions in every transformer block, while Conv 2D blocks are replaced by Pseudo Conv 3D blocks. (The conv2d blocks in the ControlNet Hint Encoder, so called 'ControlNetConditioningEmbedding' class remains as conv2d since converting them drastically degrades performance.)<br>
2.  Then the <b>ControlNet-attached Tune-A-Video</b> model is trained on the input video for 500 steps.
3.  For future extensions, we implemented support for <b>Multi-ControlNet</b> class. In other words, you can attach multiple ControlNets to the Tune-A-Video backbone depending on your GPU constraints.


## Setup


## Training



## References
#### Thanks a lot for open-sourcing your works!
- [Official Tune-A-Video Implementation](https://github.com/showlab/Tune-A-Video)
- [Unofficial Tune-A-Video Implementation](https://github.com/bryandlee/Tune-A-Video)
- [Official ControlNet Implementation](https://github.com/lllyasviel/ControlNet)
- [Diffusers ... controlnet.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py)
- [Diffusers ... pipeline_controlnet.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet.py)

## Citation
