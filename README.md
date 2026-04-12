<div align="center">
<h2>BézierFlow: Learning Bézier Stochastic Interpolant Schedulers for Few-Step Generation</h2>

[**Yunhong Min**](https://myh4832.github.io)* · [**Juil Koo**](https://63days.github.io)* · [**Seungwoo Yoo**](https://dvelopery0115.github.io) · [**Minhyuk Sung**](https://mhsung.github.io) (* Equal Contribution)

KAIST

<span style="font-size: 1.5em;"><b>ICLR 2026</b></span>

<a href="https://arxiv.org/abs/2512.13255"><img src='https://img.shields.io/badge/arXiv-BézierFlow-red' alt='Paper PDF'></a>
<a href='https://bezierflow.github.io'><img src='https://img.shields.io/badge/Project_Page-BézierFlow-green' alt='Project Page'></a>

<img src="./assets/bezierflow_teaser.png" alt="BezierFlow Teaser" width="100%">
</div>

<blockquote align="center">
We introduce BézierFlow, a lightweight training approach for few-step generation with pretrained diffusion and flow models. BézierFlow achieves a 2–3× performance improvement for sampling with ≤ 10 NFEs while requiring only 15 minutes of training.
</blockquote>

## News
- **[2026.04.13]** 🚀 We have released the implementation of *BézierFlow: Learning Bézier Stochastic Interpolant Schedulers for Few-Step Generation*.
- **[2026.01.27]** 🔥 Our work has been accepted to ICLR 2026.

--- 

## Environment and Requirements

### Tested Environment
- **Python:** 3.10
- **CUDA:** 12.4
- **GPU:** Tested on NVIDIA RTX 3090 and RTX A6000

### Installation

```bash
conda create -n bezierflow python=3.10 -y
conda activate bezierflow
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Pretrained Checkpoints and FID References

```bash
mkdir -p pretrained fid-refs
```

### 1) Model checkpoints (`pretrained/`)

| Model | Source | Filename |
|-------|--------|----------|
| EDM (CIFAR-10) | [NVlabs/edm](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/) | `edm-cifar10-32x32-uncond-vp.pkl` |
| EDM (FFHQ) | [NVlabs/edm](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/) | `edm-ffhq-64x64-uncond-vp.pkl` |
| EDM (AFHQv2) | [NVlabs/edm](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/) | `edm-afhqv2-64x64-uncond-vp.pkl` |
| Rectified Flow | [RectifiedFlow](https://drive.google.com/file/d/10aPF5KC30SjVwr6rOnNosStpSGXnELXn/view?usp=sharing) | `reflow_1.pth` |
| FlowDCN | [MCG-NJU/FlowDCN](https://huggingface.co/wangsssssss/FlowDCN/blob/main/FlowDCN-XL-2M-R256.pth) | `FlowDCN-XL-2M-R256.pth` |

### 2) FID reference statistics (`fid-refs/`)

| Dataset | Source | Filename |
|---------|--------|----------|
| CIFAR-10 | [NVlabs/edm](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/) | `cifar10-32x32.npz` |
| FFHQ | [NVlabs/edm](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/) | `ffhq-64x64.npz` |
| AFHQv2 | [NVlabs/edm](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/) | `afhqv2-64x64.npz` |
| ImageNet 256 | [openai/guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations) | `VIRTUAL_imagenet256_labeled.npz` |

---

## Usage

BézierFlow follows a three-stage pipeline. Below are representative examples on CIFAR-10 for both a **diffusion model** (EDM) and a **flow model** (Rectified Flow). See `configs/` for all supported model configurations.

### Stage 1: Generate Teacher Data

Generate (latent, image) pairs from the pretrained model using the RK45 solver (can be different solvers):

- **EDM (Diffusion)**
    ```bash
    python gen_data.py \
        --all_config configs/cifar10_edm.yml \
        --total_samples 400 --sampling_batch_size 10 \
        --steps 1000 --solver_name rk45 --skip_type edm \
        --save_pt --save_png
    ```

- **Rectified Flow (Flow)**
    ```bash
    python gen_data.py \
        --all_config configs/cifar10_reflow.yml \
        --total_samples 400 --sampling_batch_size 10 \
        --steps 1000 --solver_name rk45 --skip_type rf \
        --save_pt --save_png
    ```

### Stage 2: Train BézierFlow

Optimize the Bézier noise scheduler with specific ODE solver and target NFE:

- **EDM (Diffusion)**
    ```bash
    python train.py \
        --all_config configs/cifar10_edm.yml \
        --solver_name uni_pc --steps 10
    ```

- **Rectified Flow (Flow)**
    ```bash
    python train.py \
        --all_config configs/cifar10_reflow.yml \
        --solver_name midpoint --steps 10
    ```

### Stage 3: Evaluate (FID)

Sample images using the learned schedule and compute FID:

- **EDM (Diffusion)**
    ```bash
    python compute_fid.py \
        --all_config configs/cifar10_edm.yml \
        --total_samples 50000 --sampling_batch_size 150 \
        --solver_name uni_pc --steps 10 \
        --load_from all_logs/cifar10_logs/
    ```

- **Rectified Flow (Flow)**
    ```bash
    python compute_fid.py \
        --all_config configs/cifar10_reflow.yml \
        --total_samples 50000 --sampling_batch_size 150 \
        --solver_name midpoint --steps 10 \
        --load_from all_logs/cifar10_logs/
    ```

---

## Supported Models and Solvers

| Config | Model | Dataset | Resolution | Supported Student Solvers |
|--------|-------|---------|------------|-------------------|
| `cifar10_edm.yml` | EDM | CIFAR-10 | 32×32 | UniPC, iPNDM |
| `ffhq.yml` | EDM | FFHQ | 64×64 | UniPC, iPNDM |
| `afhqv2.yml` | EDM | AFHQv2 | 64×64 | UniPC, iPNDM |
| `cifar10_reflow.yml` | Rectified Flow | CIFAR-10 | 32×32 | Euler (RK1), Midpoint (RK2) |
| `flowdcn_imagenet.yml` | FlowDCN | ImageNet | 256×256 | Euler (RK1), Midpoint (RK2) |

---

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{min2026bezierflow,
    title={B\'{e}zierFlow: Learning B\'{e}zier Stochastic Interpolant Schedulers for Few-Step Generation},
    author={Min, Yunhong and Koo, Juil and Yoo, Seungwoo and Sung, Minhyuk},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
}
```

---

## Acknowledgements

This repository builds upon the following projects:

- [LD3](https://github.com/vinhsuhi/ld3) (Tong et al.)
- [EDM](https://github.com/NVlabs/edm) (Karras et al.)
- [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) (Liu et al.)
- [FlowDCN](https://github.com/MCG-NJU/FlowDCN) (Wang et al.)
- [UniPC](https://github.com/wl-zhao/UniPC) (Zhao et al.)
