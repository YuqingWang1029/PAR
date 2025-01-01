# Parallelized Autoregressive Visual Generation

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.15119-b31b1b.svg)](https://arxiv.org/abs/2412.15119)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_demo-green)](https://epiphqny.github.io/PAR-project)&nbsp;

</div>

<img width="1195" alt="image" src="https://github.com/user-attachments/assets/54db2bdc-40f8-47e2-81a3-aa26cbbcd611" />

## BibTeX
```bibtex
@article{wang2024parallelized,
  title={Parallelized Autoregressive Visual Generation},
  author={Wang, Yuqing and Ren, Shuhuai and Lin, Zhijie and Han, Yujin and Guo, Haoyuan and Yang, Zhenheng and Zou, Difan and Feng, Jiashi and Liu, Xihui},
  journal={arXiv preprint arXiv:2412.15119},
  year={2024}
}
```

## Getting Started
### Requirements
- Linux with Python ≥ 3.7
- PyTorch ≥ 2.1
- A100 GPUs
  
We use the same environment as LLamaGen. For more details, please refer to [here](https://github.com/FoundationVision/LlamaGen/blob/main/GETTING_STARTED.md).

### VQ-VAE models
Method | params | tokens | rFID (256x256) | weight
--- |:---:|:---:|:---:|:---:
vq_ds16_c2i | 72M | 16x16 | 2.19 | [vq_ds16_c2i.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt) 

### AR models
Method | params | tokens | FID (256x256) | weight 
--- |:---:|:---:|:---:|:---:|
PAR-XL-4x    | 775M | 24x24 | 2.61 | [PAR-XL-4x.pt](https://huggingface.co/Epiphqny/PAR/resolve/main/PAR-XL-4x.pt)
PAR-XXL-4x   | 1.4B | 24x24 | 2.35 | [PAR-XXL-4x.pt](https://huggingface.co/Epiphqny/PAR/resolve/main/PAR-XXL-4x.pt)
PAR-3B-4x   | 3.1B | 24x24 | 2.29 | [PAR-3B-4x.pt](https://huggingface.co/Epiphqny/PAR/resolve/main/PAR-3B-4x.pt)
PAR-3B-16x   | 3.1B | 24x24 | 2.88 | [PAR-3B-16x.pt](https://huggingface.co/Epiphqny/PAR/resolve/main/PAR-3B-16x.pt)

Please download the above models, put them in the folder `./pretrained_models`

### Pre-extract discrete codes of training images
```
bash scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --data-path /path/to/imagenet/train --code-path /path/to/imagenet_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 384
```

### Train AR models with DDP
Before running, please change `nnodes, nproc_per_node, node_rank, master_addr, master_port` in `.sh`. The `spe-token-num` and `ar-token-num` represent the number of learnable tokens (`n-1`) and the number of tokens for parallel generation (`n`), respectively.
```
bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --spe-token-num 3 --ar-token-num 4 --image-size 384 --gpt-model GPT-XL

bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --spe-token-num 3 --ar-token-num 4 --image-size 384 --gpt-model GPT-XXL

bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk --code-path /path/to/imagenet_code_c2i_flip_ten_crop --spe-token-num 3 --ar-token-num 4 --image-size 384 --gpt-model GPT-3B
```

### Sampling
```

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/PAR-3B-4x.pt --spe-token-num 3 --ar-token-num 4 --gpt-model GPT-3B --image-size 384 --image-size-eval 256 --cfg-scale 1.345

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/PAR-1B-4x.pt --spe-token-num 3 --ar-token-num 4 --gpt-model GPT-XXL --image-size 384 --image-size-eval 256 --cfg-scale 1.435

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/PAR-XL-4x.pt --spe-token-num 3 --ar-token-num 4 --gpt-model GPT-XL --image-size 384 --image-size-eval 256 --cfg-scale 1.5

bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/PAR-3B-16x.pt --spe-token-num 15 --ar-token-num 16 --gpt-model GPT-3B --image-size 384 --image-size-eval 256 --cfg-scale 1.5
```


### Evaluation
Before evaluation, please refer [evaluation readme](evaluations/c2i/README.md) to install required packages. 
```
python3 evaluations/c2i/evaluator.py VIRTUAL_imagenet256_labeled.npz samples/GPT-XXL-PAR-XXL-4x-size-384-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.435-seed-0.npz
```

## Acknowledgments
The development of PAR is based on [LlamaGen](https://github.com/FoundationVision/LlamaGen). We deeply appreciate this contribution to the community.
