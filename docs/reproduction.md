# 运行说明

原训练和推理环境为 Linux、Python 3.9.18、PyTorch 2.4.1+cu121、CUDA 12.1、cuDNN 9.1.0，使用 NVIDIA A800。其余依赖版本见 [requirements.txt](../requirements.txt)。`runtime/recipe.json` 保存早期配置，正式全量日程以 [config/final.json](../config/final.json) 和 [repro_contract.py](../repro_contract.py) 为准。

## 公开源码检查

```bash
python reproduce.py --help
python -m unittest discover -s tests -v
```

合成测试可在 CPU 运行。它检查代码接口和推理输出格式，无需参赛数据。

## 使用已授权的参赛资产复现

下表是离线入口所需的文件。参赛数据、索引和最终权重来自原提交归档。公开 Gaussian 初始权重可以从 [Restormer v1.0 release](https://github.com/swz30/Restormer/releases/tag/v1.0) 获取。

| 文件 | 用途 |
| --- | --- |
| `data/official.zip` | 三器官官方训练集和测试输入 |
| `data/train_full.csv` · `data/test.csv` | 最终全量训练索引、测试输入索引 |
| `pretrained/gaussian_color_denoising_blind.pth` | 固定的公开初始化版本 |
| `checkpoints/final.pt` | 正式推理使用的最终 AVG10 checkpoint |

```bash
python reproduce.py check
python reproduce.py infer --device cuda:0 --output /tmp/M23_reproduced.zip
python reproduce.py train --device cuda:0 --output /tmp/M23_retrained
```

`check` 在 CPU 校验资产身份、训练与测试索引、权重结构和评分器。`infer` 使用固定最终权重生成预测 ZIP。`train` 是独立的全量重训，写入指定的新目录。仅运行所需的命令。
