# DAPI 到免疫组化图像的多器官虚拟染色

**2026 年第八届全球校园人工智能算法精英大赛 · 算法挑战赛道 · 国赛二等奖**（[组委会公示](https://www.aicomp.cn/notice/notice-4/6016.html)，附件 PDF 第 65 页）

我是程桦杰，来自华中科技大学，以零号团队唯一成员的身份完成了这个项目。我负责数据审计与验证设计、模型和训练实现、GPU 实验、结果分析及作品复现交付。赛题要求从单张显微 DAPI 图像生成指定免疫标记的图像。我用同一套参数处理结肠、肝脏和胃三种组织。

## 我做了什么

- **把三种组织放进同一个模型。** 我将 Restormer 的输入改为原生 256×256 单通道 DAPI，在潜在特征和输出前特征两处加入 Marker FiLM。结肠和胃共用 CD68 输出头，肝脏使用 CD45RO 输出头。两个头和共享骨干都保存在同一个 checkpoint。
- **处理多器官梯度冲突。** 胃的分数权重为 70%，我采用结肠、肝、胃为 1:2:5 的采样，并在共享骨干上以胃梯度为锚。若结肠或肝的任务梯度与胃梯度内积为负，我移除对应冲突分量，再合并梯度。这个做法受 PCGrad 启发，投影方向固定在胃任务上。
- **把数值稳定性纳入训练流程。** 我把 BF16 限于模型前向，以 FP32 计算损失，逐微批检查梯度是否有限，在梯度合并后裁剪，再执行 AdamW 更新。复现实验记录初始化、采样增强和推理随机种子 2026。
- **将本地验证与最终训练分开。** 我用固定 ROI 留出的本地验证集比较历史候选，并检查器官和 ROI 层面的表现。最终方案从公开 Gaussian Restormer 初始化独立训练，在 10,356 对官方训练样本上完成 14,000 次更新，无独立验证集。随后以固定学习率续训 2,450 次，每 245 次采样一个权重点，对 10 个权重点等权平均，得到最终模型。

## 半决赛机器评分

| 测试组织 | 输出 Marker | 总分权重 | 分数 |
| --- | --- | ---: | ---: |
| 结肠 | CD68 | 10% | 72.8693 |
| 肝脏 | CD45RO | 20% | 78.5307 |
| 胃 | CD68 | 70% | 72.6893 |
| **加权总分** | | **100%** | **73.8756** |

正式推理用一个模型、一个 checkpoint，每张图以 FP32 运行一次，输出 2,186 张 PNG。成绩衡量赛事图像生成任务中的表现。

## 阅读和运行

| 代码 | 内容 |
| --- | --- |
| [runtime/conditional_model.py](runtime/conditional_model.py) | 共享 Restormer、Marker FiLM 和双输出头 |
| [runtime/long_update.py](runtime/long_update.py) · [runtime/surgery.py](runtime/surgery.py) | 每步训练和以胃任务为锚的梯度投影 |
| [runtime/tail_engine.py](runtime/tail_engine.py) · [runtime/avgmath.py](runtime/avgmath.py) | 尾训和参数平均 |
| [config/final.json](config/final.json) · [reproduce.py](reproduce.py) | 最终参数、离线训练、检查与推理入口 |

安装 PyTorch、NumPy、Pillow、SciPy 和 scikit-image 后，以下测试使用合成输入检查单 checkpoint 推理路由、PNG 打包和非有限输出拦截。配置测试同时核验学习率日程和 AVG10 来源约束。

```bash
python -m unittest discover -s tests -v
```

正式复现所需的原始数据、索引和权重路径见[运行说明](docs/reproduction.md)。这个作品集公开核心代码、配置和测试。赛事原始图像、逐图标签与预测、最终权重及签署文件保存在本地参赛归档。

## 参考与许可

骨干改编自 [Restormer（CVPR 2022）](https://github.com/swz30/Restormer/tree/v1.0)，沿用该版本的 [Academic Public License](licenses/Restormer_LICENSE.md)。初始化来源见[公开权重说明](licenses/PRETRAINED_SOURCE.md)。梯度投影参考 [Gradient Surgery for Multi-Task Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。[官方赛题](https://www.aicomp.cn/tracks/tracks-1/3759.html)
