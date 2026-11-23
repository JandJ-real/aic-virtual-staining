# DAPI 到免疫组化图像的多器官虚拟染色

**2026 年第八届全球校园人工智能算法精英大赛 · 算法挑战赛道 · 国赛二等奖**

## 项目简介

这是一个面向结肠、肝脏和胃组织的免疫组化虚拟染色项目。模型输入单张 256×256 的 DAPI 显微图像，生成对应的免疫标记图像。结肠和胃预测 CD68，肝脏预测 CD45RO，三种组织共用一套 Restormer 骨干和同一个 checkpoint。

## 关键调优

- **Marker 条件建模。** 将 Restormer 适配为单通道输入，在潜在特征和输出头前使用 Marker FiLM 调节特征。CD68 和 CD45RO 各有一个输出头。
- **器官采样与梯度投影。** 针对胃部占总分 70% 的评分权重，按结肠、肝脏、胃为 1:2:5 采样。共享骨干以胃部梯度为锚，若结肠或肝脏的梯度与胃部梯度内积为负，先移除该梯度沿胃部梯度方向的冲突分量，再合并更新。
- **混合精度与数值稳定性。** 模型前向使用 BF16，损失在 FP32 下计算。每个微批反向后检查梯度是否有限，合并梯度后进行裁剪，再执行 AdamW 更新。初始化、采样增强和推理使用固定种子 2026。
- **ROI 验证与 AVG10。** 候选方案在固定 ROI 留出的本地验证集上比较。最终模型从公开 Gaussian Restormer 权重独立初始化，使用 10,356 对官方训练样本完成 14,000 次主训练更新。随后以固定学习率续训 2,450 次，每 245 次保留一组参数，将 10 组参数等权平均为最终权重。

## 半决赛机器评分

| 测试组织     | 输出 Marker | 总分权重 | 分数        |
| ------------ | ----------- | --------:| -----------:|
| 结肠         | CD68        | 10%      | 72.8693     |
| 肝脏         | CD45RO      | 20%      | 78.5307     |
| 胃           | CD68        | 70%      | 72.6893     |
| **加权总分** |             | **100%** | **73.8756** |

正式推理用一个模型、一个 checkpoint，每张图以 FP32 运行一次，输出 2,186 张 PNG。成绩衡量赛事图像生成任务中的表现。

## 阅读和运行

| 代码                                                                                        | 内容                                   |
| ------------------------------------------------------------------------------------------- | -------------------------------------- |
| [runtime/conditional_model.py](runtime/conditional_model.py)                                | 共享 Restormer、Marker FiLM 和双输出头 |
| [runtime/long_update.py](runtime/long_update.py) · [runtime/surgery.py](runtime/surgery.py) | 每步训练和以胃任务为锚的梯度投影       |
| [runtime/tail_engine.py](runtime/tail_engine.py) · [runtime/avgmath.py](runtime/avgmath.py) | 尾训和参数平均                         |
| [config/final.json](config/final.json) · [reproduce.py](reproduce.py)                       | 最终参数、离线训练、检查与推理入口     |

安装 PyTorch、NumPy、Pillow、SciPy 和 scikit-image 后，以下测试使用合成输入检查单 checkpoint 推理路由、PNG 打包和非有限输出拦截。配置测试同时核验学习率日程和 AVG10 来源约束。

```bash
python -m unittest discover -s tests -v
```

正式复现所需的原始数据、索引和权重路径见[运行说明](docs/reproduction.md)。这个作品集公开核心代码、配置和测试。赛事原始图像、逐图标签与预测、最终权重及签署文件保存在本地参赛归档。

## 参考与许可

骨干改编自 [Restormer（CVPR 2022）](https://github.com/swz30/Restormer/tree/v1.0)，沿用该版本的 [Academic Public License](licenses/Restormer_LICENSE.md)。初始化来源见[公开权重说明](licenses/PRETRAINED_SOURCE.md)。梯度投影参考 [Gradient Surgery for Multi-Task Learning（NeurIPS 2020）](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。[官方赛题](https://www.aicomp.cn/tracks/tracks-1/3759.html)
