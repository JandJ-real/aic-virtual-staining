# 公开预训练来源

我采用 swz30/Restormer v1.0 的 Gaussian Color Blind Denoising 公开权重初始化共享骨干。
发布页：https://github.com/swz30/Restormer/releases/tag/v1.0
作者下载地址：https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth
文件：gaussian_color_denoising_blind.pth，104611957 bytes。
SHA256：4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd。
未引入其原始训练图像。输入卷积RGB核沿通道求和，新增可学习输入偏置；两个单通道输出头使用seed2026初始化，FiLM初始为恒等映射。
完整最终模型包含共享骨干、两处Marker FiLM及两个输出头，测试只加载模型文件目录中的final.pt，不加载公开初始化权重。
Restormer原始结构及Gaussian配置、Academic Public License完整保留在本目录。修改版本同许可分发，仅用于非商业科研及本次竞赛。
