# TTD-AGS
Test-Time Defense via Adaptive Gaussian Smoothing

这是一种面向未知强度攻击的免训练自适应防御方法，主要针对CLIP实现。

该方法主要由三个核心步骤构成。首先，为避免过度加噪破坏零样本自然泛化能力，提出基于弱扰动探测的自适应样本分流（Adaptive Sample Routing，ASR）。ASR基于这样的发现：自然样本在叠加微小高斯噪声后，其分类置信度的变化幅度显著高于对抗样本。其次，本章观察到导致分类置信度变化最快的噪声标准差能够反映输入受到的攻击强度。基于此，设计了基于置信度变化率的自适应噪声映射（Adaptive Noise Mapping，ANM）。因此，通过计算不同高斯噪声下的置信度变化率，可以获取攻击强度的信息。最后，为缓解单一高斯噪声可能带来的误差，提出了基于预测熵的特征融合（Predictive Entropy-based Feature Fusion，PEFF）。

代码实现主要基于：R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning （https://github.com/TomSheng21/R-TPT）
