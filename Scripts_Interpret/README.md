# Q. 理論の理解

## Grad-CAM

元論文
>> Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization https://arxiv.org/abs/1610.02391

### 論文のサマリ

ディープラーニングの判断根拠の見える化には３つの意味がある。
1. デプロイ時に信頼性
2. ユーザーからの信頼性
3. 機械が人間により良い判断を教える（machine teaching)

普通はAccuracyと単純性（解釈のしやすさ）はトレードオフだ。（Neural NetworkとLinear Regressionを比べれば明らかですよね)

それで良い可視化というのは、
1. class-discriminative (クラスがしっかり区別できるように見えていること,既存手法Guided Backpropagationはこれができていない)
2. 高画質であること
になる。

pytorchはこちらを参考 https://github.com/kazuto1011/grad-cam-pytorch

答え
- Pytorch [answers/GradCam_pytorch.py](answers/GradCam_pytorch.py)