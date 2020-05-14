# ディープラーニング∞本ノックぅぅ!!

DeepLearningの実装Tips (WiP)

**ノック作るのが大変になってきたのでTips集に変更しました**

- 【注意】このページを利用して、または関して生じた事に関しては、**私は一切責任を負いません。** すべて **自己責任** でお願い致します。
- なんとなく本とか買わずにDLを勉強したいーーーって人向けだと思う

もしこれがみなさんのお役に立ったらGithub Sponsorになってください！

~~何問になるか分からないので∞本になってます。多分これからいろんな技術が出るからどんどん更新する予定でっす。これはイモリと一緒にディープラーニングの基礎からDLのライブラリの扱い、どういうDLの論文があったかを実装しながら学んでいくための問題集です。本とか論文読んだだけじゃ机上の空想でしかないので、ネットワークの作成や学習率などのハイパーパラメータの設定を自分の手を動かしながら勉強するための問題集です。~~

~~**問題集として使ってもテンプレやチートシートとして使っても使い方は自由です！！！！**~~

## Related

★追記 2019.11.7

Study-AI株式会社様　http://kentei.ai/
のAI実装検定のシラバスに使用していただくことになりました！(画像処理１００本ノックも）
Study-AI株式会社様ではAIスキルを学ぶためのコンテンツを作成されており、AIを学ぶ上でとても参考になります！
検定も実施されてるので、興味ある方はぜひ受けることをお勧めします！

画像処理ノックはこっち

> [画像処理100本ノック!!](https://github.com/yoyoyo-yo/Gasyori100knock)

## Update

Twitterで更新を発信してますぅ

https://twitter.com/curry_frog

- 2020.5.3 Sun [pytorch] CycleGANを追加
- 2020.4.3 Fri [tf.keras] pix2pixを追加
- 2020.3.27 Thu [tf.keras] Style Transferを追加
- 2020.2.25 Tue [Pytorch] WGAN-GPを修正
- 2020.1.1 [Pytorch] EfficientNetB1~B7を追加
- 2019.12.30 [Pytorch] EfficientNetB0を追加
- 2019.12.23 Chainerのサポートが終了したらしいので、PytorchとTensorflowに絞っていきます
- 2019.12.23 [Pytorch] 可視化 Grad-CAMを追加
- 2019.11.23 [Pytorch] 言語処理・会話生成のHREDを追加
- 2019.11.19 [Pytorch] 画像生成のWGAN-GPを追加
- 2019.11.8 [Pytorch]　画像生成のVAEとalphaGANを追加
- 2019.10.28 [Pytorch] 画像生成のWGANを追加
- 2019.10.21 [PyTorch] Semantic SegmentationでSegNetを追加
- 2019.10.16 [PyTorch] Seq2Seq Hard Attentionを追加
- 2019.10.10 [PyTorch] Seq2Seq Attention(Step別)を追加
- 2019.9.30 [Pytorch] MobileNet v2 を追加
- 2019.9.19 [TensorFlow] Xception, MobileNet_v1 を追加
- 2019.9.16 [TensorFlow] ResNet 18, 34, 50, 101, 152 を追加
- 2019.8.19 [Pytorch] NLP: Seq2seq+Attention, word2vecを追加
- 2019.8.15 [Pytorch] pix2pixを追加
- 2019.8.4 [Pytorch] DenseNet121, 169, 201, 264を追加
- 2019.7.30 [PyTorch, Keras] Xceptionを追加
- 2019.7.28 [Keras] ResNeXt-50, 101を追加
- 2019.7.23 [Pytorch] ResNeXt-50, 101を追加
- 2019.7.17 [Pytorch] VAEを追加  [keras, tensorflow, chainer] CGAN(MNIST)を追加
- 2019.7.5 [pytorch, keras]ResNet18, 34, 101, 152を追加
- 2019.6.16 [pytorch, tensorflow, keras, chainer] ResNet50を追加
- 2019.6.9 [tensorflow] DCGANを追加
- 2019.6.7 [Pytorch, tensorflow, keras, chainer]GoogleNet-v1(Inception)を追加
- 2019.5.26 [tensorflow] DCGAN, Conditional GANを追加
- 2019.5.19 [Keras, Chainer] ConditionalGANを追加
- 2019.5.18 [データセット準備] MNIST, [Pytorch]ConditionalGANを追加
- 2019.5.2 [データセット準備] Cifar10、[AutoEncoder, ConvAutoEncoder, GAN, DCGAN]Cifar10を追加
- 2019.3.31 [画像認識モデル] APIを追加
- 2019.3.19 [Pytorch][Chainer] GAN, DCGANを追加
- 2019.3.17 Pooling layerを追加したけど、あとからクラス化と学習を追加する予定
- 2019.3.17 seq2seq, convolutional layer を追加
- 2019.3.16 ニューラルネットをクラス化　を追加
- 2019.3.13 パーセプトロン系を追加
- 2019.3.12 AutoEncoder, ConvAutoEncoder, パーセプトロンを追加
- 2019.3.9 GAN, DCGANを追加
- 2019.3.6 RNN, LSTM, BDLSTMを追加
- 2019.3.5 AutoEncoder, RNNを追加　
- 2019.3.4 データ拡張・回転を追加
- 2019.3.3 UNetを追加

## 環境設定

### 1. Minicondaのインストール

https://conda.io/miniconda.html からMinicondaをインストールする。（Window、MacOS）Minicondaインストール後、端末(Windowでは端末、MacOSではターミナル)を開き、以下コマンドで仮想環境を作成する。**またはGoogle colabolatoryならGPUが使用化。**

```bash
# 1. Create 
## 仮想環境を作成
$ conda create python=3.7 -n dl_env
# 仮想環境を開始
$ source activate dl_env
(dl_env) :~/work_space/DeepLearningMugenKnock/ :$ 
```

### 2. レポジトリのインストール

下記コマンドでこのレポジトリを自分のパソコンにインストール。

```bash
$ git clone https://github.com/yoyoyo-yo/DeepLearningMugenKnock.git
```

### 3. パッケージのインストール

最後に下記コマンドで必要なパッケージをインストールする。

```bash
# pytorch
$ pip install matplotlib opencv-python easydict torch torchvision torchsummary
# tensorflow-2.1
$ pip install matplotlib opencv-python easydict tensorflow==2.1
# keras
$ pip install matplotlib opencv-python easydict tensorflow keras
# chainer
$ pip install matplotlib opencv-python easydict chainer
```


## Tips

詳細は各ディレクトリのREADME

ここから基本的に、「python scripts_XXX/YYY.py --train --test」で動く。

### [Model](Scripts_Model)

| 問題 |  Implementation | Published year
|:---:|:---:|:---:|
| [ API ](Scripts_Model#api) | [&check; pytorch](Scripts_Model/scripts/api_pytorch.py) | 
| [LeNet](Scripts_Model#q-lenet) | [&check; pytorch](Scripts_Model/pytorch/LeNet_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/lenet_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/lenet_keras.py) [&check; chaienr](Scripts_Model/scripts_chainer/lenet_chainer.py) | 
| [AlexNet](Scripts_Model#q-alexnet) | [&check; pytorch](Scripts_Model/pytorch/AlexNet_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/alexnet_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/alexnet_keras.py) [&check; chainer](Scripts_Model/scripts_chainer/alexnet_chainer.py) | 2012 |
| [ZFNet](Scripts_Model#q-zfnet) | [&check; pytorch](Scripts_Model/pytorch/ZFNet_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/zfnet_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/zfnet_keras.py) [&check; chainer](Scripts_Model/scripts_chainer/zfnet_chainer.py) | 2013 |
| [Network in network](Scripts_Model#q-network-in-network) | [&check; pytorch](Scripts_Model/pytorch/NIN_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/nin_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/nin_keras.py) [&check; chainer](Scripts_Model/scripts_chainer/nin_chainer.py) | 2013 |
| [VGG16](Scripts_Model#q-vgg16) | [&check; pytorch](Scripts_Model/pytorch/VGG16_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/vgg16_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/vgg16_keras.py) [&check; chainer](Scripts_Model/scripts_chainer/vgg16_chainer.py) | 2014 |
| [VGG19](Scripts_Model#q-vgg19) | [&check; pytorch](Scripts_Model/pytorch/VGG19_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/vgg19_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/vgg19_keras.py) [&check; chainer](Scripts_Model/scripts_chainer/vgg19_chainer.py) | 2014 |
| [GoogLeNet-v1 (Inception)](Scripts_Model#q-googlenet-v1) | [&check; pytorch](Scripts_Model/pytorch/googletnetv1_pytorch.py)  [&check; tf.slim](Scripts_Model/scripts_tf_slim/googlenetv1_tensorflow_slim.py) [&check; keras](Scripts_Model/scripts_keras/googlenetv1_keras.py) [&check; chainer](Scripts_Model/scripts_chainer/googlenetv1_chainer.py) | 2014 |
| [ResNet-18](Scripts_Model#q-resnet) | [&check; pytorch](Scripts_Model/pytorch/ResNet18_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/res18_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/res18_keras.py) | 2015 |
| [ResNet-34](Scripts_Model#q-resnet) | [&check; pytorch](Scripts_Model/pytorch/ResNet34_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/res34_tensorflow_layers.py)  [&check; keras](Scripts_Model/scripts_keras/res34_keras.py) |  2015 |
| [ResNet-50](Scripts_Model#q-resnet) | [&check; pytorch](Scripts_Model/pytorch/ResNet50_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/res50_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts/res50_keras.py) [&check; chainer (WiP)](Scripts_Model/scripts_keras/res50_chainer.py) |  2015 |
| [ResNet-101](Scripts_Model#q-resnet) | [&check; pytorch](Scripts_Model/pytorch/ResNet101_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/res101_tensorflow_layers.py)  [&check; keras](Scripts_Model/scripts_keras/res101_keras.py) | 2015 |
| [ResNet-152](Scripts_Model#q-resnet) | [&check; pytorch](Scripts_Model/pytorch/ResNet152_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/res152_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/res152_keras.py) |  2015 |
| [ResNeXt-50](Scripts_Model#q-resnext) | [&check; pytorch](Scripts_Model/pytorch/ResNeXt50_pytorch.py)  [&check; tf.layers](Scripts_Model/scripts_tf_layers/resNeXt50_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/resNeXt50_keras.py) | 2016 |
| [ResNeXt-101](Scripts_Model#q-resnext) | [&check; pytorch](Scripts_Model/pytorch/ResNeXt101_pytorch.py)  [&check; keras](Scripts_Model/scripts_keras/resNeXt101_keras.py) | 2016 |
| [Xception](Scripts_Model#q-xception) | [&check; pytorch](Scripts_Model/pytorch/Xception_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/xception_tensorflow_layers.py) [&check; keras](Scripts_Model/scripts_keras/xception_keras.py) | 2016 |
| [DenseNet121](Scripts_Model#q-densenet) | [&check; pytorch](Scripts_Model/pytorch/DenseNet121_pytorch.py)  | 2016 |
| [DenseNet169](Scripts_Model#q-densenet) | [&check; pytorch](Scripts_Model/pytorch/DenseNet169_pytorch.py) | 2016 | 
| [DenseNet201](Scripts_Model#q-densenet) | [&check; pytorch](Scripts_Model/pytorch/DenseNet201_pytorch.py) | 2016 |
| [DenseNet264](Scripts_Model#q-densenet) | [&check; pytorch](Scripts_Model/pytorch/DenseNet264_pytorch.py) | 2016 |
| [MobileNet-v1](Scripts_Model#q-mobilenet-v1) | [&check; pytorch](Scripts_Model/pytorch/MobileNet_v1_pytorch.py) [&check; tf.layers](Scripts_Model/scripts_tf_layers/MobileNet_v1_tensorflow_layers.py) | 2017 |
| [MobileNet-v2](Scripts_Model#q-mobilenet-v2) | [&check; pytorch](Scripts_Model/pytorch/MobileNet_v2_pytorch.py) | 2019 |
| [EfficientNetB0](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB0_pytorch.py) | 2019 |
| [EfficientNetB1](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB1_pytorch.py) | 2019 |
| [EfficientNetB2](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB2_pytorch.py) | 2019 |
| [EfficientNetB3](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB3_pytorch.py)  | 2019 |
| [EfficientNetB4](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB4_pytorch.py) | 2019 |
| [EfficientNetB5](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB5_pytorch.py) | 2019 |
| [EfficientNetB6](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB6_pytorch.py) | 2019 |
| [EfficientNetB7](Scripts_Model#efficientnet) | [&check; pytorch](Scripts_Model/pytorch/EfficientNetB7_pytorch.py) | 2019 |

### [Interpretation](Scripts_Interpret)

| 問題 |  Implementation | Published Year | 
|:---:|:---:|:---:|
| [Grad-CAM](Scripts_Interpret#grad-cam) | [&check; pytorch](Scripts_Interpret/scripts_interpret/GradCam_pytorch.py) | 2016 |


### [Segmentation](Scripts_Segmentation)
| 問題 |  Implementation | Published Year | 
|:---:|:---:|:---:|
| [SemanticSegmentationとは?](Scripts_Segmentation#semanticsegmentation%E3%81%A8%E3%81%AF) |
| [Binalization](Scripts_Segmentation#binalization%E3%81%AB%E3%82%88%E3%82%8Bsemasegstep3-%E3%83%86%E3%82%B9%E3%83%88%E6%99%82%E3%81%AE%E4%BA%88%E6%B8%AC%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A1%A8%E7%A4%BA) |  [&check; pytorch](Scripts_Segmentation/pytorch/bin_test_pytorch.py) [&check; tf.slim](Scripts_Segmentation/scripts_tf_slim/bin_test_tensorflow_slim.py)  [&check; keras](Scripts_Segmentation/scripts_keras/bin_test_keras.py) [&check; chainer](Scripts_Segmentation/scripts_chainer/bin_test_chainer.py) |  
| [SemanticSegmentation](Scripts_Segmentation#semantic-segmentation-step3-%E3%83%86%E3%82%B9%E3%83%88%E6%99%82%E3%81%AE%E4%BA%88%E6%B8%AC%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A1%A8%E7%A4%BA) | [&check; pytorch](Scripts_Segmentation/pytorch/semaseg_test_pytorch.py) [&check; tf.slim](Scripts_Segmentation/scripts_tf_slim/semaseg_test_tensorflow_slim.py) [&check; keras](Scripts_Segmentation/scripts_keras/semaseg_test_keras.py) [&check; chainer](Scripts_Segmentation/scripts_chainer/semaseg_test_chainer.py) |
| [UpSampling. NearestNeighbor補間](Scripts_Segmentation#upsampling%E6%89%8B%E6%B3%951-nearestneighbor%E8%A3%9C%E9%96%93) |  [&check; pytorch](Scripts_Segmentation/pytorch/nearest_pytorch.py) [&check; tf.slim](Scripts_Segmentation/scripts_tf_slim/nearest_tensorflow_slim.py) [&check; keras](Scripts_Segmentation/scripts_keras/nearest_keras.py) [&check; chainer](Scripts_Segmentation/scripts_chainer/nearest_chainer.py) |
| [UpSampling. Transposed convolution](Scripts_Segmentation#upsampling%E6%89%8B%E6%B3%952-transposed-convolution) |  [&check; pytorch](Scripts_Segmentation/pytorch/transposeconv_pytorch.py) [&check; tf.slim](Scripts_Segmentation/scripts_tf_slim/transposeconv_tensorflow_slim.py)  [&check; keras](Scripts_Segmentation/scripts_keras/transposeconv_keras.py) [&check; chainer](Scripts_Segmentation/scripts_chainer/transposeconv_chainer.py) |
| [Feature Map Concat](Scripts_Segmentation#%E7%89%B9%E5%BE%B4%E3%83%9E%E3%83%83%E3%83%97%E3%81%AEconcat) |  [&check; pytorch](Scripts_Segmentation/scripts/concat_pytorch.py)  [&check; tf.slim](Scripts_Segmentation/scripts_tf_slim/concat_tensorflow_slim.py) [&check; keras](Scripts_Segmentation/scripts_keras/concat_keras.py) [&check; chainer](Scripts_Segmentation/scripts_chainer/concat_chainer.py) |
| [UNet](Scripts_Segmentation#unet) |  [&check; pytorch](Scripts_Segmentation/pytorch/unet_pytorch.py) [&check; tf.slim](Scripts_Segmentation/scripts_tf_slim/unet_tensorflow_slim.py) [&check; keras](Scripts_Segmentation/scripts_keras/unet_keras.py) [&check; chainer](Scripts_Segmentation/scripts_chainer/unet_chainer.py) | 2015 |
| [UNet-like](Scripts_Segmentation#unet%E9%A2%A8%E3%83%A2%E3%83%87%E3%83%AB) Binarization |  [&check; pytorch](Scripts_Segmentation/pytorch/UNetLike_Binary_pytorch.py)| 2015 |
| [UNet-like](Scripts_Segmentation#unet%E9%A2%A8%E3%83%A2%E3%83%87%E3%83%AB) |  [&check; pytorch](Scripts_Segmentation/pytorch/UNetLike_pytorch.py) [&check; tf.slim](Scripts_Segmentation/scripts_tf_slim/unetlike_tensorflow_slim.py)  [&check; keras](Scripts_Segmentation/scripts_keras/unetlike_keras.py) [&check; chainer](Scripts_Segmentation/scripts/unetlike_chainer.py) | 2015 |
| [SegNet](Scripts_Segmentation#segnet) Binalization|  [&check; pytorch](Scripts_Segmentation/pytorch/SegNet_Binarization_pytorch.py) | 2015 |
| [SegNet](Scripts_Segmentation#segnet)|  [&check; pytorch](Scripts_Segmentation/pytorch/SegNet_pytorch.py) | 2015 |


### [AE](Scripts_Generative)
| 問題 |  Implementation | Published Year |
|:---:|:---:|:---:|
| [AutoEncoder](Scripts_Generative#q-auto-encoder) MNIST |  [&check; pytorch](Scripts_Generative/pytorch/AE_MNIST_pytorch.ipynb) |
| [AutoEncoder](Scripts_Generative#q-auto-encoder) cifar10 |  [&check; pytorch](Scripts_Generative/pytorch/AE_CIFAR10_pytorch.ipynb) [&check; tf.slim](Scripts_Generative/scripts_tf_slim/ae_cifar10_tensorflow_slim.py) [&check; keras](Scripts_Generative/scripts_keras/ae_cifar10_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/ae_cifar10_chainer.py) |
| [AutoEncoder](Scripts_Generative#q-auto-encoder) |  [&check; pytorch](Scripts_Generative/pytorch/AE_pytorch.ipynb) [&check; tf.slim](Scripts_Generative/scripts_tf_slim/ae_tensorflow_slim.py) [&check; keras](Scripts_Generative/scripts_keras/ae_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/ae_chainer.py) |
| AutoEncoderによる異常検知 | [&check; [MNIST] pytorch](Scripts_Generative/pytorch/AE_AnormalyDet_MNIST_pytorch.ipynb) [&check; [FashionMNIST] pytorch](Scripts_Generative/pytorch/AE_AnormalyDetection_fashionmnist_pytorch.ipynb)
| [Convolutional AutoEncoder](Scripts_Generative#q-convolutional-auto-encoder) cifar10 |  [&check; pytorch](Scripts_Generative/pytorch/ConvAE_CIFAR10_pytorch.ipynb) [&check; tf.slim](Scripts_Generative/scripts_tf_slim/convae_cifar10_tensorflow_slim.py)[&check; keras](Scripts_Generative/scripts_keras/convae_cifar10_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/convae_cifar10_chainer.py) |
| [Convolutional AutoEncoder](Scripts_Generative#q-convolutional-auto-encoder) |  [&check; pytorch](Scripts_Generative/pytorch/ConvAE_pytorch.ipynb) [&check; tf.slim](Scripts_Generative/scripts_tf_slim/convae_tensorflow_slim.py) [&check; keras](Scripts_Generative/scripts_keras/convae_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/convae_chainer.py) |
| [VAE (Variational AutoEncoder)](Scripts_Generative#vae) MNIST |  [&check; pytorch](Scripts_Generative/pytorch/VAE_MNIST_pytorch.ipynb) |
| VAE + Clustering MNIST | [&check; pytorch](Scripts_Generative/pytorch/VAE_Clustering_MNIST_pytorch.ipynb) 

### [GAN](Scripts_Generative)
| 問題 |  Implementation | Published Year |
|:---:|:---:|:---:|
| [GAN](Scripts_Generative#q-gan) cifar10 | [&check; pytorch](Scripts_Generative/pytorch/gan_cifar10_pytorch.py) [&check; tf.slim (WiP)](Scripts_Generative/scripts_tf_slim/gan_cifar10_tensorflow_slim.py)  [&check; keras](Scripts_Generative/scripts_keras/gan_cifar10_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/gan_cifar10_chainer.py) |
| [GAN](Scripts_Generative#q-gan) | [&check; pytorch](Scripts_Generative/pytorch/gan_pytorch.py) | | [&check; tf.slim (WiP)](Scripts_Generative/scripts_tf_slim/gan_tensorflow_slim.py) | | [&check; keras](Scripts_Generative/scripts_keras/gan_keras.py) | [&check; chainer](Scripts_Generative/scripts_chainer/gan_chainer.py) |
| [DCGAN](Scripts_Generative#dcgan) cifar10 | [&check; pytorch](Scripts_Generative/pytorch/DCGAN_CIFAR10_pytorch.ipynb)   [&check; tf.keras](Scripts_Generative/scripts_tf_keras/DCGAN_cifar10_tf2.1_keras.py) [&check; tf.slim](Scripts_Generative/scripts_tf_slim/dcgan_cifar10_tensorflow_slim.py) [&check; keras](Scripts_Generative/scripts_keras/dcgan_cifar10_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/dcgan_cifar10_chainer.py) |
| [DCGAN](Scripts_Generative#dcgan) | [&check; pytorch](Scripts_Generative/pytorch/DCGAN_pytorch.ipynb) [&check; tf.slim](Scripts_Generative/scripts_tf_slim/dcgan_tensorflow_slim.py) [&check; keras](Scripts_Generative/scripts_keras/dcgan_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/dcgan_chainer.py)
| [Conditional GAN](Scripts_Generative#q-conditional-gan) MNIST | [&check; pytorch](Scripts_Generative/pytorch/CGAN_MNIST_pytorch.ipynb) [&check; tf.slim](Scripts_Generative/scripts_tf_slim/cgan_mnist_tensorflow_slim.py) [&check; keras](Scripts_Generative/scripts_keras/cgan_mnist_keras.py)  [&check; chainer](Scripts_Generative/scripts_chainer/cgan_mnist_chainer.py) | 2014 |
| [Conditional GAN](Scripts_Generative#conditional-gan) CIFAR10 | [&check; pytorch](Scripts_Generative/pytorch/CGAN_CIFAR10_pytorch.ipynb) [&check;tf.slim](Scripts_Generative/scripts_tf_slim/cgan_cifar10_tensorflow_slim.py) [&check; keras](Scripts_Generative/scripts_keras/cgan_cifar10_keras.py) [&check; chainer](Scripts_Generative/scripts_chainer/cgan_cifar10_chainer.py) |
| [pix2pix](Scripts_Generative#pix2pix) Seg | [&check; pytorch](Scripts_Generative/pytorch/Pix2pix_Seg_pytorch.ipynb) [&check; tf.keras](Scripts_Generative/scripts_tf_keras/pix2pix_tf2.1_keras.py) |2016|
| [pix2pix-GP](Scripts_Generative#pix2pix-GP)| [&check; pytorch](Scripts_Generative/pytorch/pix2pixGP_pytorch.py) | - |
| [WGAN](Scripts_Generative#wgan) CIFAR10 | [&check; pytorch](Scripts_Generative/pytorch/WGAN_CIFAR10_pytorch.ipynb)| 2017 |
| [WGAN](Scripts_Generative#wgan) | [&check; pytorch](Scripts_Generative/pytorch/WGAN_pytorch.ipynb)| 2017 |
| [WGAN-GP](Scripts_Generative#wgan-gp) CIFAR0 | [&check; pytorch](Scripts_Generative/pytorch/WGANGP_CIFAR10_pytorch.ipynb) | 2017 |
| [WGAN-GP](Scripts_Generative#wgan-gp) | [&check; pytorch](Scripts_Generative/pytorch/WGANGP_pytorch.ipynb)  | 2017 |
| [alpha-GAN](Scripts_Generative#alpha-gan) MNIST | [&check; pytorch](Scripts_Generative/pytorch/alphaGAN_mnist_pytorch.py) | 2017 |
| [alpha-GAN](Scripts_Generative#alpha-gan) cifar10 | [&check; pytorch](Scripts_Generative/pytorch/alphaGAN_cifar10_pytorch.py) |2017 |
| CycleGAN | [&check; pytorch](Scripts_Generative/pytorch/CycleGAN_pytorch.ipynb) 

### [Generative Model](Scripts_Generative)
| 問題 |  Implementation | Published Year |
|:---:|:---:|:---:|
| [Style Transfer]() |  [&check; tf.keras](Scripts_Generative/scripts_tf_keras/StyleTransfer_tf2.1_keras.py) |


### [NLP](Scripts_NLP)
| 問題 |  Implementation | Published year | 
|:---:|:---:|:---:|
| [RNN](Scripts_NLP#rnn) |  [&check; pytorch](Scripts_NLP/pytorch/rnn_pytorch.py) [&check; tf.slim](Scripts_NLP/scripts_tf_slim/rnn_tensorflow_slim.py) [&check; keras](Scripts_NLP/scripts_keras/rnn_keras.py) 
| [LSTM](Scripts_NLP#lstm) |  [&check; pytorch](Scripts_NLP/pytorch/lstm_pytorch.py) [&check; tf.slim](Scripts_NLP/scripts_tf_slim/lstm_tensorflow_slim.py) [&check; keras](Scripts_NLP/scripts_keras/lstm_keras.py) [&check; chainer](Scripts_NLP/scripts_chainer/lstm_chainer.py) |
| [Bi-directional LSTM](Scripts_NLP#bi-directional-lstm) |  [&check; pytorch](Scripts_NLP/pytorch/bdlstm_pytorch.py) [&check; tf.slim](Scripts_NLP/scripts_tf_slim/bdlstm_tensorflow_slim.py) [&check; keras](Scripts_NLP/scripts_keras/bdlstm_keras.py) [&check; chainer (WiP)](Scripts_NLP/scripts_chainer/bdlstm_chainer.py) |
| [GRU](Scripts_NLP#gru) |  [&check; pytorch](Scripts_NLP/pytorch/gru_pytorch.py)  [&check; tf.slim](Scripts_NLP/scripts_tf_slim/gru_tensorflow_slim.py) [&check; keras](Scripts_NLP/scripts_keras/gru_keras.py) [&check; chainer](Scripts_NLP/scripts_chainer/gru_chainer.py) |
| [Seq2seq](Scripts_NLP#seq2seq) | [&check; pytorch](Scripts_NLP/pytorch/seq2seq_pytorch.py)  [&check; keras](Scripts_NLP/scripts_keras/seq2seq_keras.py) | | | 2014 |
| [Transformer](Scripts_NLP#transformer-final-parameter-setting) | [&check; pytorch](Scripts_NLP/pytorch/Transformer_pytorch.py) ||||||2017|
| [Transformer (Hard Attention)](Scripts_NLP#transformer-hard-attention) | [&check; pytorch](Scripts_NLP/pytorch/Transformer_HardAttention_pytorch.py) |2014?|
| [HRED](Scripts_NLP#hred) | [&check; pytorch](Scripts_NLP/pytorch/HRED_pytorch.py) |2015|
| [Word2Vec (Skip-gram)](Scripts_NLP#word2vec-skip-gram) | [&check; pytorch](Scripts_NLP/pytorch/word2vec_pytorch.py) |

## Theory

|Num | [Theory 1](Scripts_theory) | Answer | Num | [Theory 2](Scripts_theory2) | Answer |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [パーセプトロン AND](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-and) | [&check;](Scripts_theory/scripts/perceptron_1.py) | 13 | [画像認識](Scripts_theory2/#%E7%94%BB%E5%83%8F%E8%AA%8D%E8%AD%98) | [&check;](Scripts_theory2/scripts/neuralnet_classification.py)
| 2 | [パーセプトロン 学習](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%AD%A6%E7%BF%92) | [&check;](Scripts_theory/scripts/perceptron_2.py)| 14 | [誤差関数](Scripts_theory2/#%E8%AA%A4%E5%B7%AE%E9%96%A2%E6%95%B0) | [&check;](Scripts_theory2/scripts/neuralnet_loss.py)
| 3 | [パーセプトロン 収束性](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%8F%8E%E6%9D%9F%E6%80%A7) | [&check;](Scripts_theory/scripts/perceptron_3.py) |  | [Sigmoid Cross Entropy](Scripts_theory2/#sigmoid-cross-entropy) | [&check;](Scripts_theory2/scripts/neuralnet_sce.py)
| 4 | [パーセプトロン Sigmoid](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-sigmoid) | [&check;](Scripts_theory/scripts/perceptron_sigmoid.py) |  | [Convolutional Layer](Scripts_theory2/#convolutional-layer) | [&check;](Scripts_theory2/scripts/conv_kernel.py)
| 5 | [パーセプトロン バイアス](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-bias) | [&check;](Scripts_theory/scripts/perceptron_sigmoid_bias.py) |  | [Padding](Scripts_theory2/#padding) | [&check;](Scripts_theory2/scripts/conv_pad.py)
| 6 | [パーセプトロン OR](Scripts_theory#q%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-or) | [&check;](Scripts_theory/scripts/perceptron_or.py) |  | [Stride](Scripts_theory2/#stride) | [&check;](Scripts_theory2/scripts/conv_stride.py)
| 7 | [パーセプトロン NOT](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-not) | [&check;](Scripts_theory/scripts/perceptron_not.py) |  | [Max-pooling layer](Scripts_theory2#max-pooling-layer) | [&check;](Scripts_theory2/scripts/maxpool.py)
| 8 | [パーセプトロン XOR](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-xor%E3%82%B2%E3%83%BC%E3%83%88) | [&check;](Scripts_theory/scripts/perceptron_xor.py ) |  | [Average-pooling layer](Scripts_theory2#average-pooling-layer) | [&check;](Scripts_theory2/scripts/avepool.py)
| 9 | [多層パーセプトロン FeedForward](Scripts_theory#q-%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-feedforward) | [&check;](Scripts_theory/scripts/multi_perceptron_1.py)
| 10 | [多層パーセプトロン 学習](Scripts_theory#q-%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%AD%A6%E7%BF%92) | [&check;](Scripts_theory/scripts/multi_perceptron_2.py)
| 11 | [更に多層パーセプトロン](Scripts_theory#q-%E6%9B%B4%E3%81%AB%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3) | [&check;](Scripts_theory/scripts/multi_perceptron_3.py)
| 12 | [ニューラルネットのクラス化](Scripts_theory#q-%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%81%AE%E3%82%AF%E3%83%A9%E3%82%B9%E5%8C%96) | [&check;](Scripts_theory/scripts/multi_perceptron_class.py)



## [Dataset](Scripts_Dataset)

### DataAugmentation

|番号|問題| 答え | | 番号|問題|
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [自分で用意したデータセットの読み込み](Scripts_Dataset#q2-1-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Dataset/scripts/answer_data_load.py)
| 2 | [ミニバッチの作成](Scripts_Dataset#q2-2-%E3%83%9F%E3%83%8B%E3%83%90%E3%83%83%E3%83%81%E4%BD%9C%E6%88%90) | [&check;](Scripts_Dataset/scripts/answer_minibatch.py)
| 3 | [イテレーション・エポック](Scripts_Dataset#q2-3-%E3%82%A4%E3%83%86%E3%83%AC%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%A8%E3%82%A8%E3%83%9D%E3%83%83%E3%82%AF) | [&check;](Scripts_Dataset/scripts/answer_epoch.py)
| 4 | [データ拡張・水平反転](Scripts_Dataset#q4-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E5%B7%A6%E5%8F%B3%E5%8F%8D%E8%BB%A2) | [&check;](Scripts_Dataset/scripts/answer_hf.py)
| 5 | [データ拡張・上下反転](Scripts_Dataset#q5-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E4%B8%8A%E4%B8%8B%E5%8F%8D%E8%BB%A2) | [&check;](Scripts_Dataset/scripts/answer_vf.py)
| 6 | [データ拡張・回転](Scripts_Dataset#q6-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E5%9B%9E%E8%BB%A2) | [&check;](Scripts_Dataset/scripts/answer_rotation.py)

### OpenSource
|問題| 答え |
|:---:|:---:|
| [MNIST](Scripts_Dataset#mnist) | [&check;](Scripts_Dataset/scripts/load_mnist.py)
| [CIFAR-10](Scripts_Dataset#cifar-10) | [&check;](Scripts_Dataset/scripts/load_cifar10.py)
| [Fashion MNIST](Scripts_Dataset#fashion-mnist) | [&check;](Scripts_Dataset/scripts/load_fashion_mnist.py)

## フレームワーク早見表

- [**CNN・フレームワークの使い方編**](Scripts_HowTo)
- [**共通事項**](Scripts_HowTo)

| | PyTorch | Tensorflow | Keras | Chainer | Caffe |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 入力 | [mb,c,h,w] | [mb, h, w, c] | [mb, h, w, c] | [mc, c, h, w] | [mb, c, h, w] |
| 教師ラベル | index [mb] | onehot [mb, cls] | onehot [mb, cls] | index [mb] | index [mb] |
| 速度 | まあまあ早い | 早い | 早い | 普通 | まあまあ早い？ |
| how to | [&check;](Scripts_HowTo/README_pytorch.md) | [&check;](Scripts_HowTo/README_tensorflow.md)  |  [&check;](Scripts_HowTo/README_keras.md) | [&check;](Scripts_HowTo/README_chainer.md) | [&check;install(Docker)](Scripts_HowTo/README_caffe_install_docker.md) <br>  [&check;install(Native)](Scripts_HowTo/README_caffe_install_native.md) |
| sample | [&check;](Scripts_HowTo/main_pytorch.py)  | [&check;(slim)](Scripts_HowTo/main_tensorflow_slim.py) <br> [&check;(layers)](Scripts_HowTo/main_tensorflow_layers.py) <br> [&check;(raw)](Scripts_HowTo/main_tensorflow_raw.py) | [&check;](Scripts_HowTo/main_keras.py)  | [&check;](Scripts_HowTo/main_chainer.py)  |


## Citation

```bash
@article{yoyoyo-yoDeepLearningMugenKnock,
    Author = {yoyoyo-yo},
    Title = {DeepLearningMugenKnock},
    Journal = {https://github.com/yoyoyo-yo/DeepLearningMugenKnock},
    Year = {2019}
}
```

## License

&copy; Curry yoshi All Rights Reserved.

This is under MIT License.
