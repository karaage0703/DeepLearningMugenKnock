# ディープラーニング∞本ノックぅぅ!!

Update now!

何問になるか分からないので∞本になってます。多分これからいろんな技術が出るからどんどん更新する予定でっす。
これはイモリと一緒にディープラーニングの基礎からDLのライブラリの扱い、どういうDLの論文があったかを実装しながら学んでいくための問題集です。本とか論文読んだだけじゃ机上の空想でしかないので、ネットワークの作成や学習率などのハイパーパラメータの設定を自分の手を動かしながら勉強するための問題集です。

**問題集として使ってもテンプレやチートシートとして使っても使い方は自由です！！！！**

僕が研究室で３年修行してディープラーニングで必要そうだなーと思ったものを集めてます。

例えば研究してて提案手法を急にKerasでやればとか簡単に言うけどそんなすぐにできるかいってよくあると思うんだけどそういうのにもすぐ対応できる力を身につけるためのものだとも思ってます。

- **内容はいろいろな文献を調べて載っけてるので正しくないものもあるかもしれないので注意して下さい。もし間違ってたらプルリク下さい笑**
- 【注意】このページを利用して、または関して生じた事に関しては、**私は一切責任を負いません。** すべて **自己責任** でお願い致します。
- コードの書き方は私の趣向がけっこう出てるので、この書き方キモってなったら自分の書き方でやっていってください。答えはあくまで参考です。FWによってチョクチョク実装に小さな違いがあるのでそこはご愛嬌
- なんとなく本とか買わずにDLを勉強したいーーーって人向けだと思う

もしこれがみなさんのお役に立ったらGithub Sponsorになってください！

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

Python-3.6でやって下さい。(解答はPython-3.6で用意してます)

### 1. Minicondaのインストール

https://conda.io/miniconda.html のサイトからMinicondaをインストールします。これはWindowでもMacOSでも可能です。Minicondaがインストールできたら、端末(Windowでは端末、MacOSではターミナル)を開き、以下コマンドで仮想環境を作成します。**もしくはGoogle colabolatoryを使って見て下さい。GPUが使えます。**

```bash
$ conda create python=3.6 -n dlmugenknock
```

作成できたら、以下コマンドで仮想環境を動作します。

```bash
$ source activate dlmugenknock
```

するとこうなります。

```bash
(dlmugenknock) :~/work_space/DeepLearningMugenKnock/ :$ 
```

### 2. gitのインストール

gitをインストールします。そして、端末を開いて、以下のコマンドを実行します。このコマンドでこのディレクトリを丸ごと自分のパソコンにコピーできます。

```bash
$ git clone https://github.com/yoyoyo-yo/DeepLearningMugenKnock.git
```

### 3. パッケージのインストール

以下のコマンドで必要なパッケージをインストールします。これで準備は完了です！！


```bash
$ pip install -r requirements.txt
```

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

## 問題

詳細な問題内容は各ディレクトリのREADMEにあります。（ディレクトリで下にスクロールすればあります）

## 自分でフルスクラッチから実装する(理論)

## [理論編](Scripts_theory)

| 番号 | 問題 | 答え | | 番号 | 問題 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [パーセプトロン AND](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-and) | [&check;](Scripts_theory/answers/perceptron_1.py)
| 2 | [パーセプトロン 学習](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%AD%A6%E7%BF%92) | [&check;](Scripts_theory/answers/perceptron_2.py)
| 3 | [パーセプトロン 収束性](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%8F%8E%E6%9D%9F%E6%80%A7) | [&check;](Scripts_theory/answers/perceptron_3.py)
| 4 | [パーセプトロン Sigmoid](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-sigmoid) | [&check;](Scripts_theory/answers/perceptron_sigmoid.py)
| 5 | [パーセプトロン バイアス](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-bias) | [&check;](Scripts_theory/answers/perceptron_sigmoid_bias.py)
| 6 | [パーセプトロン OR](Scripts_theory#q%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-or) | [&check;](Scripts_theory/answers/perceptron_or.py)
| 7 | [パーセプトロン NOT](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-not) | [&check;](Scripts_theory/answers/perceptron_not.py)
| 8 | [パーセプトロン XOR](Scripts_theory#q-%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-xor%E3%82%B2%E3%83%BC%E3%83%88) | [&check;](Scripts_theory/answers/perceptron_xor.py )
| 9 | [多層パーセプトロン FeedForward](Scripts_theory#q-%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-feedforward) | [&check;](Scripts_theory/answers/multi_perceptron_1.py)
| 10 | [多層パーセプトロン 学習](Scripts_theory#q-%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3-%E5%AD%A6%E7%BF%92) | [&check;](Scripts_theory/answers/multi_perceptron_2.py)
| 11 | [更に多層パーセプトロン](Scripts_theory#q-%E6%9B%B4%E3%81%AB%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3) | [&check;](Scripts_theory/answers/multi_perceptron_3.py)
| 12 | [ニューラルネットのクラス化](Scripts_theory#q-%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%81%AE%E3%82%AF%E3%83%A9%E3%82%B9%E5%8C%96) | [&check;](Scripts_theory/answers/multi_perceptron_class.py)

## [理論編2](Scripts_theory2/)

| 番号 | 問題 | 答え | | 番号 | 問題 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 13 | [画像認識](Scripts_theory2/#%E7%94%BB%E5%83%8F%E8%AA%8D%E8%AD%98) | [&check;](Scripts_theory2/answers/neuralnet_classification.py)
| 14 | [誤差関数](Scripts_theory2/#%E8%AA%A4%E5%B7%AE%E9%96%A2%E6%95%B0) | [&check;](Scripts_theory2/answers/neuralnet_loss.py)
|  | [Sigmoid Cross Entropy](Scripts_theory2/#sigmoid-cross-entropy) | [&check;](Scripts_theory2/answers/neuralnet_sce.py)
|  | [Convolutional Layer](Scripts_theory2/#convolutional-layer) | [&check;](Scripts_theory2/answers/conv_kernel.py)
|  | [Padding](Scripts_theory2/#padding) | [&check;](Scripts_theory2/answers/conv_pad.py)
|  | [Stride](Scripts_theory2/#stride) | [&check;](Scripts_theory2/answers/conv_stride.py)
|  | [Max-pooling layer](Scripts_theory2#max-pooling-layer) | [&check;](Scripts_theory2/answers/maxpool.py)
|  | [Average-pooling layer](Scripts_theory2#average-pooling-layer) | [&check;](Scripts_theory2/answers/avepool.py)


## [データセット用意](Scripts_Dataset)

### 1. 自分で用意したデータセットを読み込む + 前処理

|番号|問題| 答え | | 番号|問題|
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [自分で用意したデータセットの読み込み](Scripts_Dataset#q2-1-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Dataset/answers/answer_data_load.py)
| 2 | [ミニバッチの作成](Scripts_Dataset#q2-2-%E3%83%9F%E3%83%8B%E3%83%90%E3%83%83%E3%83%81%E4%BD%9C%E6%88%90) | [&check;](Scripts_Dataset/answers/answer_minibatch.py)
| 3 | [イテレーション・エポック](Scripts_Dataset#q2-3-%E3%82%A4%E3%83%86%E3%83%AC%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%A8%E3%82%A8%E3%83%9D%E3%83%83%E3%82%AF) | [&check;](Scripts_Dataset/answers/answer_epoch.py)
| 4 | [データ拡張・水平反転](Scripts_Dataset#q4-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E5%B7%A6%E5%8F%B3%E5%8F%8D%E8%BB%A2) | [&check;](Scripts_Dataset/answers/answer_hf.py)
| 5 | [データ拡張・上下反転](Scripts_Dataset#q5-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E4%B8%8A%E4%B8%8B%E5%8F%8D%E8%BB%A2) | [&check;](Scripts_Dataset/answers/answer_vf.py)
| 6 | [データ拡張・回転](Scripts_Dataset#q6-%E3%83%87%E3%83%BC%E3%82%BF%E6%8B%A1%E5%BC%B5%E5%9B%9E%E8%BB%A2) | [&check;](Scripts_Dataset/answers/answer_rotation.py)

### 2.　オープンソースのデータセットを使う

|番号|問題| 答え | | 番号|問題|
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1-1 | [MNIST Step.1 ダウンロード](Scripts_Dataset#q-mnist-10-step1-%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89) | [&check;](Scripts_Dataset/answers/load_mnist_step1.py)
| 1-2 | [MNIST Step.2 学習データの読み込み](Scripts_Dataset#q-mnist-10-step2-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Dataset/answers/load_mnist_step2.py)
| 1-3 | [MNIST Step.Final テストデータの読み込み](Scripts_Dataset#q-mnist-10-stepfinal-%E3%83%86%E3%82%B9%E3%83%88%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Dataset/answers/load_mnist.py)
| 2-1 | [CIFAR-10 Step.1 ダウンロード](Scripts_Dataset#q-cifar-10-step1-%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89) | [&check;](Scripts_Dataset/answers/load_cifar10_step1.py)
| 2-2| [CIFAR-10 Step.2 学習データの読み込み](Scripts_Dataset#q-cifar-10-step2-%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Dataset/answers/load_cifar10_step2.py)
| 2-3 | [CIFAR-10 Step.Final テストデータの読み込み](Scripts_Dataset#q-cifar-10-stepfinal-%E3%83%86%E3%82%B9%E3%83%88%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Dataset/answers/load_cifar10.py)
| 3 | [Fashion MNIST](Scripts_Dataset#q-cifar-10-stepfinal-%E3%83%86%E3%82%B9%E3%83%88%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Dataset/answers/load_fashion_mnist.py)


## 自分でネットワーク組む編

ここから基本的に、「python answers/##.py --train --test」と打てば動きます。

### [画像認識編](Scripts_Model)

| 問題 |  PyTorch | TensorFlow | || Keras | Chainer | Published year
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 問題 |  PyTorch | tf.layers | tf.slim | tf.keras | Keras | Chainer | 
| [モデルの書き方の簡潔化](Scripts_Model#q-vgg19) | [&check;](Scripts_Model/answers/easy_pytorch.py) | [&check;](Scripts_Model/answers/easy_tensorflow_layers.py) | ||[&check;](Scripts_Model/answers/easy_keras.py) | [&check;](Scripts_Model/answers/easy_chainer.py) |
| [ API ](Scripts_Model#api) | [&check;](Scripts_Model/answers/api_pytorch.py) | 
| [LeNet](Scripts_Model#q-lenet) | [&check;](Scripts_Model/answers/lenet_pytorch.py) | [&check;](Scripts_Model/answers/lenet_tensorflow_layers.py) | || [&check;](Scripts_Model/answers/lenet_keras.py) | [&check;](Scripts_Model/answers/lenet_chainer.py) | 
| [AlexNet](Scripts_Model#q-alexnet) | [&check;](Scripts_Model/answers/alexnet_pytorch.py) | [&check;](Scripts_Model/answers/alexnet_tensorflow_layers.py) | ||[&check;](Scripts_Model/answers/alexnet_keras.py) | [&check;](Scripts_Model/answers/alexnet_chainer.py) | 2012 |
| [ZFNet](Scripts_Model#q-zfnet) | [&check;](Scripts_Model/answers/zfnet_pytorch.py) | [&check;](Scripts_Model/answers/zfnet_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/zfnet_keras.py) | [&check;](Scripts_Model/answers/zfnet_chainer.py) | 2013 |
| [Global Average Pooling](Scripts_Model#q-zfnet) | [&check;](Scripts_Model/answers/gap_pytorch.py) | [&check;](Scripts_Model/answers/gap_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/gap_keras.py) | [&check;](Scripts_Model/answers/gap_chainer.py) | 2013 |
| [Network in network](Scripts_Model#q-network-in-network) | [&check;](Scripts_Model/answers/nin_pytorch.py) | [&check;](Scripts_Model/answers/nin_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/nin_keras.py) | [&check;](Scripts_Model/answers/nin_chainer.py) | 2013 |
| [VGG16](Scripts_Model#q-vgg16) | [&check;](Scripts_Model/answers/vgg16_pytorch.py) | [&check;](Scripts_Model/answers/vgg16_tensorflow_layers.py) | || [&check;](Scripts_Model/answers/vgg16_keras.py) | [&check;](Scripts_Model/answers/vgg16_chainer.py) | 2014 |
| [VGG19](Scripts_Model#q-vgg19) | [&check;](Scripts_Model/answers/vgg19_pytorch.py) | [&check;](Scripts_Model/answers/vgg19_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/vgg19_keras.py) | [&check;](Scripts_Model/answers/vgg19_chainer.py) | 2014 |
| [GoogLeNet-v1(Inception)](Scripts_Model#q-googlenet-v1) | [&check;](Scripts_Model/answers/googletnetv1_pytorch.py) | | [&check;](Scripts_Model/answers/googlenetv1_tensorflow_slim.py) | | [&check;](Scripts_Model/answers/googlenetv1_keras.py) | [&check;](Scripts_Model/answers/googlenetv1_chainer.py) | 2014 |
| [Batch Normalization](Scripts_Model#q-vgg19) | [&check;](Scripts_Model/answers/bn_pytorch.py) | [&check;](Scripts_Model/answers/bn_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/bn_keras.py) | [&check;](Scripts_Model/answers/bn_chainer.py) | 2015 |
| [ResNet-18](Scripts_Model#q-resnet) | [&check;](Scripts_Model/answers/res18_pytorch.py) | [&check;](Scripts_Model/answers/res18_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/res18_keras.py) | | 2015 |
| [ResNet-34](Scripts_Model#q-resnet) | [&check;](Scripts_Model/answers/res34_pytorch.py) | [&check;](Scripts_Model/answers/res34_tensorflow_layers.py)  ||| [&check;](Scripts_Model/answers/res34_keras.py) | |  2015 |
| [ResNet-50](Scripts_Model#q-resnet) | [&check;](Scripts_Model/answers/res50_pytorch.py) | [&check;](Scripts_Model/answers/res50_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/res50_keras.py) | [&check;(not good)](Scripts_Model/answers/res50_chainer.py) |  2015 |
| [ResNet-101](Scripts_Model#q-resnet) | [&check;](Scripts_Model/answers/res101_pytorch.py) | [&check;](Scripts_Model/answers/res101_tensorflow_layers.py)  ||| [&check;](Scripts_Model/answers/res101_keras.py) | | 2015 |
| [ResNet-152](Scripts_Model#q-resnet) | [&check;](Scripts_Model/answers/res152_pytorch.py) | [&check;](Scripts_Model/answers/res152_tensorflow_layers.py)  ||| [&check;](Scripts_Model/answers/res152_keras.py) | |  2015 |
| [ResNeXt-50](Scripts_Model#q-resnext) | [&check;](Scripts_Model/answers/resNeXt50_pytorch.py) |  [&check;](Scripts_Model/answers/resNeXt50_tensorflow_layers.py) ||| [&check;](Scripts_Model/answers/resNeXt50_keras.py) | | 2016 |
| [ResNeXt-101](Scripts_Model#q-resnext) | [&check;](Scripts_Model/answers/resNeXt101_pytorch.py) | ||| [&check;](Scripts_Model/answers/resNeXt101_keras.py) | | 2016 |
| [Xception](Scripts_Model#q-xception) | [&check;](Scripts_Model/answers/xception_pytorch.py) | [&check;](Scripts_Model/answers/xception_tensorflow_layers.py) | ||[&check;](Scripts_Model/answers/xception_keras.py) | | 2016 |
| [DenseNet121](Scripts_Model#q-densenet) | [&check;](Scripts_Model/answers/DenseNet121_pytorch.py) | ||| | | 2016 |
| [DenseNet169](Scripts_Model#q-densenet) | [&check;](Scripts_Model/answers/DenseNet169_pytorch.py) | | ||| | 2016 | 
| [DenseNet201](Scripts_Model#q-densenet) | [&check;](Scripts_Model/answers/DenseNet201_pytorch.py) | | | ||| 2016 |
| [DenseNet264](Scripts_Model#q-densenet) | [&check;](Scripts_Model/answers/DenseNet264_pytorch.py) | | ||| | 2016 |
| [MobileNet-v1](Scripts_Model#q-mobilenet-v1) | [&check;](Scripts_Model/answers/MobileNet_v1_pytorch.py) | [&check;](Scripts_Model/answers/MobileNet_v1_tensorflow_layers.py) ||| | | 2017 |
| [MobileNet-v2](Scripts_Model#q-mobilenet-v2) | [&check;](Scripts_Model/answers/MobileNet_v2_pytorch.py) |||| | | 2019 |
| [EfficientNetB0](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB0_pytorch.py) |||| | | 2019 |
| [EfficientNetB1](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB1_pytorch.py) |||| | | 2019 |
| [EfficientNetB2](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB2_pytorch.py) |||| | | 2019 |
| [EfficientNetB3](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB3_pytorch.py) |||| | | 2019 |
| [EfficientNetB4](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB4_pytorch.py) |||| | | 2019 |
| [EfficientNetB5](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB5_pytorch.py) |||| | | 2019 |
| [EfficientNetB6](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB6_pytorch.py) |||| | | 2019 |
| [EfficientNetB7](Scripts_Model#efficientnet) | [&check;](Scripts_Model/answers/EfficientNetB7_pytorch.py) |||| | | 2019 |

### [Interpretation](Scripts_Interpret)
| 問題 |  PyTorch | TensorFlow | || Keras | Chainer | Published Year | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 問題 |  PyTorch | tf.layers | tf.slim | tf.keras | Keras | Chainer | - |
| [Grad-CAM](Scripts_Interpret#grad-cam) | [&check;](Scripts_Interpret/answers/GradCam_pytorch.py) | |||| | 2016 |



### [Semantic Segmentation編](Scripts_Segmentation)
| 問題 |  PyTorch | TensorFlow | || Keras | Chainer | Published Year | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 問題 |  PyTorch | tf.layers | tf.slim | tf.keras | Keras | Chainer | - |
| [SemanticSegmentationとは?](Scripts_Segmentation#semanticsegmentation%E3%81%A8%E3%81%AF) |
| [Binalization Step.1. データセット読み込み](Scripts_Segmentation#binalization%E3%81%AB%E3%82%88%E3%82%8Bsemasegstep1-%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Segmentation/answers/bin_dataset_pytorch.py) | | [&check;](Scripts_Segmentation/answers/bin_dataset_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/bin_dataset_keras.py) | [&check;](Scripts_Segmentation/answers/bin_dataset_chainer.py) |
| [Binalization Step.2. 学習時のLoss計算](Scripts_Segmentation#binalization-step2-%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%AEloss%E8%A8%88%E7%AE%97)| [&check;](Scripts_Segmentation/answers/bin_loss_pytorch.py) | | [&check;](Scripts_Segmentation/answers/bin_loss_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/bin_loss_keras.py) |  [&check;](Scripts_Segmentation/answers/bin_loss_chainer.py) | 
| [Binalization Step.3. テスト時の予測結果の表示](Scripts_Segmentation#binalization%E3%81%AB%E3%82%88%E3%82%8Bsemasegstep3-%E3%83%86%E3%82%B9%E3%83%88%E6%99%82%E3%81%AE%E4%BA%88%E6%B8%AC%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A1%A8%E7%A4%BA) |  [&check;](Scripts_Segmentation/answers/bin_test_pytorch.py) | | [&check;](Scripts_Segmentation/answers/bin_test_tensorflow_slim.py) |  | [&check;](Scripts_Segmentation/answers/bin_test_keras.py) | [&check;](Scripts_Segmentation/answers/bin_test_chainer.py) |  
| [SemanticSegmentation Step.1. データセット読み込み](Scripts_Segmentation#semantic-segmentation-step1-%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) | [&check;](Scripts_Segmentation/answers/semaseg_dataset_pytorch.py) | | [&check;](Scripts_Segmentation/answers/semaseg_dataset_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/semaseg_dataset_keras.py) | [&check;](Scripts_Segmentation/answers/semaseg_dataset_chainer.py) |
| [SemanticSegmentation Step.2. 学習時のLoss計算](Scripts_Segmentation#semantic-segmentation-step2-%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%AEloss%E8%A8%88%E7%AE%97)| [&check;](Scripts_Segmentation/answers/semaseg_loss_pytorch.py) | | [&check;](Scripts_Segmentation/answers/semaseg_loss_tensorflow_slim.py) | |[&check;](Scripts_Segmentation/answers/semaseg_loss_keras.py) | [&check;](Scripts_Segmentation/answers/semaseg_loss_chainer.py) |
| [SemanticSegmentation Step.3. テスト時の予測結果の表示](Scripts_Segmentation#semantic-segmentation-step3-%E3%83%86%E3%82%B9%E3%83%88%E6%99%82%E3%81%AE%E4%BA%88%E6%B8%AC%E7%B5%90%E6%9E%9C%E3%81%AE%E8%A1%A8%E7%A4%BA) | [&check;](Scripts_Segmentation/answers/semaseg_test_pytorch.py) | | [&check;](Scripts_Segmentation/answers/semaseg_test_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/semaseg_test_keras.py) | [&check;](Scripts_Segmentation/answers/semaseg_test_chainer.py) |
| [UpSampling手法1. NearestNeighbor補間](Scripts_Segmentation#upsampling%E6%89%8B%E6%B3%951-nearestneighbor%E8%A3%9C%E9%96%93) |  [&check;](Scripts_Segmentation/answers/nearest_pytorch.py) | | [&check;](Scripts_Segmentation/answers/nearest_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/nearest_keras.py) | [&check;](Scripts_Segmentation/answers/nearest_chainer.py) |
| [UpSampling手法2. Transposed convolution](Scripts_Segmentation#upsampling%E6%89%8B%E6%B3%952-transposed-convolution) |  [&check;](Scripts_Segmentation/answers/transposeconv_pytorch.py) | | [&check;](Scripts_Segmentation/answers/transposeconv_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/transposeconv_keras.py) | | [&check;](Scripts_Segmentation/answers/transposeconv_chainer.py) |
| [特徴マップのconcat](Scripts_Segmentation#%E7%89%B9%E5%BE%B4%E3%83%9E%E3%83%83%E3%83%97%E3%81%AEconcat) |  [&check;](Scripts_Segmentation/answers/concat_pytorch.py) | | [&check;](Scripts_Segmentation/answers/concat_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/concat_keras.py) | [&check;](Scripts_Segmentation/answers/concat_chainer.py) |
| [UNet](Scripts_Segmentation#unet) |  [&check;](Scripts_Segmentation/answers/unet_pytorch.py) | | [&check;](Scripts_Segmentation/answers/unet_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/unet_keras.py) | [&check;](Scripts_Segmentation/answers/unet_chainer.py) | 2015 |
| [UNet風モデル](Scripts_Segmentation#unet%E9%A2%A8%E3%83%A2%E3%83%87%E3%83%AB)|  [&check;](Scripts_Segmentation/answers/unetlike_pytorch.py) | | [&check;](Scripts_Segmentation/answers/unetlike_tensorflow_slim.py) | | [&check;](Scripts_Segmentation/answers/unetlike_keras.py) | [&check;](Scripts_Segmentation/answers/unetlike_chainer.py) | 2015 |
| [SegNet](Scripts_Segmentation#segnet)|  [&check;](Scripts_Segmentation/answers/SegNet_pytorch.py) | | | | | | 2015 |


### [画像生成編](Scripts_Generative)
| 問題 |  PyTorch | TensorFlow | || Keras | Chainer | Published Year |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 問題 |  PyTorch | tf.layers | tf.slim | tf.keras | Keras | Chainer | 
| [AutoEncoder](Scripts_Generative#q-auto-encoder) |  [&check;](Scripts_Segmentation/answers/ae_pytorch.py) | | [&check;?](Scripts_Generative/answers/ae_tensorflow_slim.py) | | [&check;](Scripts_Generative/answers/ae_keras.py) |  [&check;](Scripts_Generative/answers/ae_chainer.py) |
| [AutoEncoder cifar10](Scripts_Generative#q-auto-encoder) |  [&check;](Scripts_Segmentation/answers/ae_cifar10_pytorch.py) | | [&check;](Scripts_Generative/answers/ae_cifar10_tensorflow_slim.py)  | | [&check;](Scripts_Generative/answers/ae_cifar10_keras.py) |  [&check;](Scripts_Generative/answers/ae_cifar10_chainer.py) |
| [ConvolutionalAutoEncoder](Scripts_Generative#q-convolutional-auto-encoder) |  [&check;](Scripts_Segmentation/answers/convae_pytorch.py) | | [&check;?](Scripts_Generative/answers/convae_tensorflow_slim.py) | |[&check;](Scripts_Generative/answers/convae_keras.py) | [&check;](Scripts_Generative/answers/convae_chainer.py) |
| [ConvolutionalAutoEncoder cifar10](Scripts_Generative#q-convolutional-auto-encoder) |  [&check;](Scripts_Segmentation/answers/convae_cifar10_pytorch.py) |  | [&check;](Scripts_Generative/answers/convae_cifar10_tensorflow_slim.py)  | | [&check;](Scripts_Generative/answers/convae_cifar10_keras.py) |  [&check;](Scripts_Generative/answers/convae_cifar10_chainer.py) |
| [VAE (Variational AutoEncoder) MNIST](Scripts_Generative#vae) |  [&check;](Scripts_Segmentation/answers/vae_mnist_pytorch.py) |
| [VAE MNIST　潜在変数の可視化](Scripts_Generative#vae-潜在変数の可視化) |  [&check;](Scripts_Segmentation/answers/vae_latest_show_mnist_pytorch.py) |
| [VAE MNIST　潜在変数の操作による画像の生成1](Scripts_Generative#vae-潜在変数の操作による画像の生成) |  [&check;](Scripts_Segmentation/answers/vae_latent_change_mnist_pytorch.py) |
| [VAE MNIST　潜在変数の操作による画像の生成2](Scripts_Generative#vae-潜在変数の操作による画像の生成) |  [&check;](Scripts_Segmentation/answers/vae_latent_change2_mnist_pytorch.py) |
| [GAN](Scripts_Generative#q-gan) | [&check;](Scripts_Generative/answers/gan_pytorch.py) | | [&check; not good](Scripts_Generative/answers/gan_tensorflow_slim.py) | | [&check;](Scripts_Generative/answers/gan_keras.py) | [&check;](Scripts_Generative/answers/gan_chainer.py) |
| [GAN cifar10](Scripts_Generative#q-gan) | [&check;](Scripts_Generative/answers/gan_cifar10_pytorch.py) | | [&check; failed](Scripts_Generative/answers/gan_cifar10_tensorflow_slim.py)  | | [&check;](Scripts_Generative/answers/gan_cifar10_keras.py) | [&check;](Scripts_Generative/answers/gan_cifar10_chainer.py) |
| [DCGAN](Scripts_Generative#dcgan) | [&check;](Scripts_Generative/answers/dcgan_pytorch.py) | | [&check;](Scripts_Generative/answers/dcgan_tensorflow_slim.py) |  | [&check;](Scripts_Generative/answers/dcgan_keras.py) | [&check;](Scripts_Generative/answers/dcgan_chainer.py)
| [DCGAN cifar10](Scripts_Generative#dcgan) | [&check;](Scripts_Generative/answers/dcgan_cifar10_pytorch.py) | | [&check;](Scripts_Generative/answers/dcgan_cifar10_tensorflow_slim.py) | | [&check;](Scripts_Generative/answers/dcgan_cifar10_keras.py) | [&check;](Scripts_Generative/answers/dcgan_cifar10_chainer.py) |
| [Conditional GAN mnist](Scripts_Generative#q-conditional-gan) | [&check;](Scripts_Generative/answers/cgan_mnist_pytorch.py) | | [&check;](Scripts_Generative/answers/cgan_mnist_tensorflow_slim.py)  | | [&check;](Scripts_Generative/answers/cgan_mnist_keras.py) |  [&check;](Scripts_Generative/answers/cgan_mnist_chainer.py) | 2014 |
| [Conditional GAN cifar10](Scripts_Generative#conditional-gan) | [&check;](Scripts_Generative/answers/cgan_cifar10_pytorch.py) | | [&check;](Scripts_Generative/answers/cgan_cifar10_tensorflow_slim.py)   | | [&check;](Scripts_Generative/answers/cgan_cifar10_keras.py) |  [&check;](Scripts_Generative/answers/cgan_cifar10_chainer.py) |
| [pix2pix](Scripts_Generative#pix2pix) | [&check;](Scripts_Generative/answers/pix2pix_segment_pytorch.py) | |||||2016|
| [pix2pix-GP](Scripts_Generative#pix2pix-GP)| [&check;](Scripts_Generative/answers/pix2pixGP_pytorch.py) |||||| - |
| [WGAN](Scripts_Generative#wgan) | [&check;](Scripts_Generative/answers/WGAN_cifar10_pytorch.py) |||||| 2017 |
| [WGAN-GP](Scripts_Generative#wgan-gp) | [&check;](Scripts_Generative/answers/WGAN-GP_pytorch.py)  |||||| 2017 |
| [WGAN-GP cifar10](Scripts_Generative#wgan-gp) | [&check;](Scripts_Generative/answers/WGAN-GP_cifar10_pytorch.py)  |||||| 2017 |
| [alpha-GAN](Scripts_Generative#alpha-gan) MNIST | [&check;](Scripts_Generative/answers/alphaGAN_mnist_pytorch.py) |||||| 2017 |
| [alpha-GAN](Scripts_Generative#alpha-gan) CIFAR10 | [&check;](Scripts_Generative/answers/alphaGAN_cifar10_pytorch.py) |||||| 2017 |

### [画像処理編]()

### [言語処理編](Scripts_NLP)
| 問題 |  PyTorch | TensorFlow | || Keras | Chainer | Published year | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 問題 |  PyTorch | tf.layers | tf.slim | tf.keras | Keras | Chainer |
| [1 hotベクトル化](Scripts_NLP#q-1hot%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E5%8C%96) |  [&check;](Scripts_NLP/answers/onehot.py) 
| [RNN (Many-to-one) Step.1. 学習 ](Scripts_NLP#q-rnn-many-to-one-step1-%E5%AD%A6%E7%BF%92) |  [&check;](Scripts_NLP/answers/rnn_pytorch.py) | | | [&check;](Scripts_NLP/answers/rnn_tensorflow_slim.py) | [&check;](Scripts_NLP/answers/rnn_keras.py) | |
| [RNN (Many-to-one) Step.2. テスト](Scripts_NLP#q-rnn-many-to-one-step2-%E3%83%86%E3%82%B9%E3%83%88) | [&check;](Scripts_NLP/answers/rnn_pytorch.py) | | | [&check;](Scripts_NLP/answers/rnn_tensorflow_slim.py) | [&check;](Scripts_NLP/answers/rnn_keras.py) |  |
| [LSTM (Many-to-one)](Scripts_NLP#q-lstm-many-to-one) |  [&check;](Scripts_NLP/answers/lstm_pytorch.py) | | |[&check;](Scripts_NLP/answers/lstm_tensorflow_slim.py) |[&check;](Scripts_NLP/answers/lstm_keras.py) | [&check;](Scripts_NLP/answers/lstm_chainer.py) |
| [Bi-directional LSTM (Many-to-one)](Scripts_NLP#q-bi-directional-lstm-many-to-one) |  [&check;](Scripts_NLP/answers/bdlstm_pytorch.py) | | | [&check;](Scripts_NLP/answers/bdlstm_tensorflow_slim.py) | [&check;](Scripts_NLP/answers/bdlstm_keras.py) | [&check;?](Scripts_NLP/answers/bdlstm_chainer.py) |
| [GRU (Many-to-one)](Scripts_NLP#q-gru-many-to-one) |  [&check;](Scripts_NLP/answers/gru_pytorch.py) | | | [&check;](Scripts_NLP/answers/gru_tensorflow_slim.py) | [&check;](Scripts_NLP/answers/gru_keras.py) | [&check;](Scripts_NLP/answers/gru_chainer.py) |
| [Seq2seq](Scripts_NLP#q-seq2seq-many-to-many) | [&check;](Scripts_NLP/answers/seq2seq_pytorch.py) | |  | [&check;](Scripts_NLP/answers/seq2seq_keras.py) | | | 2014 |
| [Transformer (Step1. Source Target Attention)](Scripts_NLP#q-seq2seq--attention-step1-source-target-attention) | [&check;](Scripts_NLP/answers/seq2seq_attention_sourceTargetAttention_pytorch.py) ||||||2017|
| [Transformer (Step2. Self Attention)](Scripts_NLP#q-seq2seq--attention-step2-self-attention) | [&check;](Scripts_NLP/answers/seq2seq_attention_selfAttention_pytorch.py) ||||||2017|
| [Transformer (Step3. Multi head Attention)](Scripts_NLP#q-seq2seq--attention-step3-multi-head-attention) | [&check;](Scripts_NLP/answers/seq2seq_attention_multiHeadAttention_pytorch.py) ||||||2017|
| [Transformer (Step4. Feed Forward Networ)](Scripts_NLP#q-seq2seq--attention-step4-feed-forward-network) | [&check;](Scripts_NLP/answers/seq2seq_attention_FFN_pytorch.py) ||||||2017|
| [Transformer (Step5. Positional Encoding)](Scripts_NLP#q-seq2seq--attention-step5-positional-encoding) | [&check;](Scripts_NLP/answers/seq2seq_attention_positionalEncoding_pytorch.py) ||||||2017|
| [Transformer (Final. Parameter setting)](Scripts_NLP#q-seq2seq--attention-final-parameter-setting) | [&check;](Scripts_NLP/answers/seq2seq_attention_pytorch.py) ||||||2017|
| [Transformer (Hard Attention)](Scripts_NLP#q-seq2seq--attention-final-parameter-setting) | [&check;](Scripts_NLP/answers/seq2seq_attention_pytorch.py) ||||||2014?|
| [HRED](Scripts_NLP#q-hred) | [&check;](Scripts_NLP/answers/HRED_pytorch.py) ||||||2015|
| [Word2Vec (Skip-gram)](Scripts_NLP#q-word2vec) | [&check;](Scripts_NLP/answers/word2vec_pytorch.py) |


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

LICENSE
