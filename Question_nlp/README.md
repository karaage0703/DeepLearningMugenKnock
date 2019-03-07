# Q. 言語処理編

ここでは自然言語処理の手法を自分の手で実装していきます。**とりあえずPyTorch, Tensorflow, Keras, Chainer全部で実装してください。**
ネットワークを作ったら、学習率やイテレーションを変えて、テストデータセット *../Dataset/test/images* でテストしてみてください。
ぼくじしん画像処理の研究が専門だったので、ところどころ間違いがあるかもしれません。あったらプルリクください。

## Q. 1hotベクトル化

まずは1hot表現について。1hot表現とは対応するクラスの位置のみが1で、ほかが0である表現ほうほう。例えば、「あいうえお」において「い」を表現するには[0,1,0,0,0]となる。

sandwitchman.txtは僕がだいすきなとある芸人のコントをひらがなにおこしたものである。
これを1hotにする。

必要ならば、これを用いて下さい。

```bash
chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっー１２３４５６７８９０！？、。"
chars = [c for c in chars]
```

答え
- [answers/onehot.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/onehot.py)

## Q. RNN (Many-to-one) Step.1. 学習 

1hotベクトルにしたら、それを基にRNNを学習させ、**many-to-one** によりn-gramを実現してみましょう。

RNNにおいては、入力と出力でいくつかパターンがあります。ここではmany-to-oneを実現することで、直前のn文字から次の1文字を予測するRNNを作成し、自動的に文章を生成するRNNを実現します。

ただし、sandwitchman.txtをデータセット化する上で、文章を全部つなげて１つの文字列としましょう。その際に改行を#、最後に@、最初にn(n-gramのn)の数だけ@をつけましょう。ボキャブラリはこれを使っても良いです。改行をわざわざ#にしているのはテキストがコントなので、ボケとツッコミを分けるためです。

```bash
_chars = "あいうおえかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉゃゅょっー１２３４５６７８９０！？、。@#"
chars = [c for c in _chars]
```
10gram(n=10)として、モデルは
1. Input=(minibatch, n, class)
2. RNN(128) (chainerは64、128だと計算が遅くなるので)
3. Dense(class) + Softmax
とします。最適化は何でもいいけど、Adamがいいと思います。

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| pytorch | torch.nn.RNN | | Keras | keras.layers.SimpleRNN |
| TensorFlow | tf.keras.layers.SimpleRNN | | Chainer | ? |

答え
- Pytorch [answers/rnn_pytorch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_pytorch.py)
- Tensorflow [answers/rnn_tensorflow_slim.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_tensorflow_slim.py)
- Keras [answers/rnn_keras.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_keras.py)

## Q. RNN (Many-to-one) Step.2. テスト

上で学習させたRNNから実際にコントを実現させましょう。

RNNへの最初の入力はin_text='@@@@@@@@@@'です。学習時にテキストの最初に@をつけたのはこのためです。出力はsoftmaxによる確率分布になっているので、np.random.choice()などを使って、確率分布からランダムに1つのインデックスを選びましょう。その選ばれたインデックスが予測された１文字です。
そしたらin_textの後ろに付けましょう。次のRNNへの入力はin_textの後ろから10文字です。以上をまとめると次のアルゴリズムになります。

1. in_text='@@@@@@@@@'をRNNへ入力
2. RNNの出力の確率分布からランダムに1個選ぶ
3. in_textの後ろに付け足す
4. in_text[-10:]をRNNへ入力し、2-4を繰り返す
5. @が出力される、もしくは予測の回数が1000回以上になったら終了する

解答例の一部
```bash
ろのつはんなだよ。なんなじしねねなんだじゃねーの！なかです。いってょっただろ。あったばななしかい。
おれいかなしゃうっきゃた。なんぇいく。ふりー。じゃねーだぞ。
おれしいーろしですしぇたしおれがいの、おかいなーみうしはい。！ほたてつくちかでとかんだろ。１かったおつらのつかるかっただじゃねーか。なんのは。。
はいはしかかんだろ！よ。
ではしか２たんかいのはんだなるのおはと１とれしは２おのたこれしうな。
つおでつっるわるりまだよ、ついんでうかな。あるるしかねたで。
おい、なうってたたなんのつうししかたいくのほう、おれごいりぇいりかたりつ？
１いらねーかー、！１ばいまったおだよ。ななんでこか？
```

答え
- Pytorch [answers/rnn_pytorch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_pytorch.py)
- Tensorflow [answers/rnn_tensorflow_slim.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_tensorflow_slim.py)
- Keras [answers/rnn_keras.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/rnn_keras.py)

## Q. LSTM (Many-to-one)

RNNをLSTMに変えてみましょう。

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| pytorch | torch.nn.LSTM | | Keras | keras.layers.LSTM |
| TensorFlow | tf.keras.layers.LSTM | | Chainer | chainer.links.LSTM |

解答例の一部
```bash
３０えんのおかえしです。
６００えんじゃねーかよ、じゃあ。なんで３０えんおおくとるんだよ。
あと、これ５００えんいじょうのおかいあげのかたにさしあげてるんですけど。
あ。なんかもらえるの？
れしーとです。
れしーとじゃねーか。ぜんいんにわたせ、これよ。
おはしは２ぜんでよろしかったですか？っていったじゃん。
はい。
おれ、そのいいかただいっきらいなんだよ。
いじ。こちくちってるかったゃ、よかきいす。
```

答え
- Pytorch [answers/lstm_pytorch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_pytorch.py)
- Tensorflow [answers/lstm_tensorflow_slim.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_tensorflow_slim.py)
- Keras [answers/lstm_keras.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_keras.py)
- Chainer [answers/lstm_chainer.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_chainer.py)

## Q. Bi-directional LSTM (Many-to-one)

LSTMをBi-directionalにしましょう。

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| pytorch | torch.nn.LSTM(bidirectional=True) | | Keras | keras.layers.Bidirectional |
| TensorFlow | tf.keras.layers.Bidirectional | | Chainer | chainer.links.LSTM |

解答例の一部
```bash
３０えんのおかえしです。
６００えんじゃねーかよ、じゃあ。なんで３０えんおおくとるんだよ。
あと、これ５００えんいじょうのおかいあげのかたににちゃあたんだいれいれてにいいのでい。
なにゅうなんで。かんなんのわかんだなにどめにがりーせろんがはしたでしきかんぜんおめしあたたよ。

いいき。
びっぐばーがーせっとがおひとつ。おのみものばななしぇいくでよろしかったですか？
```

答え
- Pytorch [answers/lstm_pytorch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_pytorch.py)
- Tensorflow [answers/lstm_tensorflow_slim.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_tensorflow_slim.py)
- Keras [answers/bdlstm_keras.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/lstm_keras.py)
- Chainer [answers/bdlstm_chainer.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_nlp/answers/bdlstm_chainer.py)
