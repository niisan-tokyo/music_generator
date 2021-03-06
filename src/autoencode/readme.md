# Autoencoder + processing data in chronological order

音楽を、オートエンコーダを使って一旦符号化する。この作業は音楽を耳コピして楽譜に落とし込むのに等しい。
オートエンコーダも機械学習によって形成するので、ここで作られる「楽譜」とは、機械が独自に開発した楽譜である。

楽譜ができたら、これらが時系列上のつながりを持っているものと解釈する。つまり、数秒間演奏したら、その次に来る楽譜がある程度決まるというもの。
つまり、「機械が続きの楽譜を創造・作曲する」ことになる。

# Using Autoencoder

音のデータは最近のものだと44.1kHz, すなわち、１秒間に44100サンプルの波形データによって構成されている。
また、ステレオ音源であることを加味すると、実質的なサンプル数は秒間88.2k となる。

これを単純に時系列データとして扱うにはかなりきついことになるので、まず扱う変数を少なくすることを考えるのだが、
最も単純な手段としてオートエンコーダを採用した。

音楽の音符は色々あるが、馴染みのある4分音符で1秒、16分音符で1/4 秒である。
例えば、これらを元に1/4 秒ごとにサンプルをエンコードし、楽譜データとして形成することを考える。

## Create Sample Data

学習に必要なデータは `/data/input` に格納しておく。
ここでまずは学習に使用するデータを作成する。
事前にデータを作っておくのは、学習が足りなかったときなどに学習し直す際、前と全く同じデータでできるようにするためである。

コマンドは以下のとおりである

```
$ python combine_encoded_test_data.py
```

## Train autoencoder

データができたら学習させる。
隠れ層が一層だけの単純なモデルだが、何故かこれが収束性が良いので採用している。
汎用性が必要ならもう少し増やしてもいいかも( モデルが肥大化し、必要なメモリも増えるけど )。

```
$ python autoencode.py
```

入力データ量によってスピードが違うので注意

## Test autoencoder

学習ができたら試しにエンコードが成功しているか確かめる

```
$ python autoencode_test.py
```

ちゃんと音楽になっていれば良いだろう。
覚え込ませる曲が多ければ多いほど、背景ノイズが大きくなるので、なんとか改善していきたい。

# Using score creater

エンコードされた曲は楽譜のようなものである。
これが時間経過とともにどのような変化をするかを学習させることで、曲ができていくということになる...はず。

## Create Score Data for using in learning

すでにエンコーダができているので、学習に使用するデータを作る。

```
$ python create_encoded_data.py
```

ここでは、約10秒ごとの楽譜の状況 ( = 40 steps ) を使用して、1.25秒程度のデータ ( = 5 steps ) を生成することを考えている。

また、今回はステートレスな処理を行うため、データの順番は不要であり、訓練データとテストデータが分割された学習過程で変わっていないことが必要なので、
テストデータを一つにまとめる

```
$ python combine_encoded_test_data.py
```

## Train score creater

時系列データを作成したらあとは学習を開始するだけである。
しかし、時系列データの生成には２つの方法が存在している
出力される結果の形状は同じだが、過程が違い、出力される内容に差異および精度の違いが発生している
現状ではLSTM のほうが強いように思われる

### Conv1D model

一次元畳み込みを利用した時系列分析モデルを使用する方法。
畳み込み -> max pooling を数回繰り返し、次元を下げていく。
例えば3回繰り返すことで、入力の形状が

(40, 512) -> (5, 512)

になる。これは今回の想定される入力と出力のデータ形状と一致している。
これを使った学習は以下のコマンドにより行う。

```
$ python conv1d_creater.py
```

収束性がそれほど良くないが、原因としては、

- 層の数が少ない
- 活性化関数が不適切

などが考えられる。

畳み込みなので、GPUを使って高速化するイメージが付きやすい。

### LSTM

自然言語処理でよく使われている、時系列データの解析手段であるが、ちょっと自分の理解が追いついていない。
今回はstateless な方のLSTMを使用して、楽譜生成器を作る

LSTMの出力次元はシーケンスではなく、パラメータの次元になるので、最後にReshapeを加えて強引に目当ての形式に変換している

```
$ python lstm_creater.py
```

Conv1Dと比べ、少々強引ではあるが、それでも精度はこっちの方が良い
ただし、そもそも

- autoencoder の精度が悪い
- モデルの最適化がすぐに停滞する

という問題があり、そこまでうまくいっているわけではない。

# Create music

種ファイルを使って音楽を作る機械を実装する。

```
$ python music_generator.py
```

動作はすごく簡単で、

- 種音源ファイルを用意する
- オートエンコーダを読み込み、エンコード部分とデコード部分に分離する
- score creater を読み込む
- 種音源から10秒ほど取り込み、エンコードにかける
- エンコードしたデータをscore creater に読ませて、時系列的な続きを予測させる
- 出来上がったscore をデコードする
- wavファイルに出力する
