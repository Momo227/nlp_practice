# データセットの準備
mkdir 'data'

cd data || exit
wget http://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_JP_v1_00.tsv.gz
gunzip amazon_reviews_multilingual_JP_v1_00.tsv.gz
head -5 amazon_reviews_multilingual_JP_v1_00.tsv
# marketplace   customer_id review_id   product_id  product_parent  product_title   product_category    star_rating helpful_votes   total_votes vine    verified_purchase   review_headline review_body review_date
# JP  65317   R33RSUB4TROT7  B000001GBJ  957145596   SONGS FROM A SECRET GARDE   Music   1   1   15  N   Y   残念ながら…  残念ながら…趣味ではありませんでした。ケルト音楽の範疇にも幅があるのですね…  2012-12-05
# JP  65317   R2U1VB8GOBBED  B000YOB2  904244932   鏡の中の鏡‾ペルト作品集(SAD)(Arvo Part:Spiegel im Spiegel)    Music   1   4   20  N   Y   残念ながら…残念ながら…趣味ではありませんでした。正直退屈…眠気も起きない…  2012-12-05
# JP  65696   R1ISOBARIC  B0002E5O9G  108978277   Les Miserables 10th Anniversary Concert Music   5   2   3   N   Y   ドリームキャスト    素晴らしいパフォーマンス。ミュージカル映画版の物足りない歌唱とは違います。   2013-03-02
# JP  67162   RL02CW5CLONE   B00004SRJ5  606528497   It Takes a Nation of Millions to Hold Us Back   Music   5   6   9   N   Y   やっぱりマスト 専門的な事を言わずにお勧めレコメを書きたいのですが、文才が無いので無理でした。ヒップホップがカルチャーとして認められましたが、此のころはまだ差別的な事件もありましたね。そんな時代を象徴するPEのコンセプトアルバムです。この完成度を越すヒップホップのアルバムが何枚あることか・・・ 2013-08-11
# shellcheck disable=SC2103
cd ..

# 交差検証
python cross_validation.py

# 学習曲線
python learning_curve.py

# ハイパーパラメータ
python hp_optimization.py

# 結果
# k分割交差検証
# [0.8175   0.83375  0.8225   0.8275   0.820625]  ：各分割の正解率
# Accuracy:0.8244(+/- 0.0114)  ：平均正解率
# 0.8440  ：分散

# ハイパーパラメータの組み合わせに対するモデルの学習と評価
# {'C': 3, 'penalty': 'l2'}
# Accuracy(best):0.8331
# Accuracy(test):0.8540
