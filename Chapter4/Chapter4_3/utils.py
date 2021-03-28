import string
import pandas as pd


# データセットのフィルタリング
# データを日本語のみにする（一定のアルファベット数を超えたら英文と評価し、取り除く）
def filter_by_ascii_rate(text, threshold=0.9):
    ascii_letters = set(string.printable)
    rate = sum(c in ascii_letters for c in text) / len(text)
    return rate <= threshold


def load_dataset(filename, n=5000, state=6):
    df = pd.read_csv(filename, sep='\t')

    # extracts Japanese texts.
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]

    # sampling.
    # データが大量のため、いくつか抽出する
    df = df.sample(frac=1, random_state=state)  # shuffle（frac=1←取り出すデータセットの割合=100%）
    grouped = df.groupby('star_rating')  # 各ラベルでグルーピング
    df = grouped.head(n=n)  # 均等にn件サンプリング
    return df.review_body.values, df.star_rating.values


# モデルの学習と評価を行う
# 入力：レビューのテキスト（学習用データと評価用データ）
# 出力：レビューの星
# モデル：ロジスティック回帰
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_eval(x_train, y_train, x_test, y_test,
                   lowercase=False, tokenize=None, preprocessor=None):
    vectorizer = CountVectorizer(lowercase=lowercase,
                                 tokenizer=tokenize,
                                 preprocessor=preprocessor)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_vec, y_train)
    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_pred)
    print('{:.4f}'.format(score))
