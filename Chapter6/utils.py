import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
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

    # Converts multi-class to binary-class.
    mapping = {1: 0, 2: 0, 4: 1, 5: 1}  # mappingの定義
    df = df[df.star_rating != 3]  # 評価3のデータを削除
    df.star_rating = df.star_rating.map(mapping)  # ラベルの変換

    # extracts Japanese texts.
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]

    # sampling.
    # データが大量のため、いくつか抽出する
    df = df.sample(frac=1, random_state=state)  # shuffle（frac=1←取り出すデータセットの割合=100%）
    grouped = df.groupby('star_rating')  # 各ラベルでグルーピング
    df = grouped.head(n=n)  # 均等にn件サンプリング
    return df.review_body.values, df.star_rating.values


# 学習データセットのサイズを変更したときの学習とテストのスコアを交差検証して算出する。
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')

    plt.legend(loc='best')

    plt.show()
