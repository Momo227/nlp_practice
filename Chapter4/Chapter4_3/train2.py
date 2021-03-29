from sklearn.model_selection import train_test_split

from preprocessing import clean_html, normalize_number, tokenize, tokenize_base_form
from utils2 import load_dataset, train_and_eval


def main():
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=1000)

    # データセットを学習用（80%）,テスト用（20%）に分割
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # 単語分割
    print('Tokenization only.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize)

    # HTMLタグの除去　単語分割より少し性能が上がる
    print('Clean html.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, preprocessor=clean_html)

    # 数字の正規化
    # 性能は若干下がる → 予測したい数字がテキスト内にある場合があるため
    print('Normalize number.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, preprocessor=normalize_number)

    # 単語の原型化
    # 性能は少し下がる → （重要度）単語数の削除 < 活用形の情報
    print('Base form.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize_base_form)

    # 小文字化 性能は低下
    print('Lower text.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, lowercase=True)


if __name__ == '__main__':
    main()
