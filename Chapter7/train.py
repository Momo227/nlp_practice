from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from model import create_model
from preprocessing import clean_html, tokenize
from utils import load_dataset, plot_history


def main():
    # データセットの読み込み.
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=5000)

    # データセットの前処理.
    x = [clean_html(text, strip=True) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # データセットのベクトル化.
    vectorizer = CountVectorizer(tokenizer=tokenize)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    # ハイパラメータの設定
    vocab_size = len(vectorizer.vocabulary_)
    label_size = len(set(y_train))

    # モデルの構築
    print(vocab_size, label_size)
    model = create_model(vocab_size, label_size)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # コールバックの準備
    filepath = 'model.h5'
    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(filepath, save_best_only=True),
        TensorBoard(log_dir='logs')
    ]

    # モデルの学習
    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=32,
                        callbacks=callbacks)

    # モデルの読み込み
    model = load_model(filepath)

    # モデルを使った予測
    text = 'このアプリ超最高！'
    vec = vectorizer.transform([text])
    y_pred = model.predict(vec.toarray())
    print(y_pred)

    # 正解率と損失のグラフの描画
    plot_history(history)


if __name__ == '__main__':
    main()
