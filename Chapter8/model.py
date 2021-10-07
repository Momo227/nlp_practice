from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dot, Flatten, Embedding, Dense

class EmbeddingModel:

    def __init__(self, vocab_size, emb_dim=100):

        # 入力語、周辺語の入力＋ベクトル化
        self.word_input = Input(shape=(1,), name='word_input')
        self.word_embed = Embedding(input_dim=vocab_size,
                                    output_dim=emb_dim,
                                    input_length=1,
                                    name='word_embedding')

        self.context_input = Input(shape=(1,), name='contect_input')
        self.context_embed = Embedding(input_dim=vocab_size,
                                       output_dim=emb_dim,
                                       input_length=1,
                                       name='context_embedding')

        # Dot層, 1次元化, Dence層
        self.dot = Dot(axes=2)
        self.flatten = Flatten()
        self.output = Dense(1, activation='sigmoid')

    def bulid(self):
        word_embed = self.word_embed(self.word_input)
        context_embed = self.context_embed(self.context_input)
        dot = self.dot([word_embed, context_embed])
        flatten = self.flatten(dot)
        output = self.output(flatten)
        model = Model(inputs=[self.word_input, self.context_input],
                      outputs=output)
        return model