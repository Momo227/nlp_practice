import re

from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer

t = Tokenizer()


# HTMLのタグの除去
def clean_html(html, strip=False):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=strip)
    return text


# 単語分割
def tokenize(text):
    return t.tokenize(text, wakati=True)


# 原型を得る
def tokenize_base_form(text):
    tokens = [token.tokenize_base_form for token in t.tokenize(text)]
    return tokens


# 数字の正規化用
def normalize_number(text, reduce=False):
    if reduce:
        normalized_text = re.sub(r'\d+', '0', text)
    else:
        normalized_text = re.sub(r'\d', '0', text)
    return normalized_text
