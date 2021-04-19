from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
t = Tokenizer(wakati=True)


# HTMLのタグの除去
def clean_html(html, strip=False):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=strip)
    return text


# 単語分割
def tokenize(text):
    return t.tokenize(text)
