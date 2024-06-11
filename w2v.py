from typing import Dict

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec

import numpy as np
import re
import urllib.request
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

# from gensim.models import KeyedVectors
# from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords, preprocess_string, preprocess_string

models = {
    "w2v": Word2Vec.load("~/w2v/w2v.model"),
    "tfidf": TfidfModel.load("~/w2v/tfidf.model"),
    "dct": Dictionary.load("~/w2v/dct.dict"),
}

def get_tfidf_weights(models, sentence) -> Dict:
    transformed = models["dct"].doc2bow(sentence.split())
    tfidf_dct = {
        models["dct"][key]: value for (key, value) in models["tfidf"][transformed]
    }

    return tfidf_dct


def embed_sentence(models, sentence):
    """
    sentence: the sentence that we want to embed
    w2v_embedding: average of all word2vec embeddings of all words from sentence
    tfidf_embedding: weighted average of all word2vec embeddings of all words from corresponding to the word
    """
    # print(model.wv['i'])
    # print(model.wv['i'].shape)
    words = sentence.split(" ")
    length = len(words)
    w2v = models["w2v"]
    w2v_embeddings = np.zeros((100, ))

    # {'a': 0.17150865274544777, 'i': 0.2103360568744081, 'have': 0.3343178291040674, 'dream': 0.9025381511909127}
    tfidf_weights = get_tfidf_weights(models, sentence)
    tfidf_w_list = []

    for idx, word in enumerate(words):
        try:
            w2v_word = w2v.wv[word]
            tfidf_w_list.append(tfidf_weights[word])
        except:
            # if there is a word in the given sentence that doesn't exist in the w2v vocabulary
            # create a vector of zeros to compute embeddings and set tfidf weight to 0
            w2v_word = np.zeros((100, ))
            tfidf_w_list.append(0)

        if idx == 0:
            w2v_mat = w2v.wv[word]
        else:
            w2v_mat = np.vstack([w2v_mat, w2v_word])

        # print(f"word: {word}, tfidf_val: {tfidf_weights['word']}")

        w2v_embeddings += w2v_word # w2v

    w2v_mat = np.transpose(w2v_mat)
    tfidf_w_list = np.array(tfidf_w_list)

    tfidf_embeddings = np.dot(w2v_mat, tfidf_w_list) # tf-idf
    w2v_embeddings /= length # average

    return {"w2v_embedding": w2v_embeddings, "tfidf_embedding": tfidf_embeddings}


def create_dictionary(docs):
    dictionary = Dictionary(docs)
    dictionary.save('dct.dict')

    return dictionary, docs


def preliminaries():
    # download dataset
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

    # load and preprocess dataset
    targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
    target_text = etree.parse(targetXML)

    # xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
    parse_text = '\n'.join(target_text.xpath('//content/text()'))

    # 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
    # 해당 코드는 괄호로 구성된 내용을 제거.
    content_text = re.sub(r'\([^)]*\)', '', parse_text)

    # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
    sent_text = sent_tokenize(content_text)

    # 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
    normalized_text = []
    for string in sent_text:
        tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
        normalized_text.append(tokens)

    # 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
    # corpus = [['here', 'are', 'two', 'reasons', ...], ['to', ...], ...]
    corpus = [word_tokenize(sentence) for sentence in normalized_text]
    print('총 샘플의 개수 : {}'.format(len(corpus)))

    return corpus


def train(corpus) -> None:
    # save dictionary
    # dct = Dictionary.load("dct.dict")
    dictionary, docs = create_dictionary(corpus)
    # bow_corpus = [[(0, 1), (1, 1), (2, 2), ... (21, 1), ...], [(13, 2), ...], ...]
    bow_corpus = [dictionary.doc2bow(text) for text in docs]

    # train Word2vec
    print(f"Begin training: Word2Vec")
    model = Word2Vec(sentences=docs, vector_size=100, window=5, min_count=5, workers=4, sg=0) # 0: CBOW, 1: Skip-gram

    # save Word2vec
    model.save("w2v.model")

    # train TF-IDF
    print("Begin training: TF-IDF")
    model = TfidfModel(bow_corpus)

    # save TF-IDF
    model.save("tfidf.model")

    print("done.")


def main():
    # 1. download dataset and extract corpus
    # corpus = preliminaries()

    # 2. train w2v and tf-idf
    # train(corpus)

    sentence = "i have a dream asdkasjdlja"
    result = embed_sentence(models, sentence)

    print(result)


if __name__ == "__main__":
    main()