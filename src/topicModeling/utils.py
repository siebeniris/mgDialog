import numpy as np
import os
import torch

from src.utils.read_files import load_keywords_by_lang


def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k, :].argsort()[-topk:][::-1]
        list_w[k, :] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))
    return TD


def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1:
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1:
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj


def get_topic_coherence(beta, data, vocab):
    D = len(data)  ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp
        TC.append(TC_k)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))
    return TC


def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy()
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors ** 2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors


def visualize(m, save_path, num_topics, num_words, vocab, lang, show_emb=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(lang)
    m.eval()
    logvisual = open(os.path.join(save_path, 'log.txt'), 'a+')
    queries = load_keywords_by_lang(lang)

    # visualize topics using monte carlo
    with torch.no_grad():
        print('#' * 100)
        print('Visualize topics...')
        topics_words = []
        gammas = m.get_beta()
        for k in range(num_topics):
            gamma = gammas[k]
            top_words = list(gamma.cpu().numpy().argsort()[-num_words + 1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topics_words.append(' '.join(topic_words))
            print('Topic {}: {}'.format(k, topic_words))
            logvisual.write('Topic{},{}'.format(k, topic_words))

        if show_emb:
            # visualize word embeddings by using V to get nearest neighbors
            print('#' * 100)
            print('Visualize word embeddings by using output embedding matrix')
            try:
                embeddings = m.rho.weight  # Vocab_size x E
            except:
                embeddings = m.rho  # Vocab_size x E
            neighbors = []
            for word in queries:
                logvisual.write('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, embeddings, vocab)))

                print('word: {} .. neighbors: {}'.format(
                    word, nearest_neighbors(word, embeddings, vocab)))
            print('#' * 100)
    logvisual.write('#' * 100)
    logvisual.close()
