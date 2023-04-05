from __future__ import print_function

import argparse
import pickle
import numpy as np
import os
import json
from glob import glob

import matplotlib.pyplot as plt
import scipy.io
from sklearn.manifold import TSNE
from sklearn import cluster
import pandas as pd
import torch
from torch import nn
from numpy import dot
from numpy.linalg import norm

from src.topicModeling import data

from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



parser = argparse.ArgumentParser(description='Get confidence from centroid with ETM....')
parser.add_argument('--lang', type=str, default="de", help="the language code for the ETM model")
parser.add_argument('--query', type=str, default='eu', help='The name of corpus, either eu or un')
parser.add_argument("--pageType", type=str, default="Twitter", help="PageType of the data data/tp/eu/cs_Twitter")
parser.add_argument("--year", type=str, default=None, help="The year of the data data/tp/eu/cs_Twitter_2014")
parser.add_argument('--month', type=str, default=None, help="The month of language data/tp/eu/cs_Twitter_2014-01/")


parser.add_argument('--num_topics', type=int, default=50, help="the number of topics")


# parser.add_argument('--model_path', type=str,
#                     default="output/models/ETM/de/etm_tweets_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_trainEmbeddings_0_val_loss_2.496855906969676e+29_epoch_195",
#                     help="the path of the pretrained ETM model")

parser.add_argument('--batch_size', type=int, default=1000, help="batch size")
parser.add_argument('--num_words', type=int, default=20, help="number of top words per topic")
parser.add_argument("--reshape_centroid", type=bool, default=True, help="reduce the dimension of centroid.")
parser.add_argument("--reshape_theta", type=bool, default=True, help="reduce the dimension of topic.")
# parser.add_argument("--pca", type=bool, default=True, help="use pca rather than linear layer")
parser.add_argument("--dim", type=int, default=100, help="to which dimension the topic and centroid expand/reduce to")
parser.add_argument("--seed", type=int, default=42, help="seed number for reproduction")

args = parser.parse_args()

# define environment
torch.manual_seed(seed=args.seed)
torch.use_deterministic_algorithms(True)

if args.year is not None:
    data_dir = f"data/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}_{args.year}"
elif args.month is not None:
    data_dir = f"data/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}_{args.year}-{args.month}"
else:
    data_dir = f"data/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}"


num_topics = args.num_topics
batch_size = args.batch_size

# define device, either cuda or cpu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")


def cosine_similarity(list_1, list_2):
    cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
    return cos_sim


def load_keywords(query=args.query, lang=args.lang):
    with open(f"data/keywords/{query}/{lang}.json") as f:
        keywords = json.load(f)
    return keywords



def load_data_and_vocab(input_dir=data_dir):
    """
    Load the data from preprocessed data directory for inference of topics using trained ETM
    :param input_dir:
    :return:
    """
    # word ids.
    token_file = os.path.join(input_dir, 'data_tokens.mat')
    count_file = os.path.join(input_dir, 'data_counts.mat')
    tokens_ = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts_ = scipy.io.loadmat(count_file)['counts'].squeeze()
    print(f"tokens shape: {tokens_.shape}, counts shape: {counts_.shape}")

    with open(os.path.join(input_dir, "vocab.pkl"), "rb") as f:
        vocab_ = pickle.load(f)

    word2id_ = {word: idx for idx, word in enumerate(vocab_)}
    id2word_ = {idx: word for idx, word in enumerate(vocab_)}
    return tokens_, counts_, vocab_, word2id_, id2word_


def get_nearest_neighbors(word, vectors, vocab, num_words):
    index = vocab.index(word)
    query = vectors[index]
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors ** 2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors


def get_theta_weights_avg(model, keywords, tokens, counts, vocab, batch_size, device, output_dir):
    """
    Get theta weights
    :param model: trained ETM
    :param keywords: keywords used to crawl tweets in language lang_code
    :param tokens:
    :param counts:
    :param vocab:
    :param batch_size:
    :param device: cuda or cpu
    :return:
    """

    num_docs = len(tokens)
    vocab_size = len(vocab)
    theta_weights_list = []
    topic2words = dict()

    # check whether keywords are in vocab
    keywords_selected = []
    for k in keywords:
        if k in vocab:
            keywords_selected.append(k)
    # for cosine similarity between centroid and each topic distribution of tweets
    print("keywords selected: ", keywords_selected)
    confidences_cos = []

    model.eval()
    with torch.no_grad():
        # get model's embeddings.
        embeddings = model.rho
        embeddings = embeddings.cpu().numpy()

        # get centroid and the similar words of centroid and the vectors
        indexes_keywords = [vocab.index(word) for word in keywords_selected]
        # get the embeddings of the keywords.
        queries = [embeddings[index] for index in indexes_keywords]
        print(f'length of the keywords queries {len(queries)}')

        # reduce the dimensionality of embeddings to dim of topic embedding.
        # bad results!
        # if args.pca:
        #     pca = PCA(n_components=args.dim)
        #     queries_dim_reduced = pca.fit_transform(queries)
        #     print(f"queries dim reduced: {queries_dim_reduced.shape}")
        #     kmeans = cluster.KMeans(n_clusters=1, random_state=0)
        #     kmeans.fit(queries_dim_reduced)
        #     centroid = kmeans.cluster_centers_[0]

        # else:
        kmeans = cluster.KMeans(n_clusters=1, random_state=0)
        # kmeans.fit(queries)
        # X_dist = kmeans.transform(queries) ** 2

        kmeans.fit(queries)
        X_dist = kmeans.transform(queries) ** 2
        # the centroid of the keywords embeddings
        centroid = kmeans.cluster_centers_[0]

        centroid = centroid.reshape((1, -1))
        print(centroid.shape)

        # queries_dim_reduced.append(centroid)
        square_distances = X_dist.sum(axis=1).round(2)
        sqrt_distances = np.sqrt(square_distances)
        mean_distances = np.mean(sqrt_distances)
        print('the distances of the centroid with the vectors:', mean_distances)

        print(f'length of the neighbors queries {len(queries)}')
        n = np.append(arr=queries, values=centroid, axis=0)
        n_arr = np.array(n)
        print(f"query shape {n_arr.shape}")

        ks = keywords_selected + ['centroid']
        # +nearest_neighbors+neighbors_keywords_selected
        print(f"keywords {len(ks)}")

        tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
        T = tsne.fit_transform(n_arr)

        colors = ['blue' for x in range(0, len(keywords_selected))] + ['red']
        # ['orange' for x in range(0,len(nearest_neighbors))]
        # +['green' for x in range(0,len(neighbors_keywords_selected))]

        # visualize centroid and keywords.
        plt.figure(figsize=(40, 10))
        plt.scatter(T[:, 0], T[:, 1], s=40, c=colors)

        for label, x, y in zip(ks, T[:, 0], T[:, 1]):
            plt.annotate(label, xy=(x + 2, y + 2), xytext=(0, 0), textcoords='offset points', fontsize=30)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.savefig(os.path.join(output_dir, f'k_{num_topics}_dist_kn_{num_topics}.png'))

        # get topic distributions
        beta = model.get_beta()
        for k in range(0, num_topics):  # topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words:][::-1])
            topic_words = [vocab[a] for a in top_words]
            topic2words[k] = topic_words

        # get most used topics
        indices_ = torch.tensor(range(num_docs))
        indices = torch.split(indices_, batch_size)
        theta_avg = torch.zeros(1, num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, num_topics).to(device)

        print('theta weighted avg shape: ', thetaWeightedAvg.shape)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(tokens, counts, ind, vocab_size, device)
            # print('data_batch:', len(data_batch))
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            normalized_data_batch = data_batch / sums
            theta, _ = model.get_theta(normalized_data_batch)
            theta_avg += theta.sum(0).unsqueeze(0) / num_docs

            weighed_theta = sums * theta
            # print(weighed_theta.shape)  # [1000,50]

            # append weighed theta to list for infer topics later.
            theta_weights_list.append(weighed_theta.cpu().detach().numpy())
            weighed_theta_sum = weighed_theta.sum(0).unsqueeze(0)

            # print('*' * 40)
            thetaWeightedAvg += weighed_theta_sum
            # if idx % 100 == 0 and idx > 0:
            #     print('batch: {}/{}'.format(idx, len(indices)))
        # get topic words
        theta_weights_avg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        # the most used topics in descending order.
        theta_weights_avg_ordered = theta_weights_avg.argsort()[::-1]

    # topic embeddings for the whole data
    theta_weights = np.concatenate(theta_weights_list, axis=0)
    print(theta_weights.shape)

    centroid = centroid.flatten().astype(np.float32)
    print("centroid shape: ", centroid.shape)

    # reshape centroid
    if args.reshape_centroid and not args.reshape_theta:
        centroid = torch.from_numpy(centroid).to(device)
        # reduce the dimensionality of centroid
        torch.manual_seed(args.seed)
        lr = nn.Linear(centroid.shape[0], num_topics).to(device)
        reshaped_centroid = lr(centroid).cpu().detach().numpy()
        print("reshaped centroid: ", reshaped_centroid.shape)

        for i in theta_weights:
            cos = cosine_similarity(reshaped_centroid, i)
            confidences_cos.append(cos)

    if args.reshape_theta and not args.reshape_centroid:
        theta_weights = torch.from_numpy(theta_weights).to(device)
        torch.manual_seed(args.seed)
        lr = nn.Linear(num_topics, centroid.shape[0]).to(device)
        reshaped_theta_weights = lr(theta_weights).cpu().detach().numpy()
        print("reshaped theta weights: ", reshaped_theta_weights.shape)

        for i in theta_weights:
            cos = cosine_similarity(centroid, i)
            confidences_cos.append(cos)

    if args.reshape_centroid and args.reshape_centroid:
        centroid = torch.from_numpy(centroid).to(device)
        theta_weights_ = torch.from_numpy(theta_weights).to(device)
        torch.manual_seed(seed=args.seed)
        lr = nn.Linear(centroid.shape[0], args.dim).to(device)
        torch.manual_seed(seed=args.seed)
        lr2 = nn.Linear(theta_weights_.shape[1], args.dim).to(device)
        reshaped_centroid = lr(centroid).cpu().detach().numpy()
        reshaped_theta_weights = lr2(theta_weights_).cpu().detach().numpy()
        print("reshaped theta weights: ", reshaped_theta_weights.shape)
        print("reshaped centroid: ", reshaped_centroid.shape)
        confidences_cos = np.matmul(reshaped_theta_weights, reshaped_centroid)

    print('sorting theta weights ...')
    theta_weights_sorted = []
    for thetaWeight in theta_weights:
        theta_weights_sorted.append(thetaWeight.argsort()[::-1].tolist())

    return topic2words, theta_weights_avg_ordered, theta_weights, theta_weights_sorted, confidences_cos


def save_to_results(df, tokens, topic2words, theta_weights_avg_ordered, theta_weights_sorted, confidences_cos, id2word,
                    output_dir):
    tokens_list = []
    for sent in tokens.tolist():
        # from numpy array to list.
        tokens_list.append(sent.tolist()[0])

    topics_ = []
    sentences_ = []
    for topic_nrs, inds in zip(theta_weights_sorted, tokens_list):
        top_nr = topic_nrs[0]
        topics_.append(top_nr)
        sent = [id2word[idx] for idx in inds]
        sentences_.append(sent)

    print(f"len {len(sentences_)} df len {len(df)}")
    df['sentence_etm'] = sentences_
    df['topic'] = topics_
    df['cos_sim'] = confidences_cos

    print(f"cos_sim> 0: {len(df[df['cos_sim'] > 0])}")
    df.to_csv(os.path.join(output_dir, f"{args.lang}_etm.csv"), index=False)

    with open(os.path.join(output_dir, "topic_words.json"), 'w') as f:
        json.dump(topic2words, f)

    print(theta_weights_avg_ordered)
    with open(os.path.join(output_dir, "most_freq_topics.pkl"), "wb") as f:
        pickle.dump(theta_weights_avg_ordered, f)


if __name__ == '__main__':
    results_dir = f"output/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}"

    #TODO: LATER USE low/medium/data volumne
    if args.year is not None:
        results_dir = f"output/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}_{args.year}"
    elif args.month is not None:
        results_dir = f"output/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}_{args.year}-{args.month}"
    else:
        results_dir = f"output/tp/{args.query}/{args.lang}/{args.lang}_{args.pageType}"


    output_dir_lang = os.path.join(results_dir, str(args.num_topics))
    if not os.path.exists(output_dir_lang):
        os.mkdir(output_dir_lang)

    keywords = load_keywords(args.query, args.lang)
    print(f"{args.lang} -> {keywords}")

    tokens, counts, vocab, word2id, id2word = load_data_and_vocab(data_dir)
    df_file = os.path.join(data_dir, f"{args.lang}_{args.pageType}_non_empty.csv")

    df = pd.read_csv(df_file)
    model_dir = f"{results_dir}/best"

    model_path = [x for x in glob(f"{model_dir}/etm_{args.query}_K_{args.num_topics}_*")][0]

    print(f"loading the model file : {model_path} ")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.to(device)

    # get the topic of data
    topic2words, theta_weights_avg_ordered, theta_weights, theta_weights_sorted, confidence_cos = get_theta_weights_avg(
        model,
        keywords,
        tokens,
        counts,
        vocab,
        batch_size,
        device,
        output_dir_lang)
    # save results..
    save_to_results(df, tokens, topic2words, theta_weights_avg_ordered, theta_weights_sorted, confidence_cos, id2word,
                    output_dir_lang)