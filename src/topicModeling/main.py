# /usr/bin/python

from __future__ import print_function

import argparse

import pandas as pd
import torch
import numpy as np
import os
import math
import shutil

from torch import optim
from gensim.models import KeyedVectors

from src.topicModeling import data
from src.topicModeling.etm import ETM
from src.topicModeling.utils import nearest_neighbors, get_topic_coherence, get_topic_diversity, visualize

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

# data and file related arguments
parser.add_argument('--lang', type=str, default="en", help="The language of the data")
parser.add_argument('--dataset', type=str, default='eu', help='The name of corpus, either eu or un')
parser.add_argument('--month', type=str, default="2014-01", help="The month of language data/tp/eu/cs/2014-01/")

parser.add_argument('--batch_size', type=int, default=256, help='Input batch size for training')
# model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')

parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu',
                    help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')

# optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=256, help='input batch size for evaluation')

parser.add_argument('--load_from', type=str, default='output/tp/eu/en/2014-01/',
                    help='the name of the ckpt to eval from')

parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

#########################
data_path = f"data/tp/{args.dataset}/{args.lang}/{args.month}"
emb_path = os.path.join(data_path, f'embeddings.wordvectors')
save_path_dataset = f"output/tp/{args.dataset}"
if not os.path.exists(save_path_dataset):
    os.mkdir(save_path_dataset)

save_path_lang = os.path.join(save_path_dataset, args.lang)
if not os.path.exists(save_path_lang):
    os.mkdir(save_path_lang)

save_path = f"output/tp/{args.dataset}/{args.lang}/{args.month}"
if not os.path.exists(save_path):
    os.mkdir(save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device :', device)
print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# get data
# 1. vocabulary
vocab, train, valid, test = data.get_data(data_path)
vocab_size = len(vocab)
args.vocab_size = vocab_size

# 1. training data
train_tokens = train['tokens']
train_counts = train['counts']
args.num_docs_train = len(train_tokens)
print(f'retrieving train file, length {args.num_docs_train}')

# 2. val data
valid_tokens = valid['tokens']
valid_counts = valid['counts']
args.num_docs_valid = len(valid_tokens)
print(f'retrieving val file, length {args.num_docs_valid}')

# 3. test data
test_tokens = test['tokens']
test_counts = test['counts']
args.num_docs_test = len(test_tokens)
test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
args.num_docs_test_1 = len(test_1_tokens)
test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
args.num_docs_test_2 = len(test_2_tokens)

print(f'retrieving test file, length {args.num_docs_test}')
print(f'retrieving test1 file, length {args.num_docs_test_1}')
print(f'retrieving test2 file, length {args.num_docs_test_2}')

# load existing embeddings..
embeddings = None
if not args.train_embeddings:
    wv = KeyedVectors.load(emb_path)
    vectors = {}
    for word in vocab:
        vectors[word] = wv[word]

    embeddings = np.zeros((vocab_size, args.emb_size))
    words_found = 0
    for i, word in enumerate(vocab):
        try:
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size,))
    embeddings = torch.from_numpy(embeddings).to(device)
    args.embeddings_dim = embeddings.size()

print('=*' * 100)
print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*' * 100)

# define checkpoint
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.mode == 'eval':
    # to set up at the beginning
    ckpt = args.load_from
else:
    ckpt = os.path.join(save_path,
                        'etm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'
                        .format(
                            args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip,
                            args.theta_act,
                            args.lr, args.batch_size, args.rho_size, args.train_embeddings))
    print(f"ckpt : {ckpt}")

# define model and optimizer
model = ETM(args.num_topics, vocab_size, args.t_hidden_size, args.rho_size, args.emb_size,
            args.theta_act, embeddings, args.train_embeddings, args.enc_drop).to(device)


if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, args.batch_size)
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch
        recon_loss, kld_theta = model(data_batch, normalized_data_batch)
        total_loss = recon_loss + kld_theta
        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2)
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
            cur_real_loss = round(cur_loss + cur_kl_theta, 2)

            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

    cur_loss = round(acc_loss / cnt, 2)
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*' * 100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
        epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    print('*' * 100)

    val_loss = evaluate(model, 'val')
    return val_loss


def evaluate(m, source, tc=False, td=False):
    """Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        if source == 'val':
            val_acc_loss = 0
            val_acc_kl_theta_loss = 0
            val_cnt = 0

            print('evaluating validation dataset ...')
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            val_tokens = valid_tokens
            val_counts = valid_counts

            print(f' validation tokens length {len(val_tokens)}')
            for idx, ind in enumerate(indices):
                optimizer.zero_grad()
                m.zero_grad()
                val_data_batch = data.get_batch(val_tokens, val_counts, ind, args.vocab_size, device)
                sums = val_data_batch.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_val_data_batch = val_data_batch / sums
                else:
                    normalized_val_data_batch = val_data_batch

                val_recon_loss, val_kld_theta = m(val_data_batch, normalized_val_data_batch)
                val_acc_loss += torch.sum(val_recon_loss).item()
                val_acc_kl_theta_loss += torch.sum(val_kld_theta).item()
                val_cnt += 1
                val_total_loss = val_recon_loss + val_kld_theta

            val_cur_loss = round(val_acc_loss / val_cnt, 2)
            val_cur_kl_theta = round(val_acc_kl_theta_loss / val_cnt, 2)
            val_cur_real_loss = round(val_cur_loss + val_cur_kl_theta, 2)
            print('*' * 100)
            print('VALIDATION .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                optimizer.param_groups[0]['lr'], val_cur_kl_theta, val_cur_loss,
                val_cur_real_loss))
            print('*' * 100)
            val_loss = round(np.exp(val_cur_real_loss), 1)
            print(f'val_loss {val_loss}')
            return val_loss

        else:
            print('evaluating test dataset ...')
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)
            tokens = test_tokens
            counts = test_counts
            print(f' test tokens length {len(tokens)}')

            # get \beta here
            beta = m.get_beta()

            # do dc and tc here
            acc_loss = 0
            cnt = 0

            indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), args.eval_batch_size)
            for idx, ind in enumerate(indices_1):
                # get theta from first half of docs
                data_batch_1 = data.get_batch(test_1_tokens, test_1_counts, ind, args.vocab_size, device)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = m.get_theta(normalized_data_batch_1)

                # get prediction loss using second half
                data_batch_2 = data.get_batch(test_2_tokens, test_2_counts, ind, args.vocab_size, device)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                # loss function
                recon_loss = -(preds * data_batch_2).sum(1)

                loss = recon_loss / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1

            cur_loss = acc_loss / cnt
            ppl_dc = round(np.exp(cur_loss), 1)
            print('*' * 100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*' * 100)
            if tc or td:
                beta = beta.data.cpu().numpy()
                if tc:
                    print('Computing topic coherence...')
                    tc = get_topic_coherence(beta, train_tokens, vocab)
                if td:
                    print('Computing topic diversity...')
                    td = get_topic_diversity(beta, 25)
                tq = tc*td

                return ppl_dc, tc, td, tq
            else:
                return ppl_dc


if args.mode == 'train':
    # train model on data
    best_epoch = 0
    best_val_loss = 1e50
    all_val_loss = []

    for epoch in range(1, args.epochs):
        val_loss = train(epoch)
        ###

        # val_loss = evaluate(model, 'val')
        if val_loss < best_val_loss:
            filename = ckpt + '_val_loss_' + str(val_loss) + '_epoch_' + str(epoch)
            print(f"filename {filename}")
            with open(filename, 'wb') as f:
                print('saving BEST model to', filename)
                torch.save(model, f)
            best_epoch = epoch
            best_val_loss = val_loss
        else:
            # check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (
                    len(all_val_loss) > args.nonmono and val_loss > min(all_val_loss[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        if epoch % args.visualize_every == 0:
           visualize(model, save_path, args.num_topics, args.num_words, vocab, args.lang)
        all_val_loss.append(val_loss)

    min_val_loss = min(all_val_loss)

    files = [x for x in os.listdir(save_path)]
    files_dict = {int(file.split('_')[-1]): file for file in files if file.startswith(f"etm_tweets_K_{args.num_topics}")
                  }
    print(files_dict)
    max_id = sorted(files_dict)[-1]
    filepath = files_dict[max_id]
    # other files in the dictionary:
    files_dict.pop(max_id)
    to_delete_files = list(files_dict.values())
    # save the best model
    print(f"best model {filepath} for lang {args.lang}")
    best_model_path = os.path.join(save_path, filepath)
    with open(best_model_path, 'rb') as f:
        model = torch.load(f)

    model = model.to(device)
    ppl_dc, tc, td, tq = evaluate(model, 'test', tc=True, td=True)

    # save the best model and delete all the other models.
    best_dir = os.path.join(save_path, "best")
    if not os.path.exists(best_dir):
        os.mkdir(best_dir)
    print(f"moving the best model to {best_dir}")
    # if the file doesn't exist.
    if not os.path.exists(os.path.join(best_dir, filepath)):
        print('the file doesnt exist...')
        shutil.move(best_model_path, best_dir)

    print("removing rest of the model files ...")
    for file in to_delete_files:
        filepath = os.path.join(save_path, file)
        os.remove(filepath)

    result_dict = {"K": args.num_topics,
                   "val_loss": min_val_loss,
                   "epoch": max_id,
                   "test_ppl": ppl_dc,
                   "tc": tc,
                   "td": td,
                   "tq": tq}

    # record the results to a csv.
    # keys are columns.
    print("recoding results....")
    result_df = pd.DataFrame([result_dict])
    result_df_file = os.path.join(save_path, "result.csv")
    if not os.path.exists(result_df_file):
        result_df.to_csv(result_df_file, index=False)
    else:
        df = pd.read_csv(result_df_file)
        df_merge = pd.concat([df, result_df])
        df_merge.sort_values(by="K", inplace=True)
        df_merge.to_csv(result_df_file, index=False)
else:
    # get the best model to evaluate
    # test model with test dataset.
    filepath = args.load_from
    print('load file: ', filepath)
    with open(filepath, 'rb') as f:
        model = torch.load(f)
    print('device:', device)
    model = model.to(device)
    model.eval()
    visualize(model, save_path, args.num_topics, args.num_words, vocab, args.lang)
    with torch.no_grad():
        # get document completion perplexities
        test_ppl = evaluate(model, 'test', tc=args.tc, td=args.td)

        # get most used topics
        indices = torch.tensor(range(args.num_docs_train))
        indices = torch.split(indices, args.batch_size)
        thetaAvg = torch.zeros(1, args.num_topics).to(device)
        thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
        cnt = 0
        for idx, ind in enumerate(indices):
            data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _ = model.get_theta(normalized_data_batch)
            thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
            weighed_theta = sums * theta
            thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
            if idx % 100 == 0 and idx > 0:
                print('batch: {}/{}'.format(idx, len(indices)))
        thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
        print('\nThe 20 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:20]))

        # show topics
        beta = model.get_beta()
        topic_indices = list(np.random.choice(args.num_topics, 10))  # 10 random topics
        print('\n')
        for k in range(args.num_topics):  # topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words + 1:][::-1])
            topic_words = [vocab[a] for a in top_words]
            print('Topic {}: {}'.format(k, topic_words))

        # whether to train or fix the embeddings.
        if args.train_embeddings:
            # show etm embeddings
            try:
                rho_etm = model.rho.weight.cpu()
            except:
                rho_etm = model.rho.cpu()
            queries = ['refugee', 'immigrant']

            print('\n')
            print('ETM embeddings...')
            for word in queries:
                print('word: {} .. etm neighbors: {}'.format(word, nearest_neighbors(word, rho_etm, vocab)))
            print('\n')
