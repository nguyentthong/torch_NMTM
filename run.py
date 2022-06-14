from torch.utils.data import Dataset, DataLoader
import os
import scipy.sparse
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch
from tqdm import tqdm, trange
from model import NMTM
from dataset import TextData
import argparse

parser = argparse.ArgumentParser('Multimodal Neural Topic Model')
parser.add_argument('--topic_num', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--e1', type=int, default=100)
parser.add_argument('--e2', type=int, default=100)
parser.add_argument('--lam', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./data/Amazon_Review')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

def train(config, model, dataset):
    lr = config['lr']
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))

    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    for epoch in range(config['epoch']):
        train_loss = []
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data_en, batch_data_cn = batch
            batch_data_en = batch_data_en.cuda()
            batch_data_cn = batch_data_cn.cuda()
            z_cn, z_mean_cn, z_log_sigma_sq_cn, z_en, z_mean_en, z_log_sigma_sq_en, x_recon_cn, x_recon_en = model(batch_data_en, batch_data_cn)

            # get_loss
            loss_cn = model.get_loss(batch_data_cn, x_recon_cn, z_mean_cn, z_log_sigma_sq_cn)
            loss_en = model.get_loss(batch_data_en, x_recon_en, z_mean_en, z_log_sigma_sq_en)
            loss = loss_cn + loss_en
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        print('Epoch {} \t Loss: {}'.format(epoch, train_loss))
    return model

def print_top_words(config, beta, id2word, lang, n_top_words=15):
    top_words = []
    for i in range(len(beta)):
        top_words.append(" ".join([id2word[j] for j in beta[i].argsort().tolist()[:-n_top_words-1:-1]]))
    
    with open(os.path.join(config['output_dir'], 'top_words_T{}_K{}_{}'.format(n_top_words, config['topic_num'], lang)), 'w') as f:
        for line in top_words:
            f.write(line + '\n')
            print(line)

def export_beta(config, model, data):
    beta_en, beta_cn = model.beta_en, model.beta_cn
    print_top_words(config, beta_en, data.id2word_en, lang='en')
    print_top_words(config, beta_cn, data.id2word_cn, lang='cn')


def evaluate(config, model, dataset):
    export_beta(config, model, dataset)


def main():
    trainDocSet = TextData(args.data_dir,'train')
    testDocSet = TextData(args.data_dir,'test')

    config = dict()
    config.update(vars(args))
    config['vocab_size_cn'] = trainDocSet.vocab_size_cn
    config['vocab_size_en'] = trainDocSet.vocab_size_en

    model = NMTM(config, trainDocSet.Map_en2cn, trainDocSet.Map_cn2en).cuda()
    print('Finished initializing the topic model, embark on the training phase')
    model = train(config, model, trainDocSet)
    print('Finished training the model')
    torch.save(model, os.path.join(args.output_dir,'model.pt'))
    # model = torch.load(os.path.join(args.output_dir,'model.pt'))

    evaluate(config, model, trainDocSet)

if __name__ == '__main__':
    main()