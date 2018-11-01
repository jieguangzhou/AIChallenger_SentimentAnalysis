from collections import Counter
import logging
import random
import numpy as np
import jieba
from hah_classification.develop.IO import read_file, write_file
import pandas as pd
import os

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

PAD_IDX = 0
UNK_IDX = 1

COLUMNS = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
           'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
           'price_level', 'price_cost_effective', 'price_discount',
           'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
           'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
           'others_overall_experience', 'others_willing_to_consume_again']


def segment(sentence):
    return [i for i in sentence if i.strip()]


def load_vocab(vocab_path):
    """
    读取词典
    """
    vocab = {token: index for index, token in
             enumerate(read_file(vocab_path, deal_function=lambda x: x.strip() if x != '\n' else x))}
    logger.info('load vocab (size:%s) to  %s' % (len(vocab), vocab_path))
    return vocab


def save_vocab(vocab, vocab_path):
    """
    保存词典
    """
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    write_file(sorted_vocab, vocab_path, deal_function=lambda x: x[0] + '\n')
    logger.info('save vocab (size:%s) to  %s' % (len(vocab), vocab_path))


def load_data(data_path, vocab_path, label_vocab_path, create_vocab=False, create_label_vocab=False, min_freq=1,
              vocab_size=None, return_label_vocab=False):
    msg = 'load data from  %s, ' % data_path
    data_set = pd.read_csv(data_path)
    vocab_ = Counter() if create_vocab else load_vocab(vocab_path)
    label_vocab = {} if create_label_vocab else load_vocab(label_vocab_path)

    sequences, lengths = [], []
    for content in data_set.iloc[:, 1]:
        tokens = segment(content)
        if create_vocab:
            vocab_.update(tokens)
        sequences.append(tokens)
        lengths.append(len(tokens))

    if create_vocab:
        vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        # vocab_size 必须大于2
        print('ori vocab size %s' % len(vocab_))
        vocab_size = max(vocab_size or len(vocab_), 2) - 2
        logger.info('create vocab, min freq: %s, vocab_size: %s' % (min_freq, vocab_size))
        for token, count in vocab_.most_common(vocab_size):
            if not token:
                continue
            if count < min_freq:
                break
            else:
                vocab[token] = len(vocab)
        save_vocab(vocab, vocab_path)
    else:
        vocab = vocab_

    columns = data_set.columns.values.tolist()[2:]
    dict_labels = {}
    dict_label_vocab = {}
    for col in columns:
        labels = [str(i) for i in data_set[col]]
        col_vocab_path = label_vocab_path + '.' + col
        if create_label_vocab:
            label_vocab = {vocab: index for index, vocab in enumerate(sorted(set(labels)))}
            save_vocab(label_vocab, col_vocab_path)
        else:
            label_vocab = load_vocab(col_vocab_path)
        if not return_label_vocab:
            labels = list(map(lambda x: label_vocab[x], labels))
            dict_labels[col] = np.array(labels)
        dict_label_vocab[col] = label_vocab

    if create_label_vocab:
        save_vocab(label_vocab, label_vocab_path)
    sequences = [[vocab.get(token, UNK_IDX) for token in sequence] for sequence in sequences]
    msg += 'total : %s' % len(sequences)
    logger.info(msg)
    if return_label_vocab:
        return np.array(sequences), dict_labels, np.array(lengths), dict_label_vocab
    else:
        return np.array(sequences), dict_labels, np.array(lengths)


def load_muti_label_data(data_path, vocab_path, create_vocab=False,
                         min_freq=1,
                         vocab_size=None):
    msg = 'load data from  %s, ' % data_path
    data_set = pd.read_csv(data_path)
    vocab_ = Counter() if create_vocab else load_vocab(vocab_path)

    sequences, lengths = [], []
    for content in data_set.iloc[:, 1]:
        tokens = segment(content)
        if create_vocab:
            vocab_.update(tokens)
        sequences.append(tokens)
        lengths.append(len(tokens))

    if create_vocab:
        vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        # vocab_size 必须大于2
        print('ori vocab size %s' % len(vocab_))
        vocab_size = max(vocab_size or len(vocab_), 2) - 2
        logger.info('create vocab, min freq: %s, vocab_size: %s' % (min_freq, vocab_size))
        for token, count in vocab_.most_common(vocab_size):
            if not token:
                continue
            if count < min_freq:
                break
            else:
                vocab[token] = len(vocab)
        save_vocab(vocab, vocab_path)
    else:
        vocab = vocab_


    labels = data_set[COLUMNS].values + 2
    sequences = [[vocab.get(token, UNK_IDX) for token in sequence] for sequence in sequences]
    msg += 'total : %s' % len(sequences)
    logger.info(msg)
    return np.array(sequences), labels, np.array(lengths)


def batch_iter(sequences, labels, lengths, batch_size=64, reverse=False, cut_length=None, shuffle=True):
    """
    将数据集分成batch输出
    :param sequences: 文本序列
    :param labels: 类别
    :param lengths: 文本长度
    :param reverse: 是否reverse文本
    :param cut_length: 截断文本
    :return:
    """

    # 打乱数据
    data_num = len(sequences)
    indexs = list(range(len(sequences)))
    if shuffle:
        random.shuffle(indexs)
    batch_start = 0
    shuffle_sequences = sequences[indexs]
    shuffle_labels = labels[indexs]
    shuffle_lengths = lengths[indexs]

    while batch_start < data_num:
        batch_end = batch_start + batch_size
        batch_sequences = shuffle_sequences[batch_start:batch_end]
        batch_labels = shuffle_labels[batch_start:batch_end]
        batch_lengths = shuffle_lengths[batch_start:batch_end]

        if isinstance(cut_length, int):
            # 截断数据
            batch_sequences = [sequence[:cut_length] for sequence in batch_sequences]
            batch_lengths = np.where(batch_lengths > cut_length, cut_length, batch_lengths)

        # padding长度
        batch_max_length = batch_lengths.max()

        batch_padding_sequences = []
        for sequence, length in zip(batch_sequences, batch_lengths):
            sequence += [PAD_IDX] * (batch_max_length - length)
            if reverse:
                sequence.reverse()
            batch_padding_sequences.append(sequence)

        batch_padding_sequences = np.array(batch_padding_sequences)

        yield batch_padding_sequences, batch_labels, batch_lengths
        batch_start = batch_end


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    vocab_path = '../data/vocab.txt'
    label_vocab_path = '../cnews/label.txt'
    data_set = load_data('../data/sentiment_analysis_validationset.csv', vocab_path, label_vocab_path,
                         create_vocab=True, create_label_vocab=True, vocab_size=5000)
    # num = 0
    # for sequences, labels, lengths in batch_iter(*data_set, batch_size=64):
    #     print(sequences.shape[1], lengths.max(), sequences.shape[1] == lengths.max())
