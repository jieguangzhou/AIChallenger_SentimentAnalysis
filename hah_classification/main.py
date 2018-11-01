import tensorflow as tf
import argparse
import os
from sklearn.metrics import f1_score
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from hah_classification.models.model import ClassificationModel
from hah_classification.opt import *
from hah_classification.data import batch_iter, load_muti_label_data
from hah_classification.develop.timer import Timer
from hah_classification.develop.IO import check_path
from hah_classification.utils import save_opt

message = 'step:{0:6}, train loss:{1:6.2}, train accuarcy:{2:7.2%}, val loss :{3:6.2}, val accuary:{4:7.2%}, val f1:{5:6.2}, cost:{6}'


def create_classification_model(class_first,
                                class_second,
                                filter_num,
                                kernel_sizes,
                                skip,
                                layer_num,
                                vocab_size,
                                embedding_size,
                                class_num,
                                learning_rate,
                                keep_drop_prob=1.0,
                                embedding_path=None,
                                inference=False):
    if not inference:
        with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
            train_model = ClassificationModel(class_first,
                                              class_second,
                                              filter_num,
                                              kernel_sizes,
                                              skip,
                                              layer_num,
                                              vocab_size,
                                              embedding_size,
                                              class_num,
                                              learning_rate,
                                              keep_drop_prob=keep_drop_prob,
                                              is_train=True,
                                              embedding_path=embedding_path)

    else:
        train_model = None

    with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
        inference_model = ClassificationModel(class_first,
                                              class_second,
                                              filter_num,
                                              kernel_sizes,
                                              skip,
                                              layer_num,
                                              vocab_size,
                                              embedding_size,
                                              class_num,
                                              learning_rate,
                                              keep_drop_prob=keep_drop_prob,
                                              is_train=False,
                                              embedding_path=None)

    return train_model, inference_model


def get_feed_dict(model, sequences, labels, lengths):
    feed_dict = {model.input_sequences: sequences,
                 model.input_labels: labels,
                 model.input_lengths: lengths}
    return feed_dict


def evaluate(sess, model, dataset, batch_size=64, reverse=False, cut_length=None):
    """评估模型在特定数据集上的loss和accuracy"""
    total_num = len(dataset[0])
    total_loss = 0
    total_accuracy = 0
    correct_labels = None
    predict_labels = None
    for sequences, labels, lengths in batch_iter(*dataset,
                                                 batch_size=batch_size,
                                                 reverse=reverse,
                                                 cut_length=cut_length,
                                                 shuffle=False):
        loss, accuracy, predict_label = sess.run([model.loss, model.accuracy, model.class_labels],
                                                 feed_dict=get_feed_dict(model, sequences, labels, lengths))

        if correct_labels is None:
            correct_labels = labels
            predict_labels = predict_label
        else:
            correct_labels = np.concatenate([correct_labels, labels])
            predict_labels = np.concatenate([predict_labels, predict_label])
        batch_num = len(labels)
        total_loss += batch_num * loss
        total_accuracy += batch_num * accuracy

    all_f1 = []
    for correct_label, predict_label in zip(correct_labels.T, predict_labels.T):
        f1 = f1_score(correct_label, predict_label, average='macro')
        all_f1.append(f1)

    mean_f1 = sum(all_f1) / len(all_f1)

    return total_loss / total_num, total_accuracy / total_num, mean_f1


def create_model(opt, inference=False):
    train_model, inference_model = create_classification_model(opt.class_first,
                                                               opt.class_second,
                                                               opt.filter_num,
                                                               opt.kernel_sizes,
                                                               opt.skip,
                                                               opt.layer_num,
                                                               opt.vocab_size,
                                                               opt.embedding_size,
                                                               opt.class_num,
                                                               opt.learning_rate,
                                                               keep_drop_prob=opt.keep_drop_prob,
                                                               embedding_path=opt.embedding_path,
                                                               inference=inference)

    if not inference:
        inference_model.print_parms()

    return train_model, inference_model


def train(opt):
    save_path = os.path.join(opt.save_path)
    check_path(save_path, create=True)

    print('create model')
    train_model, inference_model = create_model(opt)

    save_opt(opt, save_path)

    # 读取train集和val集,并给予训练集合创建词典
    print('load data set')
    train_dataset = load_muti_label_data(opt.train_data, opt.vocab_path,
                                              create_vocab=True, vocab_size=opt.vocab_size)
    val_dataset = load_muti_label_data(opt.val_data, opt.vocab_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=1)
    timer = Timer()
    best_f1 = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt:
            print('load model from : %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        tensorboard_path = os.path.join(save_path, 'tensorborad')
        summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        for epoch in range(opt.epoch_num):
            epoch_key = 'Epoch: %s' % (epoch + 1)
            print(epoch_key)
            timer.mark(epoch_key)
            total_loss, total_accuracy, total_num = 0, 0, 0
            train_batch_data = batch_iter(*train_dataset,
                                          batch_size=opt.batch_size,
                                          reverse=opt.reverse,
                                          cut_length=opt.cut_length)
            for sequences, labels, lengths in train_batch_data:
                loss, accuracy, global_step, _ = sess.run(
                    [train_model.loss, train_model.accuracy, train_model.global_step, train_model.optimize],
                    feed_dict=get_feed_dict(train_model, sequences, labels, lengths))
                batch_num = len(labels)
                total_num += batch_num
                total_loss += batch_num * loss
                total_accuracy += batch_num * accuracy

                if global_step % opt.print_every_step == 0:
                    train_loss = total_loss / total_num
                    train_accuary = total_accuracy / total_num

                    val_loss, val_accuary, f1 = evaluate(sess, inference_model, val_dataset,
                                                         batch_size=opt.batch_size,
                                                         reverse=opt.reverse,
                                                         cut_length=opt.cut_length)
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                        tf.Summary.Value(tag='train_accuary', simple_value=train_accuary),
                        tf.Summary.Value(tag='val_loss', simple_value=val_loss),
                        tf.Summary.Value(tag='val_accuary', simple_value=val_accuary),
                        tf.Summary.Value(tag='val_f1', simple_value=f1),
                    ])

                    summary_writer.add_summary(summary, global_step)
                    cost_time = timer.cost_time()

                    print(message.format(global_step, train_loss, train_accuary, val_loss, val_accuary, f1,
                                         cost_time))
                    if f1 > best_f1:
                        best_f1 = f1
                        saver.save(sess, os.path.join(save_path, inference_model.name), global_step=global_step)
                    total_loss, total_accuracy, total_num = 0, 0, 0

        val_loss, val_accuary, val_f1 = evaluate(sess, inference_model, val_dataset,
                                             batch_size=opt.batch_size,
                                             reverse=opt.reverse,
                                             cut_length=opt.cut_length)

        if val_f1 > best_f1:
            saver.save(sess, os.path.join(save_path, inference_model.name))

        cost_time = timer.cost_time()


        print('val accuary:{0:7.2%}, val f1:{1:6.2}, cost:{2}'.format(val_accuary, val_f1, cost_time))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_nn_opt(parser)
    add_cnn_opt(parser)
    add_train_opt(parser)
    add_data_opt(parser)
    opt = parser.parse_args()
    train(opt)
