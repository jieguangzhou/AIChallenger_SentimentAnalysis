import argparse
import pandas as pd
import numpy as np
import os
from hah_classification.utils import load_opt
from hah_classification.opt import add_train_opt
from hah_classification.main import create_model
from hah_classification.data import load_muti_label_data, batch_iter, COLUMNS


def main(opt):
    base_dataset = load_muti_label_data(opt.test_data, opt.vocab_path)
    result_data = pd.read_csv(opt.test_data)
    _, model = create_model(opt, inference=True)
    model.to_inference(os.path.join(opt.save_path))
    predict_labels = None
    for sequences, labels, lengths in batch_iter(*base_dataset,
                                                 batch_size=opt.batch_size,
                                                 reverse=opt.reverse,
                                                 cut_length=opt.cut_length,
                                                 shuffle=False):
        predict_label, _ = model.inference(sequences, lengths)
        if predict_labels is None:
            predict_labels = predict_label
        else:
            predict_labels = np.concatenate([predict_labels, predict_label])

    result_data[COLUMNS] = predict_labels - 2
    model.session.close()
    result_data.to_csv(os.path.join(opt.save_path, 'test_data_predict_out.csv'), encoding="utf_8", index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_opt(parser)
    save_path = parser.parse_args().save_path
    opt = load_opt(save_path)
    opt.test_data = 'data/sentiment_analysis_testa.csv'
    main(opt)