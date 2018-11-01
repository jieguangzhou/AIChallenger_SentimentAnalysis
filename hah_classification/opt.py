def add_cnn_opt(parser):
    group = parser.add_argument_group('cnn')

    group.add_argument('-class_first', type=int, default=6,
                       help="cnn class_first")

    group.add_argument('-class_second', type=int, nargs='+', default=[3, 4, 3, 4, 4, 2],
                       help="cnn class_second")

    group.add_argument('-filter_num', type=int, default=128,
                       help="cnn filter num")

    group.add_argument('-kernel_sizes', type=int, nargs='+', default=5,
                       help="cnn kernel_sizes, a list of int")

    group.add_argument('-skip', type=int, default=5,
                       help="cnn skip")

    group.add_argument('-layer_num', type=int, default=2,
                       help="cnn layer_num")


def add_nn_opt(parser):
    group = parser.add_argument_group('nn')

    group.add_argument('-embedding_size', type=int, default=128,
                       help="embedding size")

    group.add_argument('-vocab_size', type=int, default=8000,
                       help="vocab size")

    group.add_argument('-embedding_path', type=str, default=None,
                       help="embedding path, 暂不使用")

    group.add_argument('-keep_drop_prob', type=float, default=0.5,
                       help="keep_drop_prob")

    group.add_argument('-class_num', type=int, default=4,
                       help="class_num")


def add_train_opt(parser):
    group = parser.add_argument_group('train')

    group.add_argument('-learning_rate', type=float, default=1e-3,
                       help="learning_rate")

    group.add_argument('-batch_size', type=float, default=64,
                       help="batch_size")

    group.add_argument('-epoch_num', type=int, default=5)

    group.add_argument('-print_every_step', type=int, default=500)

    group.add_argument('-save_path', type=str, default='save/model')


def add_data_opt(parser):
    group = parser.add_argument_group('data')
    group.add_argument('-train_data', type=str, default='data/sentiment_analysis_trainingset.csv',
                       help="train data path")

    # group.add_argument('-train_data', type=str, default='data/sentiment_analysis_validationset.csv',
    #                    help="train data path")

    group.add_argument('-val_data', type=str, default='data/sentiment_analysis_validationset.csv',
                       help="val data path")

    group.add_argument('-test_data', type=str, default='data/sentiment_analysis_testa.csv',
                       help="test data path")

    group.add_argument('-vocab_path', type=str, default='data/vocab.txt',
                       help="vocab_pathe")

    group.add_argument('-cut_length', type=int, default=600,
                       help="cut_length")

    group.add_argument('-reverse', action='store_true',
                       help="reverse the sequence")
