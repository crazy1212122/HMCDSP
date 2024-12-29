import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run circRDRP.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="/home/swk/code/CircRDRP/dataset",
                        help="Training datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=200,
                        help="Number of training epochs. Default is 651.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=16,#256
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--circRNA-number",
                        type=int,
                        default=1885,
                        help="circRNA number. Default is 585.")

    parser.add_argument("--fcir",
                        type=int,
                        default=128,#128
                        help="circRNA feature dimensions. Default is 128.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=30,
                        help="disease number. Default is 88.")

    parser.add_argument("--fdis",
                        type=int,
                        default=128,#128
                        help="disease feature number. Default is 128.")

    parser.add_argument("--drug-number",
                        type=int,
                        default=27,
                        help="disease number. Default is 88.")

    parser.add_argument("--fdrug",
                        type=int,
                        default=128,
                        help="drug feature number. Default is 128.")
    
    parser.add_argument("--batch",
                        type=int,
                        default=32,
                        help="batch size.Default is 32."
                        )

    parser.add_argument("--fold",
                        type=int,
                        default=5,
                        help="Number of folds for cross-validation. Default is 5.")

    parser.add_argument("--round",
                        type=int,
                        default=10,
                        help="Number of experiment rounds. Default is 10.")
    parser.add_argument('--edge_dim', type=int, default=1, help='Dimension of edge features')
    return parser.parse_args()
