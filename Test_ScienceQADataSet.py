from util.datasets import ScienceQADataSet
import os
import argparse
import datetime
import json
import time
import numpy as np
from pathlib import Path

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--llama_model_path', default='data/weights/', type=str,
                        help='path of llama model')
    parser.add_argument('--max_seq_len', type=int, default=128, metavar='LENGTH',
                        help='the maximum sequence length')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    return parser

def main(args):
    dataset_train = ScienceQADataSet(args, 'train', args.llama_model_path, args.max_seq_len)
    for example, labels, example_mask, image,indicator in dataset_train:
        print(example, labels, example_mask, image,indicator)

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)

