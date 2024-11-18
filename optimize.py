import random
import torch
import os
from time import time
import argparse
import numpy as np
import h5py
import logging

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from optimizer.optimizer_icaf import ANCSHOptimizer

log = logging.getLogger('optimize')


def parse_args():
    parser = argparse.ArgumentParser('Opitimize')
    parser.add_argument('--category', type=str, default='laptop', help='category of articulation')
    parser.add_argument('--num_part', type=int, default=2, help='number of articulation part')
    parser.add_argument('--process_num', type=int, default=64, help='process_num')
    parser.add_argument('--dataset', type=str, default='ArtImage', help='the type of dataset')

    return parser.parse_args()

def main():
    args = parse_args()
    cat = args.category
    process_num = args.process_num
    dataset = args.dataset
    ancsh_results_path = f'/home/xuwenbo/code/ICAF-4/log/{dataset}_ICAF-4_log/ICAF_{cat}/results/results.h5'
    npcs_results_path =  f'/home/xuwenbo/code/ICAF-4/log/{dataset}_ICAF-4_log/ICAF_{cat}/results/results.h5'
    output_dir = f'/home/xuwenbo/code/ICAF-4/log/{dataset}_ICAF-4_log/ICAF_{cat}/refined_results_naocs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_parts = args.num_part
    log.info(f'Instances in dataset have {num_parts} parts')
    #  ancsh_results_path, npcs_results_path, num_parts, niter, choose_threshold, output_dir, optimization_result_path
    optimizer = ANCSHOptimizer(ancsh_results_path, npcs_results_path, num_parts=num_parts, niter=200, choose_threshold=0.1, output_dir=output_dir, optimization_result_path='refined_results.h5')
    optimizer.optimize(process_num=process_num, do_eval=True)
    optimizer.print_and_save()


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    log.info(str(stop - start) + " seconds")
