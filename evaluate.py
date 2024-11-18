import logging
from time import time
import argparse
import os


from evaluator.evaluator_icaf import ICAFEvaluator

log = logging.getLogger('evaluator')


def parse_args():
    parser = argparse.ArgumentParser('Opitimize')
    parser.add_argument('--category', type=str, default='laptop', help='category of articulation')
    parser.add_argument('--num_part', type=int, default=2, help='number of articulation part')
    parser.add_argument('--dataset', type=str, default='Art', help='the type of dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    cat = args.category
    dataset = args.dataset

    optimization_result_path = f'/home/xuwenbo/code/ICAF-4/log/{dataset}_log/ancsh_log_{cat}/refined_results/refined_results.h5'
    combined_result_path = optimization_result_path
    output_dir = f'/home/xuwenbo/code/ICAF-4/log/{dataset}_log/ancsh_log_{cat}/prediction_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    num_parts = args.num_part
    log.info(f'Instances in dataset have {num_parts} parts')

    evaluator = ICAFEvaluator(combined_result_path, num_parts=num_parts, thres_r=0.2, output_dir=output_dir, prediction_filename='prediction_results.h5')
    evaluator.process_ICAF(do_eval=True)


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(str(stop - start) + " seconds")
