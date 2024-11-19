import os
import h5py
import numpy as np
import multiprocessing
import logging
from time import time

from optimizer.optimize_util import optimize_with_kinematic


def h5_dict(ins):
    result = {}
    for k, v in ins.items():
        result[k] = np.asarray(v[:])
    return result

class ICAFOptimizer:
    def __init__(self, icaf_results_path, npcs_results_path, num_parts, niter, choose_threshold, output_dir, optimization_result_path, do_eval=True):
        start = time()
        self.niter = niter
        self.choose_threshold = choose_threshold
        self.output_dir = output_dir
        self.optimization_result_path = optimization_result_path
        self.num_parts = num_parts
        self.log = logging.getLogger('optimizer')
        self.results = None
        self.log_string("Loading the data from results hdf5 file")
        self.f_icaf = h5py.File(icaf_results_path, "r")
        self.f_npcs = h5py.File(npcs_results_path, "r")
        self.instances = sorted(self.f_icaf.keys())
        self.log_string(f"Load the data: {time() - start} seconds")
        self.do_eval = do_eval

        # test_instances = []
        # for instance in self.instances:
        #     if instance.split("_")[1] == "0042" or instance.split("_")[1] == "0014":
        #         test_instances.append(instance)

        # self.instances = test_instances[:10] + self.instances[-10:]

    def log_string(self, str):
        self.log.info(str)
        print(str)

    def optimize(self, process_num=4, do_eval=True):
        self.do_eval = do_eval
        pool = multiprocessing.Pool(processes=process_num)
        self.log_string(f"runing {self.niter} iterations for ransac")
        process = []
        # start = True
        # This should automatically change the result file
        for ins in self.instances:
            process.append(
                pool.apply_async(
                    optimize_with_kinematic,
                    (
                        ins,
                        h5_dict(self.f_icaf[ins]),
                        h5_dict(self.f_npcs[ins]),
                        self.num_parts,
                        self.niter,
                        self.choose_threshold,
                        self.log,
                        False,
                        do_eval,
                    ),
                )
            )
        pool.close()
        pool.join()

        self.results = [p.get() for p in process]

    def print_and_save(self):
        if self.do_eval:
            # Calculate the mean error for each part
            errs_rotation = []
            errs_translation = []
            errs_naocs_rotation = []
            errs_naocs_translation = []
        valid_num = 0
        for result in self.results:
            if result["is_valid"][0] == True:
                valid_num += 1
                if self.do_eval:
                    errs_rotation.append(result["err_rotation"])
                    errs_translation.append(result["err_translation"])
                    errs_naocs_rotation.append(result["err_naocs_rotation"])
                    errs_naocs_translation.append(result["err_naocs_translation"])
        if self.do_eval:
            errs_rotation = np.array(errs_rotation)
            errs_translation = np.array(errs_translation)
            errs_naocs_rotation = np.array(errs_naocs_rotation)
            errs_naocs_translation = np.array(errs_naocs_translation)

            mean_err_rotation = np.mean(errs_rotation, axis=0)
            mean_err_translation = np.mean(errs_translation, axis=0)
            mean_err_naocs_rotation = np.mean(errs_naocs_rotation, axis=0)
            mean_err_naocs_translation = np.mean(errs_naocs_translation, axis=0)
            # Calculate the accuaracy for err_rotation < 5 degree
            acc_err_rotation = np.mean(errs_rotation < 5, axis=0)
            acc_err_naocs_rotation = np.mean(errs_naocs_rotation < 5, axis=0)
            # Calculate the the accuracy for err_rt, rotation < 5 degree, translation < 5 cm
            acc_err_rt = np.mean(
                np.logical_and((errs_rotation < 5), (errs_translation < 0.05)),
                axis=0,
            )
            acc_err_naocs_rt = np.mean(
                np.logical_and((errs_naocs_rotation < 5), (errs_naocs_translation < 0.05)),
                axis=0,
            )

        self.log_string(f"Valid Number {valid_num} / Total number {len(self.results)}")
        if self.do_eval:
            self.log_string(f"The mean rotation error for each part is: {mean_err_rotation}")
            self.log_string(f"The mean translation error for each part is: {mean_err_translation}")
            self.log_string(f"The accuracy for rotation error < 5 degree is: {acc_err_rotation}")
            self.log_string(f"The mean naocs crotation error for each part is: {mean_err_naocs_rotation}")
            self.log_string(f"The mean naocs translation error for each part is: {mean_err_naocs_translation}")
            self.log_string(f"The accuracy for naocs rotation error < 5 degree is: {acc_err_naocs_rotation}")
            self.log_string(
                f"The accuracy for rotation error < 5 degree and translation error < 5 cm is: {acc_err_rt}"
            )
            self.log_string(
                f"The accuracy for naocs rotation error < 5 degree and translation error < 5 cm is: {acc_err_naocs_rt}"
            )

        optimization_result_path = os.path.join(self.output_dir,
                                                self.optimization_result_path)
        f = h5py.File(optimization_result_path, "w")
        # Record the errors
        f.attrs["valid_num"] = valid_num
        if self.do_eval:
            f.attrs["err_pose_rotation"] = mean_err_rotation
            f.attrs["err_pose_translation"] = mean_err_translation
            f.attrs["acc_pose_rotation"] = acc_err_rotation
            f.attrs["acc_pose_rt"] = acc_err_rt
            f.attrs["err_naocs_pose_rotation"] = mean_err_naocs_rotation
            f.attrs["err_naocs_pose_translation"] = mean_err_naocs_translation
            f.attrs["acc_naocs_pose_rotation"] = acc_err_naocs_rotation
            f.attrs["acc_naocs_pose_rt"] = acc_err_naocs_rt
        for i, ins in enumerate(self.instances):
            result = self.results[i]
            group = f.create_group(ins)
            for k, v in self.f_icaf[ins].items():
                # Use the pred seg and npcs from the npcs model
                if k == "pred_npcs_per_point" or k == "pred_seg_per_point":
                    group.create_dataset(
                        k, data=self.f_npcs[ins][k], compression="gzip"
                    )
                else:
                    group.create_dataset(k, data=v, compression="gzip")
            for k, v in result.items():
                group.create_dataset(k, data=v, compression="gzip")
        f.close()

