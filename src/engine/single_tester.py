import json
import os.path as osp
import time
from typing import Any

import torch
from tqdm import tqdm

from configs import Config
from utils import torch_util
from utils.common import get_log_string
from utils.logger import Logger
from utils.summary_board import SummaryBoard

from .base_tester import BaseTester


class SingleTester(BaseTester):
    def __init__(self, cfg: Config, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)
        self.summary_board = SummaryBoard(last_n=None, adaptive=True)
        self.snapshot_dir = cfg.snapshot_dir
        self.log_dir = cfg.log_dir
        self.event_dir = cfg.event_dir
        self.output_dir = cfg.output_dir
        self.log_dir = cfg.log_dir
        log_file = osp.join(
            self.log_dir, "train-{}.log".format(time.strftime("%Y%m%d-%H%M%S"))
        )
        self.logger = Logger(log_file, local_rank=self.args.local_rank)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Any:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Any:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        assert self.model is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_dir, "snapshot.pth.tar"))

        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)

        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            # on start
            output_dict = None
            result_dict = None
            self.iteration = iteration + 1
            data_dict = (
                torch_util.to_cuda(data_dict)
                if self.device == torch.device("cuda")
                else data_dict
            )
            self.before_test_step(self.iteration, data_dict)
            # test step
            if self.device == torch.device("cuda"):
                torch.cuda.synchronize()
            self.timer.add_prepare_time()
            try:
                output_dict = self.test_step(self.iteration, data_dict)
                if len(output_dict) == 0:
                    continue
                result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            except Exception as e:
                self.logger.error(f"Error in iteration {self.iteration}")
                import traceback

                traceback.print_exc()
                if output_dict is not None:
                    output_dict = self.release_tensors(output_dict)
                    del output_dict
                if result_dict is not None:
                    result_dict = self.release_tensors(result_dict)
                    del result_dict
                if data_dict is not None:
                    data_dict = self.release_tensors(data_dict)
                    del data_dict
                if self.device == torch.device("cuda"):
                    torch.cuda.empty_cache()
                continue
            if self.device == torch.device("cuda"):
                torch.cuda.synchronize()
            self.timer.add_process_time()
            # eval step
            result_dict = self.release_tensors(result_dict)
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            self.summary_board.update_from_result_dict(result_dict)
            message = get_log_string(
                result_dict=self.summary_board.summary(),
                epoch=self.iteration,
                iteration=self.iteration,
                max_iteration=total_iterations,
                timer=self.timer,
            )

            pbar.set_description(message)
            if iteration % 10 == 0:
                self.logger.info(message)
                self.dump_metrics(
                    self.summary_board.full_summary(),
                    osp.join(self.event_dir, "metrics.json"),
                )
            if output_dict is not None:
                output_dict = self.release_tensors(output_dict)
                del output_dict
            if result_dict is not None:
                result_dict = self.release_tensors(result_dict)
                del result_dict
            if data_dict is not None:
                data_dict = self.release_tensors(data_dict)
                del data_dict
            torch.cuda.empty_cache()

        self.after_test_epoch()
        self.print_metrics(self.summary_board.full_summary())
        self.dump_metrics(
            self.summary_board.full_summary(), osp.join(self.event_dir, "metrics.json")
        )

    def print_metrics(self, results_dict):
        message = get_log_string(results_dict)
        self.logger.info(message)
        print(message)
        return message

    def dump_metrics(self, results_dict, file_path):
        file_path = osp.join(self.event_dir, "metrics.json")
        json.dump(results_dict, open(file_path, "w"))
        self.logger.info(f"Metrics have been saved to {file_path}")
