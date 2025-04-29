import abc
import gc
import os.path as osp
import sys
from typing import Any, Dict, Tuple

import ipdb
import torch
import tqdm

from utils import torch_util
from utils.common import get_log_string
from utils.summary_board import SummaryBoard
from utils.timer import Timer

from .base_trainer import BaseTrainer


class EpochBasedTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        super().__init__(
            cfg,
            parser=parser,
            cudnn_deterministic=cudnn_deterministic,
            autograd_anomaly_detection=autograd_anomaly_detection,
            save_all_snapshots=save_all_snapshots,
            run_grad_check=run_grad_check,
            grad_acc_steps=grad_acc_steps,
        )
        self.max_epoch = cfg.train.optim.max_epoch
        self.best_val_loss = sys.float_info.max
        self.run_grad_check = run_grad_check
        self.mode = cfg.mode
        self.snapshot_steps = cfg.train.snapshot_steps
        self.visualize_steps = cfg.train.visualize_steps
        self.early_stopping_patience = 50

    def before_train_step(self, epoch, iteration, data_dict) -> None:
        pass

    def before_val_step(self, epoch, iteration, data_dict) -> None:
        pass

    def after_train_step(
        self, epoch, iteration, data_dict, output_dict, result_dict
    ) -> None:
        pass

    def after_val_step(
        self, epoch, iteration, data_dict, output_dict, result_dict
    ) -> None:
        pass

    def before_train_epoch(self, epoch) -> None:
        pass

    def before_val_epoch(self, epoch) -> None:
        pass

    def after_train_epoch(self, epoch) -> None:
        pass

    def after_val_epoch(self, epoch) -> None:
        pass

    def train_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def val_step(self, epoch, iteration, data_dict) -> Tuple[Dict, Dict]:
        pass

    def after_backward(
        self, epoch, iteration, data_dict, output_dict, result_dict
    ) -> None:
        pass

    def log_gradients(self, epoch, iteration):
        if not self.run_grad_check:
            return
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            self.writer.add_histogram(
                f"{name}.grad", param.grad, self.epoch, bins="auto"
            )

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        if not self.run_grad_check:
            return
        if not self.check_invalid_gradients():
            self.logger.error(
                "Epoch: {}, iter: {}, invalid gradients.".format(epoch, iteration)
            )
            torch.save(data_dict, "data.pth")
            self.save_snapshot("model.pth.tar")
            self.logger.error("Data_dict and model snapshot saved.")
            ipdb.set_trace()

    def train_epoch(self):
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.epoch)

        self.before_train_epoch(self.epoch)
        self.optimizer.zero_grad()
        total_iterations = len(self.train_loader)

        output_dict = None
        result_dict = None
        self.run_vis = True
        for iteration, data_dict in enumerate(self.train_loader):
            output_dict = None
            result_dict = None
            self.inner_iteration = iteration + 1
            self.iteration += 1
            if data_dict is None:
                continue
            data_dict = (
                torch_util.to_cuda(data_dict)
                if torch.cuda.is_available()
                else data_dict
            )
            self.before_train_step(self.epoch, self.inner_iteration, data_dict)
            self.timer.add_prepare_time()

            # forward
            try:
                output_dict, result_dict = self.train_step(
                    self.epoch, self.inner_iteration, data_dict
                )
                result_dict["loss"].backward(retain_graph=False)
            except (RuntimeError, KeyError) as e:
                self.logger.warning(e)
                data_dict = self.release_tensors(data_dict)
                if output_dict is not None:
                    output_dict = self.release_tensors(output_dict)
                    del output_dict
                if result_dict is not None:
                    result_dict = self.release_tensors(result_dict)
                    del result_dict
                del data_dict
                torch.cuda.empty_cache()
                gc.collect()
                continue
            self.after_backward(
                self.epoch, self.inner_iteration, data_dict, output_dict, result_dict
            )
            self.check_gradients(
                self.epoch, self.inner_iteration, data_dict, output_dict, result_dict
            )
            self.log_gradients(self.epoch, self.epoch)
            if self.cfg.train.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.clip_grad
                )
            self.optimizer_step(self.epoch)

            # after training
            self.timer.add_process_time()
            self.after_train_step(
                self.epoch, self.inner_iteration, data_dict, output_dict, result_dict
            )
            result_dict = self.release_tensors(result_dict)

            self.summary_board.update_from_result_dict(result_dict)
            lr_dict = {"lr": self.get_lr()}
            self.summary_board.update_from_result_dict(lr_dict, 1)

            # logging
            if self.inner_iteration % self.log_steps == 0:
                summary_dict = self.summary_board.summary()
                message = get_log_string(
                    result_dict=summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    lr=self.get_lr(),
                    timer=self.timer,
                )
                self.logger.info(message)
                self.write_event("train", summary_dict, self.iteration)

            if (self.inner_iteration) % 2000 == 0:
                self.inference_epoch()
                self.run_vis = True

            if (self.epoch - 1) % self.visualize_steps == 0 and self.run_vis:
                self.logger.info("Visualizing...")
                try:
                    self.visualize(output_dict, self.epoch, mode="train")
                except RuntimeError as e:
                    self.logger.error(e)
                self.run_vis = False
            torch.cuda.empty_cache()
        self.after_train_epoch(self.epoch)
        message = get_log_string(
            self.summary_board.summary(), epoch=self.epoch, timer=self.timer
        )
        self.logger.critical(message)

        # scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        # snapshot
        if self.epoch % self.snapshot_steps == 0:
            self.save_snapshot(f"epoch-{self.epoch}.pth.tar")

    def inference_epoch(self):
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.val_loader)
        pbar = tqdm.tqdm(enumerate(self.val_loader), total=total_iterations)

        output_dict = None
        self.run_vis = True
        for iteration, data_dict in pbar:
            output_dict = None
            result_dict = None
            self.inner_iteration = iteration + 1
            if data_dict is None:
                continue
            data_dict = torch_util.to_cuda(data_dict)
            self.before_val_step(self.epoch, self.inner_iteration, data_dict)
            timer.add_prepare_time()
            try:
                output_dict, result_dict = self.val_step(
                    self.epoch, self.inner_iteration, data_dict
                )
            except RuntimeError as e:
                self.logger.warning(e)
                data_dict = self.release_tensors(data_dict)
                output_dict = self.release_tensors(output_dict)
                result_dict = self.release_tensors(result_dict)
                del output_dict
                del result_dict
                del data_dict
                torch.cuda.empty_cache()
                gc.collect()
                continue
            timer.add_process_time()
            self.after_val_step(
                self.epoch, self.inner_iteration, data_dict, output_dict, result_dict
            )
            result_dict = self.release_tensors(result_dict)
            summary_board.update_from_result_dict(result_dict)

            if self.inner_iteration % self.log_steps == 0:
                message = get_log_string(
                    result_dict=summary_board.summary(),
                    epoch=self.epoch,
                    iteration=self.inner_iteration,
                    max_iteration=total_iterations,
                    timer=timer,
                )
                self.logger.info(message)

            if len(output_dict) > 0 and self.run_vis:
                self.visualize(output_dict, self.epoch, mode="val")
                self.run_vis = False

            torch.cuda.empty_cache()

        summary_dict = summary_board.summary()
        message = "[Val] " + get_log_string(summary_dict, epoch=self.epoch, timer=timer)
        val_loss = summary_dict["loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_snapshot("best_snapshot.pth.tar")
            self.early_stopping_counter = 0  # Reset early stopping counter
        else:
            self.early_stopping_counter += 1

        self.logger.critical(message)
        self.write_event("val", summary_dict, self.iteration)
        self.set_train_mode()

        # Check for early stopping
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.logger.critical("Early stopping triggered.")
            self.stop_training = True

    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    @abc.abstractmethod
    def visualize(self, data_dict: Dict[str, Any], epoch: int, mode: str = "train"):
        raise NotImplementedError()

    def run(self):
        assert self.train_loader is not None if self.mode == "train" else True
        assert self.val_loader is not None

        if self.args.resume:
            self.load_snapshot(osp.join(self.snapshot_dir, "snapshot.pth.tar"))

        elif self.args.snapshot is not None:
            self.load_snapshot(self.args.snapshot)

        self.set_train_mode()
        while self.epoch < self.max_epoch:
            self.epoch += 1
            if self.mode == "train" or self.mode == "debug_few_scan":
                self.train_epoch()
            if (self.epoch - 1) % self.val_steps == 0:
                self.inference_epoch()
