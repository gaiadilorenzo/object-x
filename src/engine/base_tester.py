import abc
import json
import os.path as osp
import sys
import time

import torch

from utils import torch_util
from utils.logger import Logger
from utils.timer import Timer


class BaseTester(abc.ABC):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        # parser
        self.args, _ = parser.parse_known_args()

        # logger
        log_file = osp.join(
            cfg.log_dir, "test-{}.log".format(time.strftime("%Y%m%d-%H%M%S"))
        )
        self.logger = Logger(log_file=log_file)

        # command executed
        message = "Command executed: " + " ".join(sys.argv)
        self.logger.info(message)

        # print config
        message = "Configs:\n" + json.dumps(cfg.dict(), indent=4)
        self.logger.info(message)

        # cuda
        if not torch.cuda.is_available():
            self.logger.warning("No CUDA devices available.")
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.cudnn_deterministic = cudnn_deterministic
        self.seed = cfg.seed
        torch_util.initialize(
            seed=self.seed, cudnn_deterministic=self.cudnn_deterministic
        )

        # state
        self.model = None
        self.iteration = None

        self.test_loader = None
        self.saved_states = {}

        self.timer = Timer()

    def load_snapshot(self, snapshot):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=self.device)
        assert "model" in state_dict, "No model can be loaded."
        self.model.load_state_dict(state_dict["model"], strict=False)
        # log missing keys
        missing_keys = set(self.model.state_dict().keys()) - set(
            state_dict["model"].keys()
        )
        if len(missing_keys) > 0:
            self.logger.warning("Missing keys: {}".format(missing_keys))
        # load additional keys
        additional_keys = set(state_dict["model"].keys()) - set(
            self.model.state_dict().keys()
        )
        if len(additional_keys) > 0:
            self.logger.warning("Additional keys: {}".format(additional_keys))
        self.logger.info("Model has been loaded.")

    def register_model(self, model):
        r"""Register model. DDP is automatically used."""
        self.model = model
        message = "Model description:\n" + str(model)
        self.logger.info(message)
        return model

    def register_dataset(self, test_dataset):
        r"""Register data set."""
        self.test_dataset = test_dataset

    def register_loader(self, test_loader):
        r"""Register data loader."""
        self.test_loader = test_loader

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    def release_tensors(self, result_dict):
        r"""All reduce and release tensors."""
        result_dict = torch_util.release_cuda(result_dict)
        return result_dict
