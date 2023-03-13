import os
import json
from collections import namedtuple

class BaseTrainOptions():

    def parse_args(self, arg_list=None):
        if arg_list is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(arg_list)

        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self._save_dump()
            return self.args

    def _save_dump(self):
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return

class BaseTestOptions():

    def parse_args(self):
        args = self.parser.parse_args()
        path_to_json = os.path.abspath(args.json)
        with open(path_to_json, "r") as f:
            train_args = json.load(f)
            train_args = namedtuple("train_args", train_args.keys())(**train_args)
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        return args, train_args
