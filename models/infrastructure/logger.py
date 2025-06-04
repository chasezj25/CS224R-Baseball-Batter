"""
Logger for TensorBoard using tensorboardX. Adapted from original code in
homework 1.
"""

import os
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self._log_dir = log_dir
        print('########################')
        print('Logging outputs to ', log_dir)
        print('########################')
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, iter):
        self._summ_writer.add_scalar('{}'.format(name), scalar, iter)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()
