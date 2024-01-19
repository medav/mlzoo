import os
import sys
import pandas as pd
from dataclasses import dataclass
import subprocess

default_metrics = [
    'gpu__time_duration.sum',
    'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    # 'sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed'
]

NCU_PATH = os.environ.get('NCU_PATH', 'ncu')

@dataclass
class Kernel:
    name : str
    metrics : dict

    @property
    def lat(self): return self.metrics['gpu__time_duration.sum']

    @property
    def dram_util(self): return self.metrics['gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'] / 100

    @property
    def sm_util(self): return self.metrics['sm__throughput.avg.pct_of_peak_sustained_elapsed'] / 100

    @property
    def tensor_util(self): return self.metrics['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'] / 100

class Reader(object):
    def __init__(self, g):
        self.g = g
    def read(self, n=0):
        try: return next(self.g)
        except StopIteration: return ''

def read_ncu_output(output):
    it = iter(output.split('\n'))
    line = ''
    while not line.startswith('"ID","Process ID","Process Name",'):
        line = next(it)

    yield line + '\n'

    for line in it: yield line + '\n'


def run_ncu(prog_args, **kwargs):
    print('>>> Running NCU...')

    prof_from_start = kwargs.get('prof_from_start', True)
    replay_mode = kwargs.get('replay_mode', 'application')
    metrics = kwargs.get('metrics', default_metrics)

    # NV_COMPUTE_PROFILER_REPLAY_MODE=off

    cmdline = [
        NCU_PATH,
        '--csv',
        '--target-processes', 'all',
        '--profile-from-start', 'yes' if prof_from_start else 'no',
        '--replay-mode', replay_mode,
        '--metrics', ','.join(metrics)
    ] + prog_args

    output = subprocess.check_output(cmdline).decode()

    print('>>> Done!')

    df = pd.read_csv(
        Reader(read_ncu_output(output)),
        low_memory=False,
        thousands=r',')

    names = dict()
    metrics = dict()

    for row in df.iterrows():
        row = row[1]
        names[row['ID']] = row['Kernel Name']
        if row['ID'] not in metrics: metrics[row['ID']] = dict()
        metrics[row['ID']][row['Metric Name']] = row['Metric Value']

    return [Kernel(names[k], metrics[k]) for k in sorted(names.keys())]


if __name__ == '__main__':
    prog_args = sys.argv[1:]
    kerns = run_ncu(prog_args)

    for k in kerns:
        print(k.name)
        print(f'    DRAM:   {k.dram_util:.2%}')
        print(f'    SM:     {k.sm_util:.2%}')
        print(f'    Tensor: {k.tensor_util:.2%}')
