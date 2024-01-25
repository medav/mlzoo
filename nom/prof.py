import os
import sys
import functools
import tempfile
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
NSYS_PATH = os.environ.get('NSYS_PATH', 'nsys')

@dataclass
class Kernel:
    name : str
    grid : tuple
    block : tuple
    reg_per_thread : int
    static_smem : int
    dynamic_smem : int
    metrics : dict

    @functools.cached_property
    def sanitized_name(self):
        sn = self.name.replace('void ', '').replace('at::native::', '').replace('<unnamed>::', '').replace('cutlass::', '')
        if '<' in sn: sn = sn[:sn.index('<')]
        if '(' in sn: sn = sn[:sn.index('(')]
        return sn

    @property
    def threads_per_block(self): return self.block[0] * self.block[1] * self.block[2]

    @property
    def reg_per_block(self): return self.reg_per_thread * self.threads_per_block

    @property
    def tot_smem(self): return self.static_smem + self.dynamic_smem

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

def read_nsys_output(output):
    it = iter(output.split('\n'))
    line = ''
    while not line.startswith('Start (ns)'):
        line = next(it)

    yield line + '\n'

    for line in it: yield line + '\n'

def run_prof(prog_args, **kwargs) -> list[Kernel]:
    use_cuda_profiler_capture = kwargs.get('use_cuda_profiler_capture', False)
    ncu_replay_mode = kwargs.get('ncu_replay_mode', 'application')
    ncu_metrics = kwargs.get('ncu_metrics', default_metrics)

    print('>>> Running NCU...')

    cmdline = [
        NCU_PATH,
        '--csv',
        '--target-processes', 'all',
        '--profile-from-start', 'no' if use_cuda_profiler_capture else 'yes',
        '--replay-mode', ncu_replay_mode,
        '--metrics', ','.join(ncu_metrics)
    ] + prog_args

    ncu_output = subprocess.check_output(cmdline).decode()

    print('>>> Done!')


    ncu_df = pd.read_csv(
        Reader(read_ncu_output(ncu_output)),
        low_memory=False,
        thousands=r',')

    ncu_names = dict()
    ncu_metrics = dict()

    for row in ncu_df.iterrows():
        row = row[1]
        ncu_names[row['ID']] = row['Kernel Name']
        if row['ID'] not in ncu_metrics: ncu_metrics[row['ID']] = dict()
        ncu_metrics[row['ID']][row['Metric Name']] = row['Metric Value']


    print('>>> Running NSYS...')
    ofile = tempfile.mktemp(suffix='.nsys-rep')

    cmdline = [
        NSYS_PATH,
        'profile',
        '-t', 'cuda,cudnn,cublas',
        '-o', ofile
    ] + prog_args

    if use_cuda_profiler_capture:
        cmdline += [
            '--capture-range=cudaProfilerApi',
            '--capture-range-end=stop'
        ]

    subprocess.check_output(cmdline).decode()

    print('>>> Done!')

    stats_cmdline = [
        NSYS_PATH,
        'stats',
        '-r', 'cuda_gpu_trace',
        '-f', 'csv',
        ofile
    ]

    stats_output = subprocess.check_output(stats_cmdline).decode()
    os.remove(ofile)


    nsys_df = pd.read_csv(
        Reader(read_nsys_output(stats_output)),
        low_memory=False,
        thousands=r',')

    ordered_ids = sorted(ncu_names.keys())
    ordered_names = [ncu_names[i] for i in ordered_ids]

    kerns = []

    kid = 0
    for row in nsys_df.iterrows():
        row = row[1]
        if row['Name'] != ordered_names[kid]: continue

        kerns.append(Kernel(
            name=row['Name'],
            grid= (int(row['GrdX']), int(row['GrdY']), int(row['GrdZ'])),
            block=(int(row['BlkX']), int(row['BlkY']), int(row['BlkZ'])),
            reg_per_thread=int(row['Reg/Trd']),
            static_smem=int(float(row['StcSMem (MB)']) * 2**20),
            dynamic_smem=int(float(row['DymSMem (MB)']) * 2**20),
            metrics=ncu_metrics[ordered_ids[kid]]
        ))

        kid += 1

    assert kid == len(ordered_ids)
    return kerns

if __name__ == '__main__':
    kerns = run_prof(sys.argv[2:])
    for k in kerns:
        print(k.name)
