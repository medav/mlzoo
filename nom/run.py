import sys
import functools
import itertools
import numpy as np
import random

from . import prof
from . import cache

def shortstr(s : str, maxlen : int = 20):
    if len(s) <= maxlen: return s
    return s[:maxlen - 3] + '...'

def can_overlap(ks : list[prof.Kernel]):
    if sum(k.threads_per_block for k in ks) > 1024: return False
    if sum(k.tot_smem for k in ks) > 192 * 1024: return False
    if sum(k.reg_per_block for k in ks) > 65536: return False
    return True


def overlap_lat(ks : list[prof.Kernel]):
    new_lat = max(k.lat for k in ks)
    slowdown = 1

    tot_dram = sum(k.dram_util for k in ks) / 0.9
    if tot_dram > 1: slowdown = max(slowdown, tot_dram)

    tot_sm = sum(k.sm_util for k in ks) / 0.9
    if tot_sm > 1: slowdown = max(slowdown, tot_sm)

    tot_tensor = sum(k.tensor_util for k in ks) / 0.9
    if tot_tensor > 1: slowdown = max(slowdown, tot_tensor)

    return new_lat * slowdown


def powerset(iterable):
    s = list(iterable)
    return list(itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)))

def no_overlap_latency(kern_lists : list[list[prof.Kernel]]):
    return sum(sum(k.lat for k in kl) for kl in kern_lists)

def optimal_latency(kern_lists : list[list[prof.Kernel]]):
    dp_times = np.zeros(tuple(len(kl) + 1 for kl in kern_lists))
    dp_path = np.empty(tuple(len(kl) + 1 for kl in kern_lists), dtype=set)

    def compute_entry(idx : tuple[int, ...]):
        best_lat = None
        best_path = None

        if all(i == 0 for i in idx): return 0, set()

        sets = powerset(range(len(kern_lists)))
        random.shuffle(sets)

        for s in sets:
            s = set(i for i in s if idx[i] > 0)
            if len(s) == 0: continue

            overlap_kerns = [kern_lists[i][idx[i] - 1] for i in s]

            if not can_overlap(overlap_kerns): continue

            resid_idx = tuple(
                idx[i] - 1 if i in s else idx[i]
                for i in range(len(kern_lists))
            )

            lat = dp_times[resid_idx] + overlap_lat(overlap_kerns)

            if best_lat is None or lat < best_lat:
                best_lat = lat
                best_path = s

        return best_lat, best_path

    for idx in itertools.product(*(range(len(kl) + 1) for kl in kern_lists)):
        lat, path = compute_entry(idx)
        dp_times[idx] = lat
        dp_path[idx] = path

    path = []
    idx = tuple(len(kl) for kl in kern_lists)

    while idx != (0,) * len(kern_lists):
        path.append(dp_path[idx])
        idx = tuple(
            idx[i] - 1 if i in dp_path[idx] else idx[i]
            for i in range(len(kern_lists))
        )

    path = list(reversed(path))

    return dp_times[tuple(len(kl) for kl in kern_lists)], path

def print_path(window : list[list[prof.Kernel]], path : list[set[int]], num_streams : int):
    COL_WIDTH = 20
    stream_idx = {i: 0 for i in range(num_streams)}

    for i, s in enumerate(path):
        cols = []
        for j in range(num_streams):
            if j in s:
                k = window[j][stream_idx[j]]
                cols.append(shortstr(k.sanitized_name, maxlen=COL_WIDTH).ljust(COL_WIDTH))
                stream_idx[j] += 1
            else:
                cols.append(' ' * COL_WIDTH)

        print(f'{i:3d} | ' + ' | '.join(cols) + ' |')

def fixed_window_latency(kern_lists : list[list[prof.Kernel]], winlen=8):
    tot_lat = 0
    paths = []
    i = 0
    while not all(len(kl) == 0 for kl in kern_lists):
        window = [
            kl[:winlen] if len(kl) > winlen else kl
            for kl in kern_lists
        ]

        kern_lists = [
            kl[winlen:] if len(kl) > winlen else []
            for kl in kern_lists
        ]

        orig_lat = no_overlap_latency(window)
        window_lat, window_path = optimal_latency(window)
        tot_lat += window_lat
        paths.append(window_path)

        print(f'Window {i}: {window_lat:.2f} ms ({orig_lat / window_lat:.2f} x)')
        print_path(window, window_path, len(window))
        print()

        i += 1

    return tot_lat, paths

@cache.cache_pickle
def cached_prof(prog_args):
    return prof.run_prof(prog_args)

if __name__ == '__main__':
    idx = sys.argv.index('--')
    tool_args = sys.argv[1:idx]
    prog_args = sys.argv[idx + 1:]

    num_mult = 2
    kerns = cached_prof(prog_args)

    orig_lat = sum(k.lat for k in kerns) * num_mult
    print(f'Original latency: {orig_lat:.2f} ms')

    new_lat, path = fixed_window_latency([kerns] * num_mult, winlen=16)
    print(f'New latency: {new_lat:.2f} ms')

    print(f'Speedup: {orig_lat / new_lat:.2f}x')


