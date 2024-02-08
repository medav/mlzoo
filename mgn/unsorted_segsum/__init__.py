import os
import torch
from torch.utils.cpp_extension import load
from torch.library import Library, impl

import time

cur_path = os.path.dirname(os.path.realpath(__file__))

cpu_unsorted_segsum = load('cpu_unsorted_segsum',
    [f'{cur_path}/cpu_extension.cc'],
    extra_cflags=['-fopenmp', '-O3', '-march=native'],
    extra_ldflags=['-lgomp', '-O3', '-march=native'],
    verbose=False)

import cpu_unsorted_segsum

if torch.cuda.is_available():
    cuda_unsorted_segsum = load('cuda_unsorted_segsum',
        [f'{cur_path}/cuda_extension.cu'],
        extra_cflags=['-fopenmp', '-O3', '-march=native'],
        extra_ldflags=['-lgomp', '-O3', '-march=native'],
        verbose=False)

    import cuda_unsorted_segsum

else:
    cuda_unsorted_segsum = None
    print('CUDA not available, cuda_unsorted_segsum will not be available')

def unsorted_segment_sum_ref(
    data : torch.Tensor,
    indices : torch.Tensor,
    num_segments : int
) -> torch.Tensor:
    return torch.cat([
        data[indices == i].sum(dim=0, keepdim=True)
        for i in range(num_segments)
    ], dim=0)


lib = torch.library.Library("unsorted_segsum", "DEF")
lib.define("unsorted_segment_sum_fwd(Tensor data, Tensor indices, int num_segments) -> Tensor")
lib.define("unsorted_segment_sum_bwd(Tensor grad, Tensor indices) -> Tensor")


@impl(lib, 'unsorted_segment_sum_fwd', 'Meta')
def unsorted_segment_sum_fwd_meta(data, indices, num_segments):
    return torch.empty((num_segments, data.size(1)), dtype=data.dtype, device='meta')

@impl(lib, 'unsorted_segment_sum_fwd', 'CPU')
def unsorted_segment_sum_fwd_cpu(data, indices, num_segments):
    return cpu_unsorted_segsum.unsorted_segment_sum_fwd(data, indices, num_segments)

@impl(lib, 'unsorted_segment_sum_fwd', 'CUDA')
def unsorted_segment_sum_fwd_cuda(data, indices, num_segments):
    return cuda_unsorted_segsum.unsorted_segment_sum_fwd(data, indices, num_segments)

@impl(lib, 'unsorted_segment_sum_bwd', 'Meta')
def unsorted_segment_sum_bwd_meta(grad, indices):
    return torch.empty((indices.size(0), grad.size(1)), dtype=grad.dtype, device='meta')

@impl(lib, 'unsorted_segment_sum_bwd', 'CPU')
def unsorted_segment_sum_bwd_cpu(grad, indices):
    return cpu_unsorted_segsum.unsorted_segment_sum_bwd(grad, indices)

@impl(lib, 'unsorted_segment_sum_bwd', 'CUDA')
def unsorted_segment_sum_bwd_cuda(grad, indices):
    return cuda_unsorted_segsum.unsorted_segment_sum_bwd(grad, indices)

class UnsortedSegmentSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data : torch.Tensor, indices  : torch.Tensor, num_segments : int) -> torch.Tensor:
        assert data.shape[0] > 0, f'UnsortedSegmentSum: data.shape[0] must be > 0, got shape={data.shape}'
        ctx.save_for_backward(indices)
        return torch.ops.unsorted_segsum.unsorted_segment_sum_fwd(data.contiguous(), indices.contiguous(), num_segments)

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return torch.ops.unsorted_segsum.unsorted_segment_sum_bwd(grad.contiguous(), indices.contiguous()), None, None

def unsorted_segment_sum(
    data : torch.Tensor,
    indices  : torch.Tensor,
    num_segments : int
) -> torch.Tensor:
    return UnsortedSegmentSum.apply(data, indices, num_segments)

def unit_test_cpu():
    print('==== Correctness Test CPU ====')
    data = torch.randn(1000, 3, requires_grad=False)
    indices = torch.randint(0, 100, (1000,), requires_grad=False)
    num_segments = 100

    d1 = data.clone().requires_grad_()
    d2 = data.clone().requires_grad_()

    ref = unsorted_segment_sum_ref(d1, indices, num_segments)
    out = UnsortedSegmentSum.apply(d2, indices, num_segments)

    print('(FWD) L2 = ', (ref - out).pow(2).sum().sqrt())

    ref.pow(2).sum().backward()
    out.pow(2).sum().backward()

    print('(BWD) L2 = ', (d1.grad - d2.grad).pow(2).sum().sqrt())

def unit_test_gpu():
    print('==== Correctness Test GPU ====')
    data = torch.randn(1000, 3, requires_grad=False)
    indices = torch.randint(0, 100, (1000,), requires_grad=False)
    num_segments = 100

    d1 = data.clone().requires_grad_()
    d2 = data.clone().cuda().requires_grad_()

    ref = unsorted_segment_sum_ref(d1, indices, num_segments)
    out = UnsortedSegmentSum.apply(d2, indices.clone().cuda(), num_segments)

    print('(FWD) L2 = ', (ref - out.cpu()).pow(2).sum().sqrt())

    ref.pow(2).sum().backward()
    out.pow(2).sum().backward()

    print('(BWD) L2 = ', (d1.grad - d2.grad.cpu()).pow(2).sum().sqrt())


if __name__ == '__main__':
    unit_test_cpu()
    unit_test_gpu()

    exit(0)

    # Benchmark

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = unsorted_segment_sum_ref(data, indices, num_segments)
    t1 = time.perf_counter()
    print(f'Reference (Fwd): {(t1 - t0) * 1000:.2f} ms')

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = UnsortedSegmentSum.apply(data, indices, num_segments)
    t1 = time.perf_counter()
    print(f'Extension (Fwd): {(t1 - t0) * 1000:.2f} ms')

    t0 = time.perf_counter()
    for _ in range(1000):
        out = unsorted_segment_sum_ref(d1, indices, num_segments)
        out.pow(2).sum().backward()
    t1 = time.perf_counter()
    print(f'Reference (Fwd + Bwd): {(t1 - t0) * 1000:.2f} ms')

    t0 = time.perf_counter()
    for _ in range(1000):
        out = UnsortedSegmentSum.apply(d2, indices, num_segments)
        out.pow(2).sum().backward()
    t1 = time.perf_counter()
    print(f'Extension (Fwd + Bwd): {(t1 - t0) * 1000:.2f} ms')



