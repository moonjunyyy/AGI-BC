
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /data/moonjunyyy/workspace/AGI-BC/torchinductor_moonjunyyy/dh/cdhlil5phnkklyjqfzoxrlyatq6ah4u3tfnzylxvog7sgmktb5cv.py
# Source Nodes: [getattr_l__fn_____0___relu], Original ATen: [aten.relu]
# getattr_l__fn_____0___relu => relu
triton_poi_fused_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65261568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /data/moonjunyyy/workspace/AGI-BC/torchinductor_moonjunyyy/23/c23wlyhrpfqq7dereq6xpz3dwo67awnqlzlalrz25rv7vhk57quv.py
# Source Nodes: [add, getattr_l__fn_____0___dropout, getattr_l__fn_____0___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
# add => add
# getattr_l__fn_____0___dropout => gt, mul, mul_1
# getattr_l__fn_____0___relu_1 => relu_1
triton_poi_fused_add_native_dropout_relu_threshold_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_relu_threshold_backward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32630784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 768
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp7 = tl.load(in_ptr2 + (x0), None)
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp6 = tmp4.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = triton_helpers.maximum(0, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 1.1111111111111112
    tmp13 = tmp11 * tmp12
    tmp14 = tmp5 + tmp13
    tmp15 = 0.0
    tmp16 = tmp10 <= tmp15
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp14, None)
    tl.store(out_ptr3 + (x0), tmp16, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35 = args
    args.clear()
    assert_size_stride(primals_1, (1536, 768), (768, 1))
    assert_size_stride(primals_2, (1536, ), (1, ))
    assert_size_stride(primals_3, (768, 1536), (1536, 1))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (1536, 768), (768, 1))
    assert_size_stride(primals_6, (1536, ), (1, ))
    assert_size_stride(primals_7, (768, 1536), (1536, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (1536, 768), (768, 1))
    assert_size_stride(primals_10, (1536, ), (1, ))
    assert_size_stride(primals_11, (768, 1536), (1536, 1))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (1536, 768), (768, 1))
    assert_size_stride(primals_14, (1536, ), (1, ))
    assert_size_stride(primals_15, (768, 1536), (1536, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (1536, 768), (768, 1))
    assert_size_stride(primals_18, (1536, ), (1, ))
    assert_size_stride(primals_19, (768, 1536), (1536, 1))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (1536, 768), (768, 1))
    assert_size_stride(primals_22, (1536, ), (1, ))
    assert_size_stride(primals_23, (768, 1536), (1536, 1))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (1536, 768), (768, 1))
    assert_size_stride(primals_26, (1536, ), (1, ))
    assert_size_stride(primals_27, (768, 1536), (1536, 1))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (1536, 768), (768, 1))
    assert_size_stride(primals_30, (1536, ), (1, ))
    assert_size_stride(primals_31, (768, 1536), (1536, 1))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (256, 768), (768, 1))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (42488, 768), (768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(primals_35, reinterpret_tensor(primals_1, (768, 1536), (1, 768), 0), out=buf0)
        del primals_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [getattr_l__fn_____0___relu], Original ATen: [aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_relu_0.run(buf1, primals_2, 65261568, grid=grid(65261568), stream=stream0)
        del primals_2
        buf2 = empty((42488, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf1, reinterpret_tensor(primals_3, (1536, 768), (1, 1536), 0), out=buf2)
        buf3 = empty((8, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [8], out=buf3)
        buf5 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf6 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf57 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add, getattr_l__fn_____0___dropout, getattr_l__fn_____0___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, primals_35, buf2, primals_4, buf5, buf6, buf57, 0, 32630784, grid=grid(32630784), stream=stream0)
        del primals_4
        buf7 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf6, reinterpret_tensor(primals_5, (768, 1536), (1, 768), 0), out=buf7)
        buf8 = buf7; del buf7  # reuse
        # Source Nodes: [getattr_l__fn_____1___relu], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf8, primals_6, 65261568, grid=grid(65261568), stream=stream0)
        del primals_6
        buf9 = buf2; del buf2  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_7, (1536, 768), (1, 1536), 0), out=buf9)
        buf11 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf12 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf56 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add_1, getattr_l__fn_____1___dropout, getattr_l__fn_____1___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, buf6, buf9, primals_8, buf11, buf12, buf56, 1, 32630784, grid=grid(32630784), stream=stream0)
        del primals_8
        buf13 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf12, reinterpret_tensor(primals_9, (768, 1536), (1, 768), 0), out=buf13)
        buf14 = buf13; del buf13  # reuse
        # Source Nodes: [getattr_l__fn_____2___relu], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf14, primals_10, 65261568, grid=grid(65261568), stream=stream0)
        del primals_10
        buf15 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf14, reinterpret_tensor(primals_11, (1536, 768), (1, 1536), 0), out=buf15)
        buf17 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf18 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf55 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add_2, getattr_l__fn_____2___dropout, getattr_l__fn_____2___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, buf12, buf15, primals_12, buf17, buf18, buf55, 2, 32630784, grid=grid(32630784), stream=stream0)
        del primals_12
        buf19 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf18, reinterpret_tensor(primals_13, (768, 1536), (1, 768), 0), out=buf19)
        buf20 = buf19; del buf19  # reuse
        # Source Nodes: [getattr_l__fn_____3___relu], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf20, primals_14, 65261568, grid=grid(65261568), stream=stream0)
        del primals_14
        buf21 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_15, (1536, 768), (1, 1536), 0), out=buf21)
        buf23 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf24 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf54 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add_3, getattr_l__fn_____3___dropout, getattr_l__fn_____3___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, buf18, buf21, primals_16, buf23, buf24, buf54, 3, 32630784, grid=grid(32630784), stream=stream0)
        del primals_16
        buf25 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf24, reinterpret_tensor(primals_17, (768, 1536), (1, 768), 0), out=buf25)
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [getattr_l__fn_____4___relu], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf26, primals_18, 65261568, grid=grid(65261568), stream=stream0)
        del primals_18
        buf27 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf26, reinterpret_tensor(primals_19, (1536, 768), (1, 1536), 0), out=buf27)
        buf29 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf30 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf53 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add_4, getattr_l__fn_____4___dropout, getattr_l__fn_____4___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, buf24, buf27, primals_20, buf29, buf30, buf53, 4, 32630784, grid=grid(32630784), stream=stream0)
        del primals_20
        buf31 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf30, reinterpret_tensor(primals_21, (768, 1536), (1, 768), 0), out=buf31)
        buf32 = buf31; del buf31  # reuse
        # Source Nodes: [getattr_l__fn_____5___relu], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf32, primals_22, 65261568, grid=grid(65261568), stream=stream0)
        del primals_22
        buf33 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf32, reinterpret_tensor(primals_23, (1536, 768), (1, 1536), 0), out=buf33)
        buf35 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf36 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf52 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add_5, getattr_l__fn_____5___dropout, getattr_l__fn_____5___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, buf30, buf33, primals_24, buf35, buf36, buf52, 5, 32630784, grid=grid(32630784), stream=stream0)
        del primals_24
        buf37 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf36, reinterpret_tensor(primals_25, (768, 1536), (1, 768), 0), out=buf37)
        buf38 = buf37; del buf37  # reuse
        # Source Nodes: [getattr_l__fn_____6___relu], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf38, primals_26, 65261568, grid=grid(65261568), stream=stream0)
        del primals_26
        buf39 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf38, reinterpret_tensor(primals_27, (1536, 768), (1, 1536), 0), out=buf39)
        buf41 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf42 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf51 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add_6, getattr_l__fn_____6___dropout, getattr_l__fn_____6___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, buf36, buf39, primals_28, buf41, buf42, buf51, 6, 32630784, grid=grid(32630784), stream=stream0)
        del primals_28
        buf43 = empty((42488, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf42, reinterpret_tensor(primals_29, (768, 1536), (1, 768), 0), out=buf43)
        buf44 = buf43; del buf43  # reuse
        # Source Nodes: [getattr_l__fn_____7___relu], Original ATen: [aten.relu]
        triton_poi_fused_relu_0.run(buf44, primals_30, 65261568, grid=grid(65261568), stream=stream0)
        del primals_30
        buf45 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_31, (1536, 768), (1, 1536), 0), out=buf45)
        buf47 = empty((42488, 768), device='cuda', dtype=torch.bool)
        buf48 = empty((42488, 768), device='cuda', dtype=torch.float32)
        buf50 = empty((42488, 768), device='cuda', dtype=torch.bool)
        # Source Nodes: [add_7, getattr_l__fn_____7___dropout, getattr_l__fn_____7___relu_1], Original ATen: [aten.add, aten.native_dropout, aten.relu, aten.threshold_backward]
        triton_poi_fused_add_native_dropout_relu_threshold_backward_1.run(buf3, buf42, buf45, primals_32, buf47, buf48, buf50, 7, 32630784, grid=grid(32630784), stream=stream0)
        del buf3
        del buf45
        del primals_32
        buf49 = empty((42488, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [fn_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_34, buf48, reinterpret_tensor(primals_33, (768, 256), (1, 768), 0), alpha=1, beta=1, out=buf49)
        del primals_34
        return (buf49, primals_35, buf1, buf5, buf6, buf8, buf11, buf12, buf14, buf17, buf18, buf20, buf23, buf24, buf26, buf29, buf30, buf32, buf35, buf36, buf38, buf41, buf42, buf44, buf47, buf48, reinterpret_tensor(primals_33, (256, 768), (768, 1), 0), buf50, reinterpret_tensor(primals_31, (768, 1536), (1536, 1), 0), reinterpret_tensor(primals_29, (1536, 768), (768, 1), 0), buf51, reinterpret_tensor(primals_27, (768, 1536), (1536, 1), 0), reinterpret_tensor(primals_25, (1536, 768), (768, 1), 0), buf52, reinterpret_tensor(primals_23, (768, 1536), (1536, 1), 0), reinterpret_tensor(primals_21, (1536, 768), (768, 1), 0), buf53, reinterpret_tensor(primals_19, (768, 1536), (1536, 1), 0), reinterpret_tensor(primals_17, (1536, 768), (768, 1), 0), buf54, reinterpret_tensor(primals_15, (768, 1536), (1536, 1), 0), reinterpret_tensor(primals_13, (1536, 768), (768, 1), 0), buf55, reinterpret_tensor(primals_11, (768, 1536), (1536, 1), 0), reinterpret_tensor(primals_9, (1536, 768), (768, 1), 0), buf56, reinterpret_tensor(primals_7, (768, 1536), (1536, 1), 0), reinterpret_tensor(primals_5, (1536, 768), (768, 1), 0), buf57, reinterpret_tensor(primals_3, (768, 1536), (1536, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((42488, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
