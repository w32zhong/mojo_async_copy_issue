from math import ceildiv
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from gpu.host import DeviceBuffer
from gpu.memory import AddressSpace, async_copy_wait_all

from gpu import (
    block_dim,
    block_idx,
    thread_idx,
    barrier
)
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram_async,
)


fn tiled_register_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, COMPUTE_THREADS: Int,
        NUM_THREADS: Int, version: StaticString
    ](
        A: LayoutTensor[dtype, A_layout, MutAnyOrigin],
        B: LayoutTensor[dtype, B_layout, MutAnyOrigin],
        C: LayoutTensor[dtype, C_layout, MutAnyOrigin],
    ):
        var M = A.dim[0]()
        var K = B.dim[0]()
        var N = B.dim[1]()

        var subtile_row = thread_idx.x // BN
        var subtile_col = thread_idx.x % BN
        var max_subtile_rows = BM // TM
        var participates_in_compute = (
            subtile_row < max_subtile_rows and
            thread_idx.x < COMPUTE_THREADS
        )

        var A_smem = LayoutTensor[
            dtype,
            Layout.row_major(BM, BK),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var B_smem = LayoutTensor[
            dtype,
            Layout.row_major(BK, BN),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        var dst_reg = LayoutTensor[
            dtype,
            Layout(TM),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()

        var dst_subtile = C.tile[BM, BN](block_idx.y, block_idx.x)
                           .tile[TM, 1](subtile_row, subtile_col)
        dst_reg.copy_from(dst_subtile) # copy the initial zeros

        barrier()

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(BM, BK)
            comptime B_tile_layout = Layout.row_major(BK, BN)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            if version == "good":
                A_smem.copy_from(A_tile)
                B_smem.copy_from(B_tile)
            elif version == "bad":
                copy_dram_to_sram_async[thread_layout=A_tile_layout](A_smem, A_tile)
                copy_dram_to_sram_async[thread_layout=B_tile_layout](B_smem, B_tile)
                async_copy_wait_all()
            barrier()

            if participates_in_compute:
                for k in range(BK):
                    var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                    var B_subtile = B_smem.tile[1, BN](k, 0)
                    var B_element = B_subtile[0, subtile_col]

                    for t in range(TM):
                        dst_reg[t] += A_subtile[t, 0] * B_element

            barrier()

        if participates_in_compute:
            dst_subtile.copy_from(dst_reg)


@compiler.register("my_matmul")
struct MyMatMul[version: StaticString]:
    @staticmethod
    fn execute(
        raw_output: OutputTensor[rank=2],
        raw_A: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        raw_B: InputTensor[dtype = raw_output.dtype, rank = raw_output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        A = raw_A.to_layout_tensor()
        B = raw_B.to_layout_tensor()
        output = raw_output.to_layout_tensor()

        M = A.shape[0]()
        N = B.shape[1]()

        device_ctx = ctx.get_device_context()

        device_ctx.enqueue_memset(
            DeviceBuffer[output.dtype](
                device_ctx,
                output.ptr,
                (M * N),
                owning=False,
            ),
            0, # fill zeros
        )

        comptime OPTIMIZED_BLOCK_SIZE = 16
        comptime BM = OPTIMIZED_BLOCK_SIZE
        comptime BN = OPTIMIZED_BLOCK_SIZE
        comptime BK = OPTIMIZED_BLOCK_SIZE

        comptime TM = 4
        comptime COMPUTE_THREADS = (BM * BN) // TM
        comptime COPY_THREADS = max(BM * BK, BK * BN)
        comptime NUM_THREADS = max(COMPUTE_THREADS, COPY_THREADS)
        comptime kernel = tiled_register_matmul[
            output.dtype, A.layout, B.layout, output.layout,
            BM, BK, BN, TM, COMPUTE_THREADS, NUM_THREADS, version
        ]

        device_ctx.enqueue_function_checked[kernel, kernel](
            A, B, output,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=NUM_THREADS,
        )
