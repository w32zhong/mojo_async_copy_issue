from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort

fn barrier():
    pass
fn async_copy_wait_all():
    pass
struct MockGPUIndex:
    var x: Int
    var y: Int
    var z: Int
from math import ceildiv
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram_async,
)
fn tiled_register_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, COMPUTE_THREADS: Int,
        NUM_THREADS: Int, version: StaticString,

        block_idx: MockGPUIndex, thread_idx: MockGPUIndex
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
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var dst_reg = LayoutTensor[
            dtype,
            Layout(TM),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
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
fn tiled_register_matmul_wrapper(py_obj: PythonObject) raises -> PythonObject:
    return py_obj

fn echo(py_obj: PythonObject) raises -> PythonObject:
    var n = Int(py_obj)
    return n

@export
fn PyInit_bar() -> PythonObject:
    try:
        var m = PythonModuleBuilder("my lovely python binding!")
        m.def_function[echo]("my_mojo_echo")
        m.def_function[tiled_register_matmul_wrapper]("my_mojo_matmul")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))
        return PythonObject()
