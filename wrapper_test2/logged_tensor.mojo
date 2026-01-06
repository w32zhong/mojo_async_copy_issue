from random import random_si64
from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace


struct LoggedTensor[
    mut: Bool,
    //,
    dtype: DType,
    layout: Layout,
    origin: Origin[mut=mut],
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    alias ImplType = LayoutTensor[
        dtype, layout, origin,
        address_space=address_space,
    ]

    var impl: Self.ImplType
    var name: String
    var origin_x: Int
    var origin_y: Int

    fn __init__(out self,
                impl: Self.ImplType,
                name: String = "Tensor",
                origin_x: Int = 0,
                origin_y: Int = 0):
        self.impl = impl
        self.name = name
        self.origin_x = origin_x
        self.origin_y = origin_y

    fn print(read self):
        for row in range(self.impl.shape[0]()):
            print(self.name, end=": ")
            for col in range(self.impl.shape[1]()):
                print(
                    String(self.impl[row, col]).rjust(5),
                    end=""
                )
            print()

    fn dim[idx: Int](self) -> Int:
        return self.impl.dim[idx]()

    @staticmethod
    fn stack_allocation[
        *, stack_alignment: Int = Self.ImplType.alignment
    ]() -> LoggedTensor[dtype, layout, MutAnyOrigin, address_space=address_space]:
        var impl = Self.ImplType.stack_allocation[stack_alignment=stack_alignment]()
        return LoggedTensor(impl, String(address_space))


fn example_logged_tensor[
        rows: Int, cols: Int
    ](name: String) -> LoggedTensor[
        DType.float32,
        Layout.row_major(rows, cols),
        MutAnyOrigin
    ]:
    comptime buf_size = rows * cols
    var ptr = alloc[Float32](buf_size)
    for i in range(buf_size):
        ptr[i] = Float32(random_si64(-5, 5))
    comptime layout = Layout.row_major(rows, cols)
    var tensor = LayoutTensor[DType.float32, layout](UnsafePointer[Float32, MutAnyOrigin](ptr))
    return LoggedTensor(tensor, name)


struct MockGPUIndex:
    var x: Int
    var y: Int
    var z: Int
    fn __init__(out self, x: Int, y: Int, z: Int):
        self.x = x
        self.y = y
        self.z = z


fn tiled_register_matmul[
        dtype: DType, A_layout: Layout, B_layout: Layout, C_layout: Layout,
        BM: Int, BK: Int, BN: Int, TM: Int, COMPUTE_THREADS: Int,
        NUM_THREADS: Int, version: StaticString,
        ###
        block_idx: MockGPUIndex, thread_idx: MockGPUIndex
    ](
        A: LoggedTensor[dtype, A_layout, MutAnyOrigin],
        B: LoggedTensor[dtype, B_layout, MutAnyOrigin],
        C: LoggedTensor[dtype, C_layout, MutAnyOrigin],
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

        var A_smem = LoggedTensor[
            dtype,
            Layout.row_major(BM, BK),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        A_smem.print()

        #var B_smem = LoggedTensor[
        #    dtype,
        #    Layout.row_major(BK, BN),
        #    MutAnyOrigin,
        #    address_space = AddressSpace.SHARED,
        #].stack_allocation()

        #var dst_reg = LoggedTensor[
        #    dtype,
        #    Layout(TM),
        #    MutAnyOrigin,
        #    address_space = AddressSpace.LOCAL,
        #].stack_allocation()


fn main():
    alias M = 4096
    alias K = 6144
    alias N = 2048
    A = example_logged_tensor[M, K]("A")
    B = example_logged_tensor[K, N]("B")
    C = example_logged_tensor[M, N]("C")

    comptime OPTIMIZED_BLOCK_SIZE = 16
    comptime BM = OPTIMIZED_BLOCK_SIZE
    comptime BN = OPTIMIZED_BLOCK_SIZE
    comptime BK = OPTIMIZED_BLOCK_SIZE
    comptime TM = 4
    comptime COMPUTE_THREADS = (BM * BN) // TM
    comptime COPY_THREADS = max(BM * BK, BK * BN)
    comptime NUM_THREADS = max(COMPUTE_THREADS, COPY_THREADS)
    comptime version = 'good'

    ###

    alias block_idx = MockGPUIndex(0, 0, 0)
    alias thread_idx = MockGPUIndex(0, 0, 0)

    tiled_register_matmul[
        DType.float32,
        Layout.row_major(M, K),
        Layout.row_major(K, N),
        Layout.row_major(M, N),
        BM, BK, BN, TM, COMPUTE_THREADS, NUM_THREADS, version,
        block_idx, thread_idx
    ](A, B, C)
