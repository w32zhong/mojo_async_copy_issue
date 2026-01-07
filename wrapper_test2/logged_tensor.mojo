from random import random_si64
from layout import LayoutTensor, Layout
from gpu.memory import AddressSpace


fn _get_address_space_name[addr: AddressSpace]() -> String:
    @parameter
    if addr == AddressSpace.GENERIC: return "AddressSpace.GENERIC"
    @parameter
    if addr == AddressSpace.GLOBAL: return "AddressSpace.GLOBAL"
    @parameter
    if addr == AddressSpace.SHARED: return "AddressSpace.SHARED"
    @parameter
    if addr == AddressSpace.LOCAL: return "AddressSpace.LOCAL"
    return "AddressSpace(Unknown)"


struct LoggedTensor[
    mut: Bool,
    //,
    dtype: DType,
    layout: Layout,
    origin: Origin[mut=mut],
    /,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    layout_int_type: DType = DType.int32,
    linear_idx_type: DType = DType.int32,
    masked: Bool = False,
]:
    alias ImplType = LayoutTensor[
        dtype, layout, origin,
        address_space=address_space,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
    ]

    alias OriginCastType[
        mut: Bool,
        origin: Origin[mut=mut],
    ] = LoggedTensor[
        Self.dtype,
        Self.layout,
        origin,
        address_space = Self.address_space,
        layout_int_type = Self.layout_int_type,
        linear_idx_type = Self.linear_idx_type,
        masked = Self.masked,
    ]
    alias _AsMut = Self.OriginCastType[True, _]

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

    fn print(read self) raises:
        max_rows = min(self.impl.shape[0](), 100)
        max_cols = min(self.impl.shape[1](), 30)
        for row in range(max_rows):
            print("{}@[{}, {}]".format(
                self.name, self.origin_x, self.origin_y
            ), end=": ")
            for col in range(max_cols):
                print(
                    String(self.impl[row, col]).rjust(5),
                    end=""
                )
            print()

    fn dim[idx: Int](self) -> Int:
        return self.impl.dim[idx]()

    @always_inline
    @staticmethod
    fn stack_allocation[
        *, stack_alignment: Int = Self.ImplType.alignment
    ]() -> LoggedTensor[
            dtype,
            layout,
            MutAnyOrigin,
            address_space=address_space,
            layout_int_type=layout_int_type,
            linear_idx_type=linear_idx_type,
            masked=masked
        ]:
        return LoggedTensor(
            Self.ImplType.stack_allocation[stack_alignment=stack_alignment](),
            _get_address_space_name[address_space]()
        )

    @always_inline
    fn tile[
        *tile_sizes: Int
    ](self, x: Int, y: Int) raises -> LoggedTensor[
        dtype,
        Self.ImplType.TileType[*tile_sizes].layout, 
        origin,
        address_space=Self.ImplType.TileType[*tile_sizes].address_space,
        layout_int_type=Self.ImplType.TileType[*tile_sizes].layout_int_type,
        linear_idx_type=Self.ImplType.TileType[*tile_sizes].linear_idx_type,
        masked=Self.ImplType.TileType[*tile_sizes].masked,
    ]:
        var tiled_view = self.impl.tile[*tile_sizes](x, y)
        var new_name = "{}.tile({}, {})".format(self.name, x, y)

        var new_origin_x = self.origin_x + x * tile_sizes[0]
        var new_origin_y = self.origin_y + y * tile_sizes[1]

        alias NewT = Self.ImplType.TileType[*tile_sizes]
        return LoggedTensor[
            dtype,
            NewT.layout, 
            origin,
            address_space=NewT.address_space,
            layout_int_type=NewT.layout_int_type,
            linear_idx_type=NewT.linear_idx_type,
            masked=NewT.masked,
        ](tiled_view, new_name, new_origin_x, new_origin_y)

    fn copy_from(mut self: Self._AsMut, other: LoggedTensor):
        return self.impl.copy_from(other.impl)

    fn __getitem__(self, x: Int, y: Int) -> Self.ImplType.element_type:
        return self.impl[x, y]

    fn __getitem__(self, x: Int) -> Self.ImplType.element_type:
        return self.impl[x, 0]

    fn __setitem__(self, x: Int, val: Self.ImplType.element_type) where Self.mut:
        self.impl[x, 0] = val


fn example_logged_tensor[
        rows: Int, cols: Int
    ](name: String) -> LoggedTensor[
        DType.float32,
        Layout.row_major(rows, cols),
        MutAnyOrigin,
        address_space=AddressSpace.GENERIC,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=False
    ]:
    comptime buf_size = rows * cols
    var ptr = alloc[Float32](buf_size)
    for i in range(buf_size):
        ptr[i] = Float32(random_si64(-5, 5))
    comptime layout = Layout.row_major(rows, cols)
    var tensor = LayoutTensor[
        DType.float32,
        layout,
        MutAnyOrigin,
        address_space=AddressSpace.GENERIC,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=False
    ](UnsafePointer[Float32, MutAnyOrigin](ptr))
    return LoggedTensor(tensor, name)


from math import ceildiv

struct MockGPUIndex:
    var x: Int
    var y: Int
    var z: Int
    fn __init__(out self, x: Int, y: Int, z: Int):
        self.x = x
        self.y = y
        self.z = z

fn barrier():
    return

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
    ) raises:
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

        var B_smem = LoggedTensor[
            dtype,
            Layout.row_major(BK, BN),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var dst_reg = LoggedTensor[
            dtype,
            Layout.row_major(TM, 1),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        var dst_subtile = C.tile[BM, BN](block_idx.y, block_idx.x)
                           .tile[TM, 1](subtile_row, subtile_col)

        dst_reg.copy_from(dst_subtile)

        barrier()

        for block in range(ceildiv(K, BK)):
            comptime A_tile_layout = Layout.row_major(BM, BK)
            comptime B_tile_layout = Layout.row_major(BK, BN)

            var A_tile = A.tile[BM, BK](block_idx.y, block)
            var B_tile = B.tile[BK, BN](block, block_idx.x)

            A_smem.copy_from(A_tile)
            B_smem.copy_from(B_tile)

            barrier()

            for k in range(BK):
                var A_subtile = A_smem.tile[TM, 1](subtile_row, k)
                var B_subtile = B_smem.tile[1, BN](k, 0)
                var B_element = B_subtile[0, subtile_col]

                for t in range(TM):
                    product = A_subtile[t, 0] * B_element
                    print(dst_reg[t])
                    print(product)
                    dst_reg[t] += product
                    print(dst_reg[t])
                    break
                break
            break


fn main() raises:
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

    alias block_idx = MockGPUIndex(1, 0, 0)
    alias thread_idx = MockGPUIndex(2, 0, 0)

    tiled_register_matmul[
        DType.float32,
        Layout.row_major(M, K),
        Layout.row_major(K, N),
        Layout.row_major(M, N),
        BM, BK, BN, TM, COMPUTE_THREADS, NUM_THREADS, version,
        block_idx, thread_idx
    ](A, B, C)
