from layout import *
from utils import IndexList
from gpu.memory import AddressSpace

# A proxy struct that wraps LayoutTensor and logs accesses.
struct LoggedTensor[
    dt: DType, 
    lay: Layout, 
    is_mut: Bool, 
    org: Origin[is_mut], 
    masked: Bool, 
    space: AddressSpace,
    lit: DType,
    lidt: DType
]:
    alias ImplType = LayoutTensor[
        dt, lay, org, 
        masked=masked, 
        address_space=space,
        layout_int_type=lit,
        linear_idx_type=lidt
    ]
    var impl: Self.ImplType
    var name: String

    fn __init__(out self, impl: Self.ImplType, name: String = "Tensor"):
        self.impl = impl
        self.name = name

    # --- 1D Access ---
    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[dt, Self.ImplType.element_size]:
        print("[", self.name, "READ] index:", idx)
        return self.impl[idx]

    @always_inline
    fn __setitem__(mut self, idx: Int, val: SIMD[dt, Self.ImplType.element_size]):
        print("[", self.name, "WRITE] index:", idx, "val:", val)
        self.impl[idx] = val

    # --- 2D Access ---
    @always_inline
    fn __getitem__(self, x: Int, y: Int) -> SIMD[dt, Self.ImplType.element_size]:
        var offset = self.impl._offset(x, y)
        print("[", self.name, "READ] coords: (", x, ",", y, ") -> linear offset:", offset)
        return self.impl[x, y]

    @always_inline
    fn __setitem__(mut self, x: Int, y: Int, val: SIMD[dt, Self.ImplType.element_size]):
        var offset = self.impl._offset(x, y)
        print("[", self.name, "WRITE] coords: (", x, ",", y, ") -> linear offset:", offset, "val:", val)
        self.impl[x, y] = val

    # --- Vectorized Access (SIMD) ---
    @always_inline
    fn load[width: Int](self, x: Int, y: Int) -> SIMD[dt, width]:
        var offset = self.impl._offset(x, y)
        print("[", self.name, "READ SIMD] coords: (", x, ",", y, ") -> offset:", offset, "width:", width)
        return self.impl.load[width](x, y)

    # Store not supported by LayoutTensor?
    # @always_inline
    # fn store[width: Int](mut self, x: Int, y: Int, val: SIMD[dt, width]):
    #     var offset = self.impl._offset(x, y)
    #     print("[", self.name, "WRITE SIMD] coords: (", x, ",", y, ") -> offset:", offset, "width:", width)
    #     self.impl.store(x, y, val)

    # --- Tiling Operation ---
    @always_inline
    fn tile[*tile_sizes: Int](self, x: Int, y: Int) -> LoggedTensor[
        dt, 
        Self.ImplType.TileType[*tile_sizes].layout, 
        is_mut, 
        org, 
        Self.ImplType.TileType[*tile_sizes].masked, 
        Self.ImplType.TileType[*tile_sizes].address_space,
        Self.ImplType.TileType[*tile_sizes].layout_int_type,
        Self.ImplType.TileType[*tile_sizes].linear_idx_type
    ]:
        var tiled_view = self.impl.tile[*tile_sizes](x, y)
        var new_name = self.name + ".tile(" + String(x) + "," + String(y) + ")"
        print("---", "Tiling", self.name, "extracting tile at", x, ",", y, "---")
        
        alias NewT = Self.ImplType.TileType[*tile_sizes]
        return LoggedTensor[
            dt, 
            NewT.layout, 
            is_mut, 
            org, 
            NewT.masked, 
            NewT.address_space,
            NewT.layout_int_type,
            NewT.linear_idx_type
        ](tiled_view, new_name)

    # --- Utility methods ---
    fn dim(self, idx: Int) -> Int:
        return self.impl.dim(idx)

    fn size(self) -> Int:
        return self.impl.size()

# Example Usage
fn main():
    # Setup a 4x4 row-major data buffer
    var data = List[Float32](capacity=16)
    for i in range(16): data.append(Float32(i))
    
    # Create the underlying LayoutTensor
    var base_tensor = LayoutTensor[DType.float32, Layout.row_major(4, 4), MutAnyOrigin](
        data.unsafe_ptr()
    )

    # Wrap it in our logger
    # Mojo inference should deduce all params
    var tensor = LoggedTensor(base_tensor, "MyMatrix")

    print("--- Individual Access ---")
    var val = tensor[1, 2]  # Should log access to (1, 2)
    tensor[3, 3] = 100.0    # Should log write to (3, 3)

    print("\n--- Tiled Access ---")
    # Extract a 2x2 tile from the top-right (grid coords 0, 1)
    # Note: Using (x, y) directly instead of (0, 1) as tuple due to unpacking limit
    var top_right_tile = tensor.tile[2, 2](0, 1)
    
    # Accessing the tile should log both the coordinate in the tile 
    # AND show the correct linear offset into the original memory.
    var tile_val = top_right_tile[0, 0] # Should map to original (0, 2)
    top_right_tile[1, 1] = 99.0        # Should map to original (1, 3)
