import os, numpy, torch
from pathlib import Path
from max.torch import CustomOpLibrary

op_dir = os.path.abspath('./operators')

from max.graph import Graph, TensorType, DeviceRef, ops
def build_graph(session, version):
    with Graph("matmul_graph",
               input_types=[
                   TensorType(dtype=A.dtype, shape=A.shape, device=DeviceRef.from_device(device)),
                   TensorType(dtype=B.dtype, shape=B.shape, device=DeviceRef.from_device(device))
               ],
               custom_extensions=[Path(op_dir)]) as graph:
        A_value, B_value = graph.inputs
        output = ops.custom(
            name="my_matmul",
            device=DeviceRef.from_device(device),
            values=[A_value, B_value],
            out_types=[
                TensorType(dtype=A.dtype, shape=[
                        A_value.tensor.shape[0], B_value.tensor.shape[1]
                    ], device=DeviceRef.from_device(device))
            ],
            parameters={"version": version},
        )
        graph.output(output[0].tensor)
    print('loading graph...')
    return session.load(graph) # compile the graph

from max.driver import Accelerator, accelerator_count, Tensor
import torch
M, K, N = 4096, 6144, 2048
device = Accelerator()
torch_A = torch.randn(M, K)
torch_B = torch.randn(K, N)
torch_result = (torch_A @ torch_B).numpy()
A = Tensor.from_numpy(torch_A.numpy()).to(device)
B = Tensor.from_numpy(torch_B.numpy()).to(device)

from max.engine import InferenceSession
for version in ['good', 'bad']:
    session = InferenceSession(devices=[device])
    graph =  build_graph(session, version=version)
    mojo_result = graph.execute(A, B)[0].to_numpy()
    print('version:', version)
    print(mojo_result)
    assert numpy.allclose(mojo_result, torch_result, rtol=0, atol=0.005)
