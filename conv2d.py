import IPython
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

# torch version
import torch

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)

print(conv_torch)
# mistaken code, because its initial elements are random
# conv2 = np.empty((N, CO, OUT_H, OUT_W),dtype=np.int64)
conv2 = np.zeros((N, CO, OUT_H, OUT_W), dtype=np.int64)

for n in range(N):
  for ot in range(CO):
    for it in range(CI):
      for i in range(OUT_H):
        for j in range(OUT_W):
          for di in range(K):
            for dj in range(K):
              conv2[n,ot,i,j]+=data[n,it,i+di,j+dj]*weight[ot,it,di,dj]


print("---")
print(conv2)
np.testing.assert_allclose(conv2, conv_torch, rtol=1e-5)

@tvm.script.ir_module
class MyConv:
  @T.prim_func
  def conv(A: T.Buffer((N,CI,H,W),"int64"),W: T.Buffer((CO,CI,K,K),"int64"),C: T.Buffer((N,CO,OUT_H,OUT_W),"int64")):
    T.func_attr({"global_symbol": "conv", "tir.noalias": True})
    for n,ot,it,i,j,di,dj in T.grid(N,CO,CI,OUT_H,OUT_W,K,K):
      with T.block("C"):
        vn,vot,vit,vi,vj,vdi,vdj = T.axis.remap("SSSSSSS",[n,ot,it,i,j,di,dj])
        C[vn,vot,vi,vj] += A[vn,vit,vi+vdi,vj+vdj]* W[vot,vit,vdi,vdj]

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.zeros((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)