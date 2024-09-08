# Homework 3
Public repository and stub/testing code for Homework 3 of 10-714.  
Python version: 3.7

## NDArray class
1. shape
2. strides
3. offset
4. device: the device used to operate real data.
5. handle: real data.
For some operations, using python to modify shape, stride and offset is enough, like `reshape()`, `permute()`, `broadcast_to()`, `__getitem__()`.   
For others, calling functions in device is needed.

## Python array operations
1. reshape: replace shape with new_shape. `strides` is generated according to new shape. The stride of last dimension is one. The stride of the last second dimension is one times shape of last dimension.
2. permute: permute shape and strides in new order.
3. broadcast_to: insert stride 0 for the dimension that is broadcasted.
4. _get_item_: change offset (+start*stride), shape(stop-start/step) and strides(stride*step).

## CPU backend
1. Compact: convert a stride-based array to a continous array: `out[cnt++] = in[strides[0]*i + strides[1]*j + strides[2]*k];`. Because we don't know how many dimensions there are, we need a index vector.
2. EwiseSetitem, ScalarSetitem: similar to compact, but reverse.
3. Other mathematic operations: after compacting.
4. ReduceMax, ReduceSum: get max or sum over one dimension.
5. Matmul: 三层循环。
6. MatmulTiled: Matrices are tiled. Tile-mul is implemented in AlignedDot by telling compiler to use vector. Matrix-mul is implemented based on AlignedDot.

## CUDA backend
1. Compact, EwiseSetitem, ScalarSetitem: one thread for one index.
2. Other mathematic operations: similar.
3. ReduceMax, ReduceSum: One thread for one reduce_size.
4. Matmul: normal implementation: One thread is responsible for one value in out matrix.
5. Shared memory matmul: Shared memory S*L; Each thread works for V*V in L*L.
