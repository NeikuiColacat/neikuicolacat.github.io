# 在写llm推理框架的时候遇到的一些有意思的thinking

## 关于sgemv算子

下面这段代码是sgemv算子的算法设计

1. 每个cuda block负责一个权重矩阵的若干行，每个block内的thread负责权重矩阵的若干列
2. 对一个block内负责的若干行进行串行计算
3. 每个thread负责4个元素的相乘相加
4. 做完之后线程同步，进行block reduce，得到1个最终的输出元素

写完之后扔给claude 4 sonnet做代码分析，提示有内存冲突问题，即sdata在ROW_PER_BLOCK > 1的时候会被多个thread公用

给claude 4 prompt了一下sdata[tid] = 0之后不再提示有冲突问题

```cpp
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void mamutl_kernel_cu_fp32(const float* input, const float* weight, float* output,
                                      int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;

  if (start_row >= K) {
    return;
  }

  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_num * pack_size;

#pragma unrool
  for (int p = start_row; p < end_row; p++) {
    sdata[tid] = 0;
    float4* input_float4_ptr = (float4*)input;
    float4* weight_float4_ptr = (float4*)(weight + p * M);
#pragma unroll
    for (int i = tid; i < pack_num; i+=blockDim.x){
      auto [input_x, input_y, input_z, input_w] = *(input_float4_ptr + i);
      auto [weight_x, weight_y, weight_z, weight_w] = *(weight_float4_ptr + i);

      float part_sum =
          input_x * weight_x + input_y * weight_y + input_z * weight_z + input_w * weight_w;
      sdata[tid] += part_sum;
    }

    for (int i = pack_off + tid; i < M; i += blockDim.x) {
      sdata[tid] += input[i] * weight[p * M + i];
    }

    __syncthreads();
    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }

    __syncthreads();
  }
}
```

## 有关于mha算子

mha蒜子的实现需要用到多个内存区域，包括kv cache ， query ， score ， output等

不熟悉的话其实写算子的时候做内存寻址定位 ，写出偏移量表达式思维负担挺大的

其实应该单独再开一个辅助类帮助定位内存地址的。只不过有点懒就算了。

实现的时候其实是统一按照 `每个block对应一个注意力头，block内每个thread负责一个timestep` 来设计的

其实应该利用block 和 thread 可以做成多个维度组合的这个特性 ， 应该可以省去很多内存寻址的麻烦

## 关于手搓argmax用于求一块内存最大值与其下标

一般来说做块内规约

blockdim / warpSize < warpSize

用这个特性就可以吧多个warp的结果规约到一个warp

```cpp
__forceinline__ __device__ void block_reduce_argmax(float& val, size_t& ptr, float* shared_value,
                                                    size_t* shared_ptr) {
  
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  warp_reduce_argmax(val, ptr);

  __syncthreads();
  if (lane_id == 0) {
    shared_value[warp_id] = val;
    shared_ptr[warp_id] = ptr;
  }

  __syncthreads();
  if (threadIdx.x < blockDim.x / warpSize) {
    val = shared_value[lane_id];
    ptr = shared_ptr[lane_id];
  } else {
    val = 0;
    ptr = SIZE_MAX;
  }

  if (warp_id == 0) {
    warp_reduce_argmax(val, ptr);
  }
}
```

## 有关于使用gpu加速排序算法

最近看了看nv的thrust库，发现并发编程其实非常适用于一类情况

即运算和数据类型构成一个带幺元的半群结构

这样的话在做规约会非常方便，有点类似于线段树做merge时候的操作

比如说排序，排序的比较操作就可以看成是一个带幺元的半群结构，在规约的时候就可以直接用warp reduce的方式来做

事实上存在双调排序之类的算法，利用gpu并发特性感觉能做很多有意思之前不敢想的事情。

再议。

## 对于 Swiglu 的一些理解

1. 原来的vec通过两个Linear层获得 a，b
2. Swiglu = a * swish(b) , 
3. 其中 * 为逐点相乘，swish(x) = x * sigmoid(x) , swish相当于一个优化的relu

## 对于 cublas 的一些细节

cublas 做 gemm 为 列主序 ， 宿主机存矩阵为行主序

使用列主序访问 矩阵相当于 做 矩阵转置 ， 利用这一个性质 加上 矩阵乘法转置性质 ， 可以无损转置

## block size 设置

 最大取1024 , 一个 warp 32 ，最好是 32 倍数

- 主流架构里，每个 Block 最大寄存器数量是 64 K
- 每个线程所能使用的最大寄存器数量是 255 个
- 开普勒到伏特架构 每个 SM 最大线程数量为 2048 , 图灵架构为 1024
- 主流架构里，每个 Block 最大寄存器数量是 64 K
- 每个线程所能使用的最大寄存器数量是 255 个

为了保证每个 thread 都有最大寄存器数量可供使用，那每个 block 最多能启动 256 个

## tail effect

当你启动一个 CUDA kernel 时，线程块会被划分成多个 wave（波次），每个 wave 包含若干个可以并发执行的线程块。前面的 wave 通常能充分利用所有 SM（Streaming Multiprocessor），但最后一个 wave 可能只包含少量线程块，导致 GPU 资源闲置。

我们应尽量避免这种情况，将 grid_size 设置为精确的一个 wave 可能也无法避免 tail effect，因为 GPU 可能不是被当前 stream 独占的，常见的如 NCCL 执行时会占用一些 SM

