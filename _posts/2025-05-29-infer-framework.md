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