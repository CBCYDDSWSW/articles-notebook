## Reduce CUDA优化

#### 1.什么是Reduce？

`Reduce` 是一个高阶函数，它接收一个函数作为参数，这个函数接收两个参数，然后返回一个值。`Reduce` 会从左到右依次对数组中的元素进行处理，最终得到一个值。（不局限于加法）

对于CPU而言，就是一个简单的循环处理。（以下处理都以加法为例）

```c++
int reduce(int *arr,int len){
	int sum=0;
	for(int i=0;i<len;i++){
		sum+=arr[i];
	}
	return sum;
}
```

以下是GPU的原始实现：

> [!IMPORTANT]
>
> 对于待操作数组，首先加载到共享内存当中，要注意线程同步。

```c++
__global__ void reduce(int* arr, int* out,int len) {
    __shared__ int temp[32];  //每个block中的共享内存大小
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int i = bid * bdim + tid;

    if (i < len) {
        temp[tid] = arr[i];  //将数据拷贝到共享内存中
    }

    __syncthreads(); //等待所有线程都拷贝完毕

    for (int s = 1;s < bdim;s *= 2) {
        if (tid % (2 * s) == 0 && i + s < len) {
            temp[tid] += temp[tid + s];  //对数据进行归约
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[bid] = temp[0];
    }
}
```

> 在此回顾Reduce的主函数部分，加强学习CUDA训练。

```c++
#include<iostream>
#include<cuda_runtime.h>
using namespace std;

const int len = 1000;

int main() {
    int* arr = new int[len];
    int* out = new int[len];
    int* garr, * gout;

    for (int i = 0;i < len;i++) {
        arr[i] = i;
    }

    cudaMalloc((void**)&garr, sizeof(int) * len);
    cudaMalloc((void**)&gout, sizeof(int) * len);

    cudaMemcpy(garr, arr, sizeof(int) * len, cudaMemcpyHostToDevice);

    int blocksize = 32;
    int gridsize = (len + blocksize - 1) / blocksize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    reduce << <gridsize, blocksize >> > (garr, gout, len);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float seconds = 0;
    cudaEventElapsedTime(&seconds, start, stop);
    cout << "Time taken: " << seconds << "ms" << endl;

    cudaMemcpy(out, gout, sizeof(int) * len, cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0;i < gridsize;i++) {
        sum += out[i];
    }
    cout << "Sum of the array: " << sum << endl;

    cudaFree(garr);  //释放显存
    cudaFree(gout);
    delete[]arr;
    delete[]out;

    return 0;

}
```

#### 2.交错寻址优化

一个线程块中有多个Warp也就是线程束，一个warp中的线程会执行相同的指令，如果说一个warp中不同的线程需要执行不同的指令的时候，那么当某类线程执行的时候，其他的线程会被阻塞，而这就是**线程束分化**。

```
for(int s=1;s<bdim;s*=2){
	int idx = 2*s*tid;
	if((idx+s<bdim)&&(bdim*bid+s<len)){
		temp[idx]+=temp[idx+s];
	}
}
```

#### 3.bank冲突

**同一个warp**中*多个*线程访问**同一个bank**就会出现bank conflict。

**bank是啥？**他是共享内存的最小单元。

每一个bank可以同时为一个线程提供数据，相当于什么呢？这个最小的共享内存单元已经被某个线程占用，现在另外一个线程想用，这不就是要先等别人用完嘛，这也就是所谓的n路bank conflict。

```c++
for(int s=blockDim.x/2;s>0;s>>=1){
	if(tid<s){
		temp[tid]+=temp[tid+s];
	}
	__syncthreads();
}
```

#### 4.IDLE线程

解决了线程束分化、bank冲突之后，还值得考虑的一件事是目前块中线程有很多是空闲的。

那就可以想是否可以少分配一些线程去做更多的事儿？那就让它一个人做两个人的事儿。

```c++
// 修改前
__shared__ int sdata[BLOCKSIZE];
int tid = threadIdx.x;    // 线程 id (block 内)
int bid = blockIdx.x;     // block id (grid 内)
int bdim = blockDim.x;    // block 大小
int i = bid * bdim + tid; // 全局 id

// 将数据拷贝到共享内存
if (i < len)
{
    sdata[tid] = arr[i];
}

// 修改后
__shared__ int sdata[BLOCKSIZE];
int tid = threadIdx.x;    // 线程 id (block 内)
int bid = blockIdx.x;     // block id (grid 内)
int bdim = blockDim.x;    // block 大小
// 注意这里是 bdim * 2 因为我们要让一个线程干俩个线程的活
int i = bid * bdim * 2 + tid; // 全局 id

// 将数据拷贝到共享内存
if (i < len)
{
    sdata[tid] = arr[i] + arr[i + bdim];
}
```

#### 5.展开Warp

```c++
for (int s = blockDim.x / 2; s > 0; s >>= 1)
{
    if (tid < s)
    {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
//对于以上代码，可以发现会越来越小，最开始是一个block的一半，当后面s<=32的时候，实际上我们只用到了一个warp的线程，而一个warp中的线程在SIMD单元上，本来就是同步的，所以没有必要这样做。
//此时的解决的方案就是展开warp
```

如何展开？

```c++
__device__ void warp_reduce(volatile int *sdata, int tid)
{
    sdada[tid] += sdata[tid + 32];
    sdada[tid] += sdata[tid + 16];
    sdada[tid] += sdata[tid + 8];
    sdada[tid] += sdata[tid + 4];
    sdada[tid] += sdata[tid + 2];
    sdada[tid] += sdata[tid + 1];
}
```

循环中修改代码如下：

```c++
for (int s = blockDim.x / 2; s > 32; s >>= 1)
{
    if (tid < s)
    {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}

if (tid < 32)
{
    warp_reduce(sdata, tid);
}
```

当然对于循环中的代码也可以**完全展开**，这样可以减少循环的开销。

最后附上一张性能逐步优化表：

| 优化手段           | 运行时间(us) | 带宽(GB/s) | 加速比 |
| ------------------ | ------------ | ---------- | ------ |
| Baseline           | 3118.4       | 42.503     | ~      |
| 交错寻址           | 1904.4       | 73.522     | 1.64   |
| 解决 bank conflict | 1475.2       | 97.536     | 2.29   |
| 去除 idle 线程     | 758.38       | 189.78     | 4.11   |
| 展开最后一个 Warp  | 484.01       | 287.25     | 6.44   |
| 完全展开           | 477.23       | 291.77     | 6.53   |