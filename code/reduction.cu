//归约算法执行时间
#include "../common/common.h"
#include <stdio.h>

//设置元素大小64位无符号长整形，在计算时防止溢出
const size_t DSIZE=8ULL*1024ULL*1024ULL;
const int BLOCK_SIZE=256;

//原子加法
__global__ void atomic_add(const float *device_a,float *device_sum,size_t DSIZE)
{
    //全局线程索引
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if(idx<DSIZE)
    {
        atomicAdd(device_sum,device_a[idx]);
    }
}
//并行归约
__global__ void parallel(const float *device_a,float *device_sum,size_t DSIZE)
{
    //创建共享内存,每个线程块都有自己的共享内存
    __shared__ float share_data[BLOCK_SIZE];
    int tid=threadIdx.x;
    size_t idx=threadIdx.x+blockDim.x*blockIdx.x;
    //把数据加载到共享内存,利用线程网格循环，只是当前的DSIZE的数据大小刚好嫩被网格块参数覆盖所有数据
    while(idx<DSIZE)
    {
        share_data[tid]+=device_a[idx];
        idx+=gridDim.x*blockDim.x;
    }

    //使用折半归约
    for(unsigned int i=blockDim.x;i>0;i=i/2)
    {
        //执行线程同步
        __syncthreads();
        if(tid<i)
        {
            share_data[tid]+=share_data[tid+i];
            
        }
    }
    if(tid==0)
    {
        atomicAdd(device_sum,share_data[0]);
    }
}
//线程束洗牌归约
__global__ void WarpShuffle(const float *device_a,float *device_sum,size_t DSIZE)
{
    //共享内存定义为线程束大小
    __shared__ float share_data[32];
    //线程块中的id
    int tid = threadIdx.x;
    //全局id
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    //线程的临时值
    float temp_val = 0.0f;
    //在线程束中的线程id
    int lane=threadIdx.x % warpSize;
    //在线程块中的线程束id
    int warpID=threadIdx.x / warpSize;
    //线程束同步掩码
    unsigned int mask = 0xffffffff;
    //使用线程网格循环把数据拷贝存储到共享内存
    while (idx<DSIZE)
    {
        temp_val+=device_a[idx];
        idx+=gridDim.x*blockDim.x;
    }
    //线程束洗牌归约
    for (int offset = warpSize/2; offset > 0; offset=offset/2)
    {
        temp_val+=__shfl_down_sync(mask ,temp_val ,offset);
    }
    //每个线程束计算后的结果在lane=0的位置，计算完后存储到共享内存内
    if(lane==0)
    {
        share_data[warpID] = temp_val;
    }
    __syncthreads();
    //在第0个线程束内读取存入线程束内第一个线程的值，超出 范围的设置为0
    if(warpID==0)
    {
        temp_val=(tid < blockDim.x / warpSize) ? share_data[lane] : 0;
    }
    //线程束归约
    for (int offset = warpSize/2; offset > 0; offset = offset / 2)
    {
        temp_val += __shfl_down_sync(mask,temp_val,offset);
    }
    if(tid ==0)
    {
        atomicAdd(device_sum,temp_val);
    }
}

int main(void)
{
    //查询GPU设备
    int Device_NUM=0;
    cudaError_t error=ErrorCheck(cudaGetDeviceCount(&Device_NUM),__FILE__,__LINE__);
    if(error!=cudaSuccess ||Device_NUM==0)
    {
        printf("无GPU设备\n");
        return -1;
    }
    //设置GPU设备
    int device=0;
    error=ErrorCheck(cudaSetDevice(device),__FILE__,__LINE__);
    if(error!=cudaSuccess)
    {
        printf("GPU设备设置失败!\n");
        return -1;
    }
    else
    {
        printf("设置GPU成功!\n");
    }
    //定义host数据容器
    float *host_a,*host_sum;
    //定义device数据容器
    float *device_a,*device_sum;
    //分配host内存
    host_a=(float*)malloc(DSIZE*sizeof(float));
    host_sum=(float*)malloc(sizeof(float));
    //初始化host数据
    for(int i=0;i<DSIZE;i++)
    {
        host_a[i]=1;
    }
    //分配device内存
    cudaMalloc(&device_a,DSIZE*sizeof(float));
    cudaMalloc(&device_sum,sizeof(float));
    //从host拷贝数据到device
    cudaMemcpy(device_a,host_a,DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    
    float atomic_add_time,parallel_time,WarpShuffle_time;
    cudaEvent_t star1,stop1,star2,stop2,star3,stop3;
    cudaEventCreate(&star1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&star2);
    cudaEventCreate(&stop2);
    cudaEventCreate(&star3);
    cudaEventCreate(&stop3);
    //第一次时间计算
    cudaEventRecord(star1);
    //调用原子加法内核函数
    atomic_add<<<(DSIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(device_a,device_sum,DSIZE);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&atomic_add_time,star1,stop1);
    //从device拷贝计算结果到host
    cudaMemcpy(host_sum,device_sum,sizeof(float),cudaMemcpyDeviceToHost);
    //清除计算数据
    cudaMemset(device_sum,0,sizeof(float));

    //第二次时间计算
    cudaEventRecord(star2);
    //调用并行归约内核函数
    parallel<<<(DSIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(device_a,device_sum,DSIZE);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&parallel_time,star2,stop2);
    //从device拷贝计算结果到host
    cudaMemcpy(host_sum,device_sum,sizeof(float),cudaMemcpyDeviceToHost);
    //清除计算数据
    cudaMemset(device_sum,0,sizeof(float));

    //第三次时间计算
    cudaEventRecord(star3);
    //调用洗牌归约内核函数
    WarpShuffle<<<(DSIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(device_a,device_sum,DSIZE);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&WarpShuffle_time,star3,stop3);
    //从device拷贝计算结果到host
    cudaMemcpy(host_sum,device_sum,sizeof(float),cudaMemcpyDeviceToHost);
    //清除计算数据
    cudaMemset(device_sum,0,sizeof(float));
    //时间计算
    printf("原子加法:%f\n并行故归约:%f\n线程束归约:%f\n",atomic_add_time,parallel_time,WarpShuffle_time);

    //释放内存
    free(host_a);
    free(host_sum);

    cudaFree(device_a);
    cudaFree(device_sum);

    cudaDeviceReset();
    return 0;
}