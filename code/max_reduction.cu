//最大值归约
#include "../common/common.h"
#include <stdio.h>


const size_t DSIZE=8ULL*1024ULL*1024ULL; 
const dim3 BLOCK_SIZE= dim3(256);
const dim3 GRID_SIZE=dim3(32);


//核函数
__global__ void reduce(const float *device_a,float *device_sum,size_t DSIZE)
{
    __shared__ float share_data[BLOCK_SIZE.x];
    int tid=threadIdx.x;
    share_data[tid]=-1e30;
    size_t idx=threadIdx.x+blockDim.x*blockIdx.x;
    //网格循环算法比较共享内存的数据和device的数据哪个大就写入共享内存
    for (size_t i = idx; i < DSIZE; i+=blockDim.x*gridDim.x)
    {
        share_data[tid]=max(share_data[tid],device_a[i]);
    }
    //归约共享内存内的数据
    for (size_t s = blockDim.x/2; s > 0; s/=2)
    {
        __syncthreads();
        if(tid<s)
        {
            share_data[tid]=max(share_data[tid],share_data[s+tid]);
        }
    }
        if (tid==0)
    {
        device_sum[blockIdx.x]=share_data[0];
    }
}




int main()
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

    //分配主机内存
    float *host_a,*host_sum;

    host_a=(float*)malloc(DSIZE*sizeof(float));
    host_sum=(float*)malloc(sizeof(float));
    //检查是否分配错误
    if(host_a != NULL && host_sum != NULL)
    {
        printf("host分配内存成功!\n");
    }
    //分配设备内存
    float *device_a,*device_sum;

    cudaMalloc(&device_a,DSIZE*sizeof(float));
    cudaMalloc(&device_sum,GRID_SIZE.x*sizeof(float));
    //检查是否分配错误
    if(device_a != NULL && device_sum != NULL)
    {
        printf("device分配内存成功!\n");
    }
    //初始化数据
    for (size_t i = 0; i < DSIZE; i++)
    {
        host_a[i]=1.0;
    }
    host_a[100]=10.0;
    //从host拷贝数据到device
    cudaMemcpy(device_a,host_a,DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    //第一次调用，在块内取最大
    reduce<<<GRID_SIZE,BLOCK_SIZE>>>(device_a,device_sum,DSIZE);
    //第二次掉用，把块内最大再归约取全局最大
    reduce<<<1,BLOCK_SIZE>>>(device_sum,device_sum,GRID_SIZE.x);
    cudaDeviceSynchronize();
    //从device拷贝数据到host
    cudaMemcpy(host_sum,device_sum,sizeof(float),cudaMemcpyDeviceToHost);
    //验证
    if (*host_sum == (float)10.0)
    {
        /* code */
        printf("结果正确\n");
    }
    //释放内存
    free(host_a);
    free(host_sum);
    cudaFree(device_a);
    cudaFree(device_sum);
    cudaDeviceReset();
    return 0;

}