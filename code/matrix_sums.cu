//连续存储读取和跳跃存储读取
#include <stdio.h>
#include "../common/common.h"
//数据元素个数
const size_t DSIZE =16384;
//块大小
const int block_size=256;
//行和
__global__ void row_sum(const float *device_a,float *device_sum,size_t DSIZE)
{
    float temp=0;
    //全局索引
    int idx=threadIdx.x + blockDim.x * blockIdx.x;
    if(idx<DSIZE)
    {
        for (size_t i = 0; i < DSIZE; i++)
        {
            temp+=device_a[i+DSIZE*idx];
        }
        device_sum[idx]=temp;
    }
    
}
//列和
__global__ void colom_sum(const float *device_a,float *device_sum,size_t DSIZE)
{
    float temp=0;
    //全局索引
    int idx=threadIdx.x + blockDim.x * blockIdx.x;
    if(idx<DSIZE)
    {
        for (size_t i = 0; i < DSIZE; i++)
        {
            temp+=device_a[i*DSIZE+idx];
        }
        device_sum[idx]=temp;
    }
    
}

int main ()
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
    //分配内存
    host_a= (float*)malloc(DSIZE*DSIZE*sizeof(float));
    host_sum=(float*)malloc(DSIZE*sizeof(float));
    if(NULL!=host_a && NULL!=host_sum)
    {
        printf("内存分配成功!\n");
    }
    else
    {
        printf("内存分配失败!\n");
        return -1;
    }
    
    //初始化元素
    for(size_t i=0; i<DSIZE*DSIZE; i++)
    {
        host_a[i] = 1.0f;
    }

    //分配GPU内存
    float *device_a,*device_sum;
    cudaMalloc((float**)&device_a,DSIZE*DSIZE*sizeof(float));
    cudaMalloc((float**)&device_sum,DSIZE*sizeof(float));
    if(device_a!=NULL && device_sum!=NULL)
    {
        printf("GPU内存分配成功!\n");
    }
    else
    {
        printf("GPU内存分配失败!\n");
        cudaFree(device_a);
        cudaFree(device_sum);
        return -1;
    }
    //主机数据传入GPU
    cudaMemcpy(device_a,host_a,DSIZE*DSIZE*sizeof(float),cudaMemcpyHostToDevice);
    //设置网格和块
    dim3 block (block_size);
    dim3 grid ((DSIZE+block_size-1)/block_size);
    //创建Event
    cudaEvent_t time1,time2,time3,time4;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    cudaEventCreate(&time3);
    cudaEventCreate(&time4);
    //标记时间
    cudaEventRecord(time1);
    //调用行计算核函数
    row_sum<<<grid,block>>>(device_a,device_sum,DSIZE);
    //同步机制
    cudaEventRecord(time2);
    cudaEventSynchronize(time2);
    //从GPU向CPU传数据
    cudaMemcpy(host_sum,device_sum,DSIZE*sizeof(float),cudaMemcpyDeviceToHost);
    //清空
    memset(host_sum,0,DSIZE*sizeof(float));
    cudaEventRecord(time3);
    //调用列计算核函数
    colom_sum<<<grid,block>>>(device_a,device_sum,DSIZE);
    //同步机制
    cudaEventRecord(time4);
    cudaEventSynchronize(time4);
     //从GPU向CPU传数据
    cudaMemcpy(host_sum,device_sum,DSIZE*sizeof(float),cudaMemcpyDeviceToHost);
    //时间计算
    float row_time,colum_time;
    cudaEventElapsedTime(&row_time,time1,time2);
    cudaEventElapsedTime(&colum_time,time3,time4);
    printf("row核函数执行时间:%f\ncloum核函数执行时间:%f\n",row_time,colum_time);
    // 释放
    cudaEventDestroy(time1);
    cudaEventDestroy(time2);
    cudaEventDestroy(time3);
    cudaEventDestroy(time4);
    //释放内存
    free(host_a);
    free(host_sum);
    cudaFree(device_a);
    cudaFree(device_sum);
    cudaDeviceReset();
    return 0;
}