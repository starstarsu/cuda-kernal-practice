//1维共享内存计算
#include <stdio.h>
#include <time.h>
#include "common/common.h"
__global__ void stencil_1d(int *in,int *out,int radius)
{
    //共享内存
    extern __shared__ int temp[];   //传入数据为 ***-----***，每个块前后都有长度为radius的数据，记录了前一块和后一块的部分数据，取消了块与块之间的通信
    //线程全局id索引
    int global_idx=threadIdx.x+blockDim.x*blockIdx.x;
    
    int block_idx=threadIdx.x+radius;
    temp[block_idx]=in[global_idx];    //此时传的数据为被分到每个块的数据
    //把前后块的数据传入共享内存
    if(threadIdx.x<radius)
    {
        temp[block_idx-radius]=in[global_idx-radius];//传入的为上一块的后radius个数据
        temp[block_idx+blockDim.x]=in[global_idx+blockDim.x];//传入的为下一块的前radius个数据
    }
    //同步机制
    __syncthreads();
    int result=0;
    for (int i =-radius; i <= radius; i++)
    {
        //窗口计算
        result+=temp[block_idx+i];
    }
    out[global_idx]=result;

}


int main(void)
{
   //GPU设备数量
    int Device_num=0;
    //查询GPU设备数量
    cudaError_t error=ErrorCheck(cudaGetDeviceCount(&Device_num),__FILE__,__LINE__);
    if(error!=cudaSuccess || Device_num==0)
    {
        printf("GPU设备查询失败!\n");
        return -1;
    }
    else
    {
        printf("当前GPU设备数量:%d\n",Device_num);
    }
    //设置GPU设备
    int Device=0;
    error=cudaSetDevice(Device);
    if(error != cudaSuccess)
    {
        printf("设置GPU设备失败!\n");
        return -1;
    }
    else
    {
        printf("设置GPU设备成功!\n");
    }
    //定义元素个数
    int elment_num=4096;
    int radius=3;
    int size=2*radius+elment_num;
    //初始化主机容器
    int *host_in,*host_out;
    //分配主机内存
    host_in=(int *)malloc(size*sizeof(int));
    host_out=(int *)malloc(size*sizeof(int));
    //初始化数据
    for(int i=0 ; i<size; i++)
    {
        host_in[i]=3;
        host_out[i]=0;
    }
    //初始化设备容器
    int *device_in,*device_out;
    //分配设备内存
    cudaMalloc((void**)&device_in,size*sizeof(int));
    cudaMalloc((void**)&device_out,size*sizeof(int));
    //检查设备内存是否分配成功
    if(device_in!=NULL && device_out!=NULL)
    {
        printf("设备内存分配成功!\n");
    }
    else
    {
        printf("设备内存分配失败!");
        cudaFree(device_in);
        cudaFree(device_out);
        return -1;
    }
    //从主机向设备拷贝数据
    cudaMemcpy(device_in,host_in,size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(device_out,host_out,size*sizeof(int),cudaMemcpyHostToDevice);
    //设置核函数参数
    dim3 block(16);
    dim3 grid((elment_num+block.x-1)/block.x);
    //调用核函数
    stencil_1d<<<grid,block,block.x+2*radius>>>(device_in+radius,device_out+radius,radius);
    //从设备拷贝数据到主机
    cudaMemcpy(host_out,device_out,size*sizeof(int),cudaMemcpyDeviceToHost);

    //输出结果验证
    for (int i = 0; i < size; i++) 
    {
        if (i<radius || i>=elment_num+radius){
            //数据的最前radius个数据和最后radius个数据，在计算时并没有传值，看是否是0
            if (host_out[i] != 0)
                printf("Mismatch at index %d, was: %d, should be: %d\n", i, host_out[i], 0);
        } 
        else 
        {
            //每个数据为3，窗口大小为7判断结果是否正确
            if (host_out[i] !=3*7)
                printf("Mismatch at index %d, was: %d, should be: %d\n", i, host_out[i], 21);
        }
    }
    
    //释放主机内存
    free(host_in);
    free(host_out);
    //释放设备内存
    cudaFree(device_in);
    cudaFree(device_out);


    cudaDeviceReset();
    return 0;




}

