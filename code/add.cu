#include <stdio.h>
#include <time.h>
#include "common/common.h"


//核函数
__global__ void kernal(float *GPU_A,float *GPU_B,float *GPU_C,const int N)
{
    //每个线程块的id
    int id=threadIdx.x;

    if(id<N)
    {
        GPU_C[id]=GPU_A[id]+GPU_B[id];
    }
    
}
void initial_data(float* ip,int size)
{
    // 当前系统时间作为随机种子
    time_t t;
    srand((unsigned) time(&t));
    printf("Matrix is: ");
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
        printf("%.2f ",ip[i] );
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
    //设置元素大小
    int nElem=16;
    //分配内存
    size_t nBytes=nElem * sizeof(float);
    float *data_A,*data_B,*data_C;

    data_A=(float *)malloc(nBytes);
    data_B=(float *)malloc(nBytes);
    data_C=(float *)malloc(nBytes);
    if(NULL!=data_A && NULL!=data_B && NULL!=data_C)
    {
        printf("内存分配成功!\n");
    }
    else
    {
        printf("内存分配失败!\n");
        return -1;
    }
    
    //初始化元素
    initial_data(data_A,nElem);
    initial_data(data_B,nElem);
    memset(data_C,0,nBytes);

    //分配GPU内存
    float *GPU_A,*GPU_B,*GPU_C;
    cudaMalloc((float**)&GPU_A,nBytes);
    cudaMalloc((float**)&GPU_B,nBytes);
    cudaMalloc((float**)&GPU_C,nBytes);
    if(GPU_A!=NULL && GPU_B!=NULL && GPU_C!=NULL)
    {
        printf("GPU内存分配成功!\n");
    }
    else
    {
        printf("GPU内存分配失败!\n");
        free(data_A);
        free(data_B);
        free(data_C);
        return -1;
    }
    //主机数据传入GPU
    cudaMemcpy(GPU_A,data_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_B,data_B,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_C,data_C,nBytes,cudaMemcpyHostToDevice);
    //设置网格和块
    dim3 block (nElem);
    dim3 grid (1);
    //调用计算核函数
    kernal<<<grid,block>>>(GPU_A,GPU_B,GPU_C,nElem);
    //从GPU向CPU传数据
    cudaMemcpy(data_C,GPU_C,nBytes,cudaMemcpyDeviceToHost);
    //查看结果
    for(int i=0;i<nElem;i++)
    {
        printf("data_A:%f\ndata_B:%f\ndata_C:%f\n",data_A[i],data_B[i],data_C[i]);
    }
    //释放内存
    free(data_A);
    free(data_B);
    free(data_C);
    cudaFree(GPU_A);
    cudaFree(GPU_B);
    cudaFree(GPU_C);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}