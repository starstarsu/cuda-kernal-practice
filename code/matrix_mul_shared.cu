//使用共享内存的矩阵乘法计算
#include "common/common.h"
#include <stdio.h>

const int block_size=32;
__global__ void matrix_mul_shared(float*Device_A,float*Device_B,float*Device_C,int Data_Size)
{
    //定义共享内存
    __shared__ float Shared_A[block_size][block_size];
    __shared__ float Shared_B[block_size][block_size];
    //定义x方向和y方向的索引
    int idx=threadIdx.x + blockDim.x * blockIdx.x;
    int idy=threadIdx.y + blockDim.y * blockIdx.y;
    //设定线程执行界限x维和y维不超范围
    if((idx<Data_Size) && (idy<Data_Size))
    {
        float temp=0;
        for(int i=0; i<Data_Size/blockDim.x; i++)
        {
            //拷贝数据到共享内存
            Shared_A[threadIdx.y][threadIdx.x]=Device_A[idy * Data_Size + (i * blockDim.x + threadIdx.x)];//行号*矩阵宽度+列号 按行取值
            Shared_B[threadIdx.y][threadIdx.x]=Device_B[(i * blockDim.x + threadIdx.y) * Data_Size + idx];//按列取
            //等待数据拷贝完成
            __syncthreads();

            for(int k=0; k<blockDim.x; k++)
            {
                temp+=Shared_A[threadIdx.y][k] * Shared_B[k][threadIdx.x];
            }
            
        }
        Device_C[idy*Data_Size+idx]=temp;
    }
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
    //设置元素个数
    int Data_Size=8192;//8K
    //定义时间变量
    clock_t t0,t1,t2;
    //初始化主机数据容器
    float *host_a,*host_b,*host_c;
    //记录启动时间
    t0=clock();
    //分配主机数据内存
    host_a=(float *)malloc(Data_Size*Data_Size*sizeof(float));
    host_b=(float *)malloc(Data_Size*Data_Size*sizeof(float));
    host_c=(float *)malloc(Data_Size*Data_Size*sizeof(float));
    //初始化数据
    for(int i=0 ; i<Data_Size*Data_Size; i++)
    {
        host_a[i]=3;
        host_b[i]=2;
    }
    //记录主机从分配内存到初始化数据完成时间
    t1=clock();
    //初始化设备数据容器
    float *device_A,*device_B,*device_C;
    //分配设备数据内存
    cudaMalloc(&device_A,Data_Size*Data_Size*sizeof(float));
    cudaMalloc(&device_B,Data_Size*Data_Size*sizeof(float));
    cudaMalloc(&device_C,Data_Size*Data_Size*sizeof(float));
    //检查设备内存是否分配成功
    if(device_A!=NULL && device_B!=NULL && device_C!=NULL)
    {
        printf("GPU内存分配成功!\n");
    }
    else
    {
        printf("GPU内存分配失败!\n");
        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
        return -1;
    }
    //从主机拷贝数据到设备
    cudaMemcpy(device_A,host_a,Data_Size*Data_Size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(device_B,host_b,Data_Size*Data_Size*sizeof(float),cudaMemcpyHostToDevice);
    //设置内核函数参数
    dim3 block(block_size,block_size);  //设置块为二维（32,32）
    dim3 grid((Data_Size+block.x-1)/block.x,(Data_Size+block.y-1)/block.y);   //分别从x维和y维计算需要多少块能包含所有数据
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //调用内核函数
    matrix_mul_shared<<<grid,block>>>(device_A,device_B,device_C,Data_Size);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    //计算核函数执行时间
    float kernal_time=0;
    cudaEventElapsedTime(&kernal_time,start,stop);
    //获取计算结果值
    cudaMemcpy(host_c,device_C,Data_Size*Data_Size*sizeof(float),cudaMemcpyDeviceToHost);
    //记录运算完成并从设备拷贝数据到主机时间
    t2=clock();
    //输出结果
    for(int i=0 ;i<9 ;i++)
    {
        printf("host_a:%f   host_b:%f   host_c:%f\n",host_a[i],host_b[i],host_c[i]);
    }
    //通过计算时钟周期数再变换成s
    printf("t0-t1:%f\nt0-t2:%f\n核函数执行时间:%f\n",(double)(t1-t0)/CLOCKS_PER_SEC,(double)(t2-t0)/CLOCKS_PER_SEC,kernal_time);
    //释放主机内存
    free(host_a);
    free(host_b);
    free(host_c);
    //释放设备内存
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    
    return 0;
    
}