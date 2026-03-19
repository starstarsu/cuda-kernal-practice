#include<sys/time.h>
#include<cuda_runtime.h>
#include<stdio.h>
cudaError_t ErrorCheck(cudaError_t status,const char* filename,int lineNumber)
{
    if(status!=cudaSuccess)
    {
        printf("CUDA API error:\ncode=%d,name=%s,description=%s\nfile=%s,line=%d\n",
        status,cudaGetErrorName(status),cudaGetErrorString(status),filename,lineNumber);
        return status;
    }
    return status;
}