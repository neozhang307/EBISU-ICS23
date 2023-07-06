#include "stdio.h"

class PrintHelper
{
public:
    PrintHelper(int verbase);
    void PrintPtx(int ptx);
    void PrintDataType(int dataSize);
    void PrintAsync(bool async);
    void PrintDomain(int width_x, int width_y);
    void PrintDomain(int width_x, int width_y, int height_z);
    void PrintIteration(int iteration);
    void PrintDepth(int depth);
    void PrintBlockDim(int bdim);
    void PrintILP(int ilp);
    void PrintValidThreadBlockTiling(int validtb);
    void PrintGridDim(int gdimx, int gdimy, int gdimz);
    void PrintBlockPerSM(double blkpsm, int setBlckpsm);
    void PrintSharedMemory(double assigned, double smemSize);
    void PrintSmRange(int smRange, int smRangeRequest);
    // void PrintRunTime (float elapsedTimed);
    void PrintPerformance(int width_x, int width_y, int iteration, int halo, int flopspercell, float elapsedTimed);
    void PrintPerformance(int width_x, int width_y,int height_z, int iteration, int halo, int flopspercell, float elapsedTimed);
    void PirntFinish();
    // void PrintAll(int ptx, int dataSize, bool async, int width_x, int width_y, int iteration, int bdim, int validtb, int gdimx, int gdimy, int gdimz, double blkpsm, double smemSize, int smRange, int smRangeRequest, float elapsedTimed);
private:
    int verbase = 0;
};

PrintHelper::PrintHelper(int verbase)
{
    this->verbase = verbase;
}

void PrintHelper::PrintPtx(int ptx)
{
    if (verbase > 0)
    {
        printf("[FORMA] PTX version: %d\n", ptx);
    }
    else
    {
        printf("%d\t", ptx);
    }
}

void PrintHelper::PrintDataType(int dataSize)
{
    if (verbase > 0)
    {
        if (dataSize == 8)
            printf("[FORMA] Data type: double\n");
        else
            printf("[FORMA] Data type: float\n");
    }
    else
    {
        printf("%d\t", dataSize / 4);
    }
}

void PrintHelper::PrintAsync(bool async)
{
    if (verbase > 0)
    {
        if (async)
            printf("[FORMA] Async copy: true\n");
        else
            printf("[FORMA] Async copy: false\n");
    }
    else
    {
        printf("%d\t", (int)async);
    }
}

void PrintHelper::PrintDomain(int width_x, int width_y)
{
    if (verbase > 0)
    {
        printf("[FORMA] Domain: %d x %d\n", width_x, width_y);
    }
    else
    {
        printf("%d\t%d\t", width_x, width_y);
    }
}

void PrintHelper::PrintDomain(int width_x, int width_y, int height_z)
{
    if (verbase > 0)
    {
        printf("[FORMA] Domain: %d x %d x %d\n", width_x, width_y, height_z);
    }
    else
    {
        printf("%d\t%d\t%d\t", width_x, width_y, height_z
        );
    }
}

void PrintHelper::PrintIteration(int iteration)
{
    if (verbase > 0)
    {
        printf("[FORMA] Iteration: %d\n", iteration);
    }
    else
    {
        printf("%d\t", iteration);
    }
}

void PrintHelper::PrintDepth(int depth)
{
    if (verbase > 0)
    {
        printf("[FORMA] Depth: %d\n", depth);
    }
    else
    {
        printf("%d\t", depth);
    }
}
void PrintHelper::PrintBlockDim(int bdim)
{
    if (verbase > 0)
    {
        printf("[FORMA] BlockDim: %d\n", bdim);
    }
    else
    {
        printf("%d\t", bdim);
    }
}

void PrintHelper::PrintILP(int ilp)
{
    if (verbase > 0)
    {
        printf("[FORMA] Instruction Level Parallelism: %d\n", ilp);
    }
    else
    {
        printf("%d\t", ilp);
    }
}


void PrintHelper::PrintValidThreadBlockTiling(int validtb)
{
    if (verbase > 0)
    {
        printf("[FORMA] Valid thread block tiling: %d\n", validtb);
    }
    else
    {
        printf("%d\t", validtb);
    }
}

void PrintHelper::PrintGridDim(int gdimx, int gdimy, int gdimz)
{
    if (verbase > 0)
    {
        printf("[FORMA] GridDim: %d x %d x %d\n", gdimx, gdimy, gdimz);
    }
    else
    {
        printf("<%d,%d,%d>\t", gdimx, gdimy, gdimz);
    }
}

void PrintHelper::PrintBlockPerSM(double blkpsm, int setBlckpsm)
{
    if (verbase > 0)
    {
        printf("[FORMA] Block per SM (smaller due to workload not well assigned): %f \n", blkpsm);
        printf("[FORMA] Request Block per SM: %d \n", setBlckpsm);
    }
    else
    {
        printf("%.3f/%d\t", blkpsm, setBlckpsm);
    }
}
void PrintHelper::PrintSharedMemory(double assigned, double smemSize)
{
    if (verbase > 0)
    {
        printf("[FORMA] Shared memory: %f/%f KB\n", assigned,smemSize);
    }
    else
    {
        printf("%.3f/%.3f\t", assigned, smemSize);
    }
}

void PrintHelper::PrintSmRange(int smRange, int smRangeRequest)
{
    if (verbase > 0)
    {
        printf("[FORMA] SM range (might larger due to using circular queue): %d \n", smRange);
        printf("[FORMA] Request SM range: %d \n", smRangeRequest);
    }
    else
    {
        printf("%d/%d\t", smRange, smRangeRequest);
    }
}

void PrintHelper::PrintPerformance(int width_x, int width_y, int iteration, int halo, int flopspercell, float elapsedTimed)
{
    if (verbase > 0)
    {
        printf("[FORMA] Performance (ms) : %f \n", elapsedTimed);
        printf("[FORMA] Performance (GCells/s) : %f \n", (double)(width_x) * (width_y)*iteration / elapsedTimed / 1000 / 1000);
        printf("[FORMA] Valid Performance (GCells/s): %f \n", (double)(width_x - 2 * halo) * (width_y - 2 * halo) * iteration / elapsedTimed / 1000 / 1000);
    }
    else
    {
        printf("%f\t", (double)elapsedTimed);
        printf("%f\t", (double)(width_x - 2 * halo) * (width_y - 2 * halo) * iteration / elapsedTimed / 1000 / 1000);
    }
}
void PrintHelper::PrintPerformance(int width_x, int width_y, int height_z, int iteration, int halo, int flopspercell, float elapsedTimed)
{
    if (verbase > 0)
    {
        printf("[FORMA] Performance (ms) : %f \n", elapsedTimed);
        printf("[FORMA] Performance (GCells/s) : %f \n", (double)(width_x) * (width_y)*(height_z)*iteration / elapsedTimed / 1000 / 1000);
        printf("[FORMA] Valid Performance (GCells/s): %f \n", (double)(width_x - 2 * halo) * (width_y - 2 * halo) * (height_z - 2 * halo) * iteration / elapsedTimed / 1000 / 1000);
    }
    else
    {
        printf("%f\t", (double)elapsedTimed);
        printf("%f\t", (double)(width_x - 2 * halo) * (width_y - 2 * halo) *  (height_z - 2 * halo) * iteration / elapsedTimed / 1000 / 1000);
    }
}
void PrintHelper::PirntFinish()
{
    if (verbase > 0)
    {
        printf("[FORMA] Finish\n");
    }
    else
    {
        printf("\n");
    }
}