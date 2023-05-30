#include <tuple>
#include <cuda.h>
#include <cooperative_groups.h>
// Simple way to change from traditional launch to cooperative launch
// Simple way to do warmup
// Simple way to do iterative kernel launch.
// Base class with virtual function of launch and warmup
class launchHelper
{
private:
    /* data */
    int warmupaim = 350;//350ms
    int warmupiteration=1000;
public:
    launchHelper(int warmupaim = 350, int warmupiteration=1000);
    ~launchHelper();
    template <typename Kernel_t, typename... Args> void launch(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args... args);
    template <typename Kernel_t, typename... Args> void warmup(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args&&... args);
    template <typename Kernel_t, typename Function, typename... Args> 
        void warmup(Kernel_t kernel, Function func, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream,  Args&&... args);
   
};
launchHelper::launchHelper(int givenwarmupsec, int givenwarmupiteration)
{
    this->warmupaim=givenwarmupsec;
    this->warmupiteration=givenwarmupiteration;
}
launchHelper::~launchHelper()
{
}
template<typename Kernel_t, typename... Args>
void launchHelper::launch(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args... args)
{
    kernel<<<gdim, bdim, executesm, stream>>>(std::forward<Args>(args)...);
}
template<typename Kernel_t, typename... Args>
void launchHelper::warmup(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args&&...  args)
{
    cudaEvent_t warstart, warmstop;
    cudaEventCreate(&warstart);
    cudaEventCreate(&warmstop);
    cudaEventRecord(warstart, 0);
    for (int i = 0; i < warmupiteration; i++)
    {
        launch(kernel,gdim, bdim, executesm, stream,std::forward<Args>(args)...);
    }
    // launch(std::forward<Args>(args)...);
    cudaEventRecord(warmstop, 0);
    cudaEventSynchronize(warmstop);
    float warmelapsedTime;
    cudaEventElapsedTime(&warmelapsedTime, warstart, warmstop);
    int nowwarmup = warmelapsedTime;
    int nowiter = (warmupaim + nowwarmup - 1) / nowwarmup;
    for (int out = 0; out < nowiter; out++)
    {
        for (int i = 0; i < warmupiteration; i++)
        {
            // launch(std::forward<Args>(args)...);
            launch(kernel,gdim, bdim, executesm, stream,std::forward<Args>(args)...);

        }
    }
    cudaEventDestroy(warstart);
    cudaEventDestroy(warmstop);
}

template <typename Kernel_t, typename Function, typename... Args> 
void launchHelper::warmup(Kernel_t kernel,Function func,  dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args&&...  args)
{
    cudaEvent_t warstart, warmstop;
    cudaEventCreate(&warstart);
    cudaEventCreate(&warmstop);
    cudaEventRecord(warstart, 0);
    for (int i = 0; i < warmupiteration; i++)
    {
        launch(kernel,gdim, bdim, executesm, stream,std::forward<Args>(args)...);
        func(std::forward<Args>(args)...);
    }
    // launch(std::forward<Args>(args)...);
    cudaEventRecord(warmstop, 0);
    cudaEventSynchronize(warmstop);
    float warmelapsedTime;
    cudaEventElapsedTime(&warmelapsedTime, warstart, warmstop);
    int nowwarmup = warmelapsedTime;
    int nowiter = (warmupaim + nowwarmup - 1) / nowwarmup;
    for (int out = 0; out < nowiter; out++)
    {
        for (int i = 0; i < warmupiteration; i++)
        {
            // launch(std::forward<Args>(args)...);
            launch(kernel,gdim, bdim, executesm, stream,std::forward<Args>(args)...);
            func(std::forward<Args>(args)...);
        }
    }
    cudaEventDestroy(warstart);
    cudaEventDestroy(warmstop);
}