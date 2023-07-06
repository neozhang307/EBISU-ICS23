#include <tuple>
#include <cuda.h>
#include <cooperative_groups.h>
// Simple way to change from traditional launch to cooperative launch
// Simple way to do warmup
// Simple way to do iterative kernel launch.
// Base class with virtual function of launch and warmup
inline void for_each_argument_address(void **) {}

template <typename arg_t, typename... args_t>
inline void for_each_argument_address(void **collected_addresses,
                                      arg_t &&arg,
                                      args_t &&...args)
{
    collected_addresses[0] = const_cast<void *>(static_cast<const void *>(&arg));
    for_each_argument_address(collected_addresses + 1,
                              ::std::forward<args_t>(args)...);
}
template <bool useCooperativeLaunch = false>
class LaunchHelper
{
private:
    /* data */
    int warmupaim = 350; // 350ms
    int warmupiteration = 1000;

public:
    LaunchHelper(int warmupaim = 350, int warmupiteration = 1000);
    ~LaunchHelper();
    template <typename Kernel_t, typename... Args>
    void launch(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args &&...args);
    template <typename Kernel_t, typename... Args>
    void warmup(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args &&...args);
    template <typename Kernel_t, typename Function, typename... Args>
    void warmup(Kernel_t kernel, Function func, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args &&...args);
};
template <bool useCooperativeLaunch>
LaunchHelper<useCooperativeLaunch>::LaunchHelper(int givenwarmupsec, int givenwarmupiteration)
{
    this->warmupaim = givenwarmupsec;
    this->warmupiteration = givenwarmupiteration;
}
template <bool useCooperativeLaunch>
LaunchHelper<useCooperativeLaunch>::~LaunchHelper()
{
}
template <bool useCooperativeLaunch>
template <typename Kernel_t, typename... Args>
void LaunchHelper<useCooperativeLaunch>::launch(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args &&...args)
{
    if (useCooperativeLaunch)
    {
        constexpr const auto non_zero_num_params = sizeof...(Args) == 0 ? 1 : sizeof...(Args);
        void *argument_ptrs[non_zero_num_params];
        for_each_argument_address(argument_ptrs, std::forward<Args>(args)...);
        cudaLaunchCooperativeKernel(
            (void *)kernel, gdim, bdim, argument_ptrs, executesm, stream);
    }
    else
    {
        kernel<<<gdim, bdim, executesm, stream>>>(std::forward<Args>(args)...);
    }
}
template <bool useCooperativeLaunch>
template <typename Kernel_t, typename... Args>
void LaunchHelper<useCooperativeLaunch>::warmup(Kernel_t kernel, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args &&...args)
{
    cudaEvent_t warstart, warmstop;
    cudaEventCreate(&warstart);
    cudaEventCreate(&warmstop);
    cudaEventRecord(warstart, 0);
    for (int i = 0; i < warmupiteration; i++)
    {
        launch(kernel, gdim, bdim, executesm, stream, std::forward<Args>(args)...);
    }
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
            launch(kernel, gdim, bdim, executesm, stream, std::forward<Args>(args)...);
        }
    }
    cudaEventDestroy(warstart);
    cudaEventDestroy(warmstop);
}
template <bool useCooperativeLaunch>
template <typename Kernel_t, typename Function, typename... Args>
void LaunchHelper<useCooperativeLaunch>::warmup(Kernel_t kernel, Function func, dim3 gdim, dim3 bdim, int executesm, cudaStream_t stream, Args &&...args)
{
    cudaEvent_t warstart, warmstop;
    cudaEventCreate(&warstart);
    cudaEventCreate(&warmstop);
    cudaEventRecord(warstart, 0);
    for (int i = 0; i < warmupiteration; i++)
    {
        launch(kernel, gdim, bdim, executesm, stream, std::forward<Args>(args)...);
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
            launch(kernel, gdim, bdim, executesm, stream, std::forward<Args>(args)...);
            func(std::forward<Args>(args)...);
        }
    }
    cudaEventDestroy(warstart);
    cudaEventDestroy(warmstop);
}
