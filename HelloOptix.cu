#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__

#include "CommonOptix.h"

extern "C" 
{
    __constant__ Params params;
}

extern "C"
__global__ void __raygen__HelloOptix()          // prefix __raygen__ is necessary as semantic type
{
    uint3 launch_index                          = optixGetLaunchIndex();
    RayGenData* rtData                          = (RayGenData*)optixGetSbtDataPointer();

    params.image[launch_index.y * params.image_width + launch_index.x] 
                                                = make_float4(launch_index.x, launch_index.y, launch_index.z, rtData->dummy);
}
