#include <cstdint>
#include <cfloat>
#include "optixinc.h"
#include "helper_math.h"
#include <optix_device.h>

#ifdef __INTELLISENSE__
int __float_as_int(float in);
float __int_as_float(int in);
// Add other intrinsics as needed
#endif

extern "C" {
    __constant__ LaunchParams optixLaunchParams;

    __global__ void __raygen__rg()
    {
        const uint3 launch_index = optixGetLaunchIndex();

        // Load ray origin and direction from some buffer
        float3 ray_origin = ((float3*)optixLaunchParams.rays_o)[launch_index.x];
        float3 ray_direction = ((float3*)optixLaunchParams.rays_d)[launch_index.x];

        // Trace the ray
        uint32_t i, t_as_int;
        optixTrace(
            optixLaunchParams.traversable,
            ray_origin,
            ray_direction,
            0.0f,
            optixLaunchParams.t_max,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // OPTIX_RAY_FLAG_NONE,
            0,             // SBT offset
            1,             // SBT stride
            0,             // missSBTIndex 
            i, t_as_int
        );

        // Obtain hit information (like triangle ID, hit point)
        // Process the hit information
        float t = __int_as_float(t_as_int);
        ((uint32_t*)optixLaunchParams.out_i)[launch_index.x] = i;
        ((float*)optixLaunchParams.out_t)[launch_index.x] = t;
    }

    __global__ void __closesthit__ch()
    {
        optixSetPayload_0(optixGetPrimitiveIndex());
        optixSetPayload_1(__float_as_int(optixGetRayTmax()));
    }

    __global__ void __miss__far()
    {
        optixSetPayload_0(0);
        optixSetPayload_1(__float_as_int(optixLaunchParams.t_max));
    }
}
