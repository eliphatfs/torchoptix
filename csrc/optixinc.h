#pragma once
#include "optix.h"

struct LaunchParams
{
    OptixTraversableHandle traversable;
    unsigned long long rays_o, rays_d, out_t, out_i;
    float t_max;
};
