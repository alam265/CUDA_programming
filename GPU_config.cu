#include <stdio.h>
#include <cuda_runtime.h>

// Returns CUDA cores per SM based on compute capability
int cores_per_sm(int major, int minor) {
    // Turing (7.5)
    if (major == 7 && minor == 5) return 64;

    // Volta (7.0)
    if (major == 7 && minor == 0) return 64;

    // Ampere (8.6, 8.9)
    if (major == 8 && minor == 6) return 128;
    if (major == 8 && minor == 9) return 128;

    // Ada Lovelace (8.9 alternative)
    if (major == 8 && minor == 9) return 128;

    // Hopper (9.0)
    if (major == 9 && minor == 0) return 128;

    // Default fallback
    return 64;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int cores_sm = cores_per_sm(prop.major, prop.minor);
    int total_cores = prop.multiProcessorCount * cores_sm;

    printf("GPU Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("CUDA Cores per SM: %d\n", cores_sm);
    printf("Total CUDA Cores: %d\n", total_cores);

    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);

    //Warp
    printf("Warp Size: %d\n", prop.warpSize);

    return 0;
}

