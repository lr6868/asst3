#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define BLOCKDIM 32
#define BLOCKSIZE 1024
#define SCAN_BLOCK_DIM BLOCKSIZE




////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];



__device__ __inline__ float2
cudaVec2CellNoise(float3 location, int index)
{
    int integer_of_x = static_cast<int>( location.x );
    int integer_of_y = static_cast<int>( location.y );
    int integer_of_z = static_cast<int>( location.z );
    int hash = cuConstNoiseXPermutationTable[ (integer_of_x*index) & 0xFF ];
    hash = cuConstNoiseXPermutationTable[ ( hash + integer_of_y ) & 0xFF ];
    hash = cuConstNoiseXPermutationTable[ ( hash + integer_of_z ) & 0xFF ];
    float x_result = cuConstNoise1DValueTable[ hash ];
    hash = cuConstNoiseYPermutationTable[ integer_of_x & 0xFF ];
    hash = cuConstNoiseYPermutationTable[ ( hash + integer_of_y ) & 0xFF ];
    hash = cuConstNoiseYPermutationTable[ ( hash + integer_of_z ) & 0xFF ];
    float y_result = cuConstNoise1DValueTable[ hash ];

    return make_float2(x_result, y_result);
}


__device__ __inline__ float3
lookupColor(float coord) {

    float scaledCoord = coord * (COLOR_MAP_SIZE-1);

    // using short type rather than int type since 16-bit integer math
    // is faster than 32-bit integrer math on NVIDIA GPUs
    short maxValue = COLOR_MAP_SIZE-1;
    short intCoord = static_cast<short>(scaledCoord);
    short base = (intCoord < maxValue) ? intCoord : maxValue;  // min

    // linearly interpolate between values in the table based on the
    // value of coord
    float weight = scaledCoord - static_cast<float>(base);
    float oneMinusWeight = 1.f - weight;

    float r = (oneMinusWeight * cuConstColorRamp[base][0]) + (weight * cuConstColorRamp[base+1][0]);
    float g = (oneMinusWeight * cuConstColorRamp[base][1]) + (weight * cuConstColorRamp[base+1][1]);
    float b = (oneMinusWeight * cuConstColorRamp[base][2]) + (weight * cuConstColorRamp[base+1][2]);
    return make_float3(r, g, b);
}



//exclusive scan
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE
inline __device__ uint
warpScanInclusive(int threadIndex, uint idata, volatile uint *s_Data, uint size){
    uint pos = 2 * threadIndex - (threadIndex & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
        s_Data[pos] += s_Data[pos - offset];

    return s_Data[pos];
}

inline __device__ uint warpScanExclusive(int threadIndex, uint idata, volatile uint *sScratch, uint size){
    return warpScanInclusive(threadIndex, idata, sScratch, size) - idata;
}

__inline__ __device__ void
sharedMemExclusiveScan(int threadIndex, uint* sInput, uint* sOutput, volatile uint* sScratch, uint size)
{
    if (size > WARP_SIZE) {

        uint idata = sInput[threadIndex];

        //Bottom-level inclusive warp scan
        uint warpResult = warpScanInclusive(threadIndex, idata, sScratch, WARP_SIZE);

        // Save top elements of each warp for exclusive warp scan sync
        // to wait for warp scans to complete (because s_Data is being
        // overwritten)
        __syncthreads();

        if ( (threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            sScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;

        // wait for warp scans to complete
        __syncthreads();

        if ( threadIndex < (SCAN_BLOCK_DIM / WARP_SIZE)) {
            // grab top warp elements
            uint val = sScratch[threadIndex];
            // calculate exclusive scan and write back to shared memory
            sScratch[threadIndex] = warpScanExclusive(threadIndex, val, sScratch, size >> LOG2_WARP_SIZE);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();

        sOutput[threadIndex] = warpResult + sScratch[threadIndex >> LOG2_WARP_SIZE] - idata;

    } else if (threadIndex < WARP_SIZE) {
        uint idata = sInput[threadIndex];
        sOutput[threadIndex] = warpScanExclusive(threadIndex, idata, sScratch, size);
    }
}






__inline__ __device__ void
sharedMemInclusiveScan(int threadIndex, uint* sInput, uint* sOutput, volatile uint* sScratch, uint size)
{
    if (size > WARP_SIZE) {

        uint idata = sInput[threadIndex];

        //Bottom-level inclusive warp scan
        uint warpResult = warpScanInclusive(threadIndex, idata, sScratch, WARP_SIZE);

        // Save top elements of each warp for exclusive warp scan sync
        // to wait for warp scans to complete (because s_Data is being
        // overwritten)
        __syncthreads();

        if ( (threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            sScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;

        // wait for warp scans to complete
        __syncthreads();

        if ( threadIndex < (SCAN_BLOCK_DIM / WARP_SIZE)) {
            // grab top warp elements
            uint val = sScratch[threadIndex];
            // calculate exclusive scan and write back to shared memory
            sScratch[threadIndex] = warpScanExclusive(threadIndex, val, sScratch, size >> LOG2_WARP_SIZE);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();

        sOutput[threadIndex] = warpResult + sScratch[threadIndex >> LOG2_WARP_SIZE];

    } else if (threadIndex < WARP_SIZE) {
        uint idata = sInput[threadIndex];
        sOutput[threadIndex] = warpScanInclusive(threadIndex, idata, sScratch, size);
    }
}


__device__ __inline__ int
circleInBox(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // clamp circle center to box (finds the closest point on the box)
    float closestX = (circleX > boxL) ? ((circleX < boxR) ? circleX : boxR) : boxL;
    float closestY = (circleY > boxB) ? ((circleY < boxT) ? circleY : boxT) : boxB;

    // is circle radius less than the distance to the closest point on
    // the box?
    float distX = closestX - circleX;
    float distY = closestY - circleY;

    if ( ((distX*distX) + (distY*distY)) <= (circleRadius*circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}


__device__ __inline__ int
circleInBoxConservative(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // expand box by circle radius.  Test if circle center is in the
    // expanded box.

    if ( circleX >= (boxL - circleRadius) &&
         circleX <= (boxR + circleRadius) &&
         circleY >= (boxB - circleRadius) &&
         circleY <= (boxT + circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

__inline__ __device__ void
findConservativeCircles(size_t tIdx, size_t circleIdx, uint* inclusiveOutput, uint* probableCircles) {
    if (tIdx == 0) {
        if (inclusiveOutput[0] == 1) 
            probableCircles[0] = circleIdx;
    } else if (inclusiveOutput[tIdx] == (inclusiveOutput[tIdx-1]+1)) {
        probableCircles[inclusiveOutput[tIdx-1]] = circleIdx;   
    }
}

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}
__inline__ __device__ void
findDefiniteCircles(size_t tIdx, uint* inclusiveOutput, uint* definiteCircles, uint* probableCircles) {
    if (tIdx == 0) {
        if (inclusiveOutput[0] == 1) 
            definiteCircles[0] = probableCircles[0];
    } else if (inclusiveOutput[tIdx] == (inclusiveOutput[tIdx-1]+1)) {
        definiteCircles[inclusiveOutput[tIdx-1]] = probableCircles[tIdx];
    }
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel_plain(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    // there is a non-zero contribution.  Now compute the shading value

    // simple: each circle has an assigned color
    int index3 = 3 * circleIndex;
    float3 rgb = *(float3*)&(cuConstRendererParams.color[index3]);
    float alpha = .5f;

    float oneMinusAlpha = 1.f - alpha;

    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;
}

__device__ __inline__ void
shadePixel_snow(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;


    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    float3 rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f-p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
    float alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);


    float oneMinusAlpha = 1.f - alpha;

    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;
}



// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles_plain() {

    const short px=blockIdx.x * blockDim.x + threadIdx.x;
    const short py=blockIdx.y * blockDim.y + threadIdx.y;


    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    const short imageWidth = cuConstRendererParams.imageWidth;
    const short imageHeight = cuConstRendererParams.imageHeight;

    const float invWidth = 1.f / imageWidth;
    const float invHeight = 1.f / imageHeight;

    // Get the left, right, top and bottom of the section
    const float boxL = static_cast<float>(blockIdx.x) / gridDim.x;
    const float boxR = boxL + static_cast<float>(blockDim.x) *invWidth;
    const float boxB = static_cast<float>(blockIdx.y) / gridDim.y;
    const float boxT = boxB + static_cast<float>(blockDim.y) *invHeight ;
    //index for circles
    const size_t tIdx = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ uint inSection[BLOCKSIZE];
    __shared__ uint inclusiveOutput[BLOCKSIZE];
    __shared__ uint probableCircles[BLOCKSIZE];
    __shared__ uint scratchPad[2*BLOCKSIZE];


    

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * imageWidth + px)]);
    float4 color;
    color = *imgPtr;
    
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(px) + 0.5f),
                                        invHeight * (static_cast<float>(py) + 0.5f));
    const int numc=cuConstRendererParams.numCircles;
    for (int start=0; start< numc;start+=BLOCKSIZE){
        
        int index=start+tIdx; 
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[3 * index]);

        inSection[tIdx]=index<numc ?circleInBoxConservative(p.x, p.y, cuConstRendererParams.radius[index], boxL, boxR, boxT, boxB):0;
        __syncthreads();
        sharedMemInclusiveScan(tIdx, inSection, inclusiveOutput, scratchPad, BLOCKSIZE);
        __syncthreads();
        findConservativeCircles(tIdx, index, inclusiveOutput, probableCircles);
        __syncthreads();
        const short numConservativeCircles = inclusiveOutput[BLOCKSIZE-1];
        int k=probableCircles[tIdx];
        p = *(float3*)(&cuConstRendererParams.position[3*k]);
        inSection[tIdx]=tIdx< numConservativeCircles ?circleInBox(p.x, p.y,  cuConstRendererParams.radius[k], boxL, boxR, boxT, boxB):0;  
        __syncthreads();
        sharedMemInclusiveScan(tIdx, inSection, inclusiveOutput, scratchPad, BLOCKSIZE);
        __syncthreads();
        //inSection is the output, using existing memory
        findDefiniteCircles(tIdx, inclusiveOutput, inSection, probableCircles);
        __syncthreads();
        const short numDefiniteCircles = inclusiveOutput[numConservativeCircles-1];
        for(short i=0;i<numDefiniteCircles;i++){
                k=inSection[i];
                shadePixel_plain(k, pixelCenterNorm, *(float3*)(&cuConstRendererParams.position[3 * k]), &color);
            }
        __syncthreads();
    }
    *imgPtr = color;
}

__global__ void kernelRenderCircles_snow() {

    const short px=blockIdx.x * blockDim.x + threadIdx.x;
    const short py=blockIdx.y * blockDim.y + threadIdx.y;


    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    const short imageWidth = cuConstRendererParams.imageWidth;
    const short imageHeight = cuConstRendererParams.imageHeight;

    const float invWidth = 1.f / imageWidth;
    const float invHeight = 1.f / imageHeight;

    // Get the left, right, top and bottom of the section
    const float boxL = static_cast<float>(blockIdx.x) / gridDim.x;
    const float boxR = boxL + static_cast<float>(blockDim.x) *invWidth;
    const float boxB = static_cast<float>(blockIdx.y) / gridDim.y;
    const float boxT = boxB + static_cast<float>(blockDim.y) *invHeight ;
    //index for circles
    const size_t tIdx = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ uint inSection[BLOCKSIZE];
    __shared__ uint inclusiveOutput[BLOCKSIZE];
    __shared__ uint probableCircles[BLOCKSIZE];
    __shared__ uint scratchPad[2*BLOCKSIZE];


    

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * imageWidth + px)]);
    float4 color;
    color = *imgPtr;
    
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(px) + 0.5f),
                                        invHeight * (static_cast<float>(py) + 0.5f));
    const int numc=cuConstRendererParams.numCircles;
    for (int start=0; start< numc;start+=BLOCKSIZE){
        
        int index=start+tIdx; 
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[3 * index]);

        inSection[tIdx]=index<numc ?circleInBoxConservative(p.x, p.y, cuConstRendererParams.radius[index], boxL, boxR, boxT, boxB):0;
        __syncthreads();
        sharedMemInclusiveScan(tIdx, inSection, inclusiveOutput, scratchPad, BLOCKSIZE);
        __syncthreads();
        findConservativeCircles(tIdx, index, inclusiveOutput, probableCircles);
        __syncthreads();
        const short numConservativeCircles = inclusiveOutput[BLOCKSIZE-1];
        int k=probableCircles[tIdx];
        p = *(float3*)(&cuConstRendererParams.position[3*k]);
        inSection[tIdx]=tIdx< numConservativeCircles ?circleInBox(p.x, p.y,  cuConstRendererParams.radius[k], boxL, boxR, boxT, boxB):0;  
        __syncthreads();
        sharedMemInclusiveScan(tIdx, inSection, inclusiveOutput, scratchPad, BLOCKSIZE);
        __syncthreads();
        //inSection is the output, using existing memory
        findDefiniteCircles(tIdx, inclusiveOutput, inSection, probableCircles);
        __syncthreads();
        const short numDefiniteCircles = inclusiveOutput[numConservativeCircles-1];
        for(short i=0;i<numDefiniteCircles;i++){
                k=inSection[i];
                shadePixel_snow(k, pixelCenterNorm, *(float3*)(&cuConstRendererParams.position[3 * k]), &color);
            }
        __syncthreads();
    }
    *imgPtr = color;
}

////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelRenderCircles_l() {

    const int px=blockIdx.x * blockDim.x + threadIdx.x;
    const int py=blockIdx.y * blockDim.y + threadIdx.y;


    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;


    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * imageWidth + px)]);

    for (int index=0; index< cuConstRendererParams.numCircles;index++){
        int index3 = 3 * index;
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(px) + 0.5f),
                                                 invHeight * (static_cast<float>(py) + 0.5f));
            shadePixel_plain(index, pixelCenterNorm, p, imgPtr);
    }
}

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    if(sceneName== SNOWFLAKES ||sceneName== SNOWFLAKES_SINGLE_FRAME) {
        dim3 blockDim(BLOCKDIM, BLOCKDIM);
        size_t gridDimX = (image->width + blockDim.x - 1) / blockDim.x;
        size_t gridDimY = (image->height + blockDim.y - 1) / blockDim.y;
        dim3 gridDim(gridDimX, gridDimY);

        kernelRenderCircles_snow<<<gridDim, blockDim>>>();
        cudaDeviceSynchronize();
        
    }
    else{
        if (numCircles < 4){
        dim3 blockDim(BLOCKDIM, BLOCKDIM);
    size_t gridDimX = (image->width + blockDim.x - 1) / blockDim.x;
    size_t gridDimY = (image->height + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridDimX, gridDimY);

    kernelRenderCircles_l<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    }
    else{
    dim3 blockDim(BLOCKDIM, BLOCKDIM);
    size_t gridDimX = (image->width + blockDim.x - 1) / blockDim.x;
    size_t gridDimY = (image->height + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridDimX, gridDimY);

    kernelRenderCircles_plain<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    }
    }
}
