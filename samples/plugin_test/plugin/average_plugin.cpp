/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "average_plugin.h"
#include "NvInferPlugin.h"
#include "common.h" // volume(), ASSERT
#include "logger.h" // sample::gLogError
#include "average_kernel.hpp"

#include <cuda.h>

using namespace nvinfer1;

#define CUDRIVER_CALL(call)                                                                                            \
    {                                                                                                                  \
        cudaError_enum s_ = call;                                                                                      \
        if (s_ != CUDA_SUCCESS)                                                                                        \
        {                                                                                                              \
            char const *errName_, *errDesc_;                                                                           \
            cuGetErrorName(s_, &errName_);                                                                             \
            cuGetErrorString(s_, &errDesc_);                                                                           \
            sample::gLogError << "CUDA Error: " << errName_ << " " << errDesc_ << std::endl;                           \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

#define CUDA_CALL(call)                                                                                                \
    {                                                                                                                  \
        cudaError_t s_ = call;                                                                                         \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            sample::gLogError << "CUDA Error: " << cudaGetErrorName(s_) << " " << cudaGetErrorString(s_) << std::endl; \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

#define CUBLAS_CALL(call)                                                                                              \
    {                                                                                                                  \
        cublasStatus_t s_ = call;                                                                                      \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            sample::gLogError << "cuBLAS Error: " << s_ << std::endl;                                                  \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(uint8_t*& buffer, T const& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(uint8_t const*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// Static class fields initialization
PluginFieldCollection AveragePluginCreator::mFC{};
std::vector<PluginField> AveragePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AveragePluginCreator);

namespace
{
constexpr char const* kHARDMAX_NAME{"AveragePlugin"};
constexpr char const* kHARDMAX_VERSION{"1"};
} // namespace

AveragePlugin::AveragePlugin(int32_t axis)
{
    mAxis = axis;
}

AveragePlugin::AveragePlugin(void const* serialData, size_t serialLength)
{
    uint8_t const* d = static_cast<uint8_t const*>(serialData);
    uint8_t const* a = d;

    mAxis = readFromBuffer<int32_t>(d);
    mAxisSize = readFromBuffer<int32_t>(d);
    mDimProductOuter = readFromBuffer<int32_t>(d);
    mDimProductInner = readFromBuffer<int32_t>(d);

    ASSERT(d == (a + serialLength));
}

AveragePlugin::~AveragePlugin()
{
    terminate();
}

int32_t AveragePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t AveragePlugin::initialize() noexcept
{
    return 0;
}

char const* AveragePlugin::getPluginType() const noexcept
{
    return kHARDMAX_NAME;
}

char const* AveragePlugin::getPluginVersion() const noexcept
{
    return kHARDMAX_VERSION;
}

nvinfer1::DimsExprs AveragePlugin::getOutputDimensions(
    int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(index == 0);

    // Dimensions are unchanged
    return inputs[0];
}

void AveragePlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    ASSERT(
        cublasContext != nullptr && "AveragePlugin given a null cuBLAS Context. Was the CUBLAS TacticSource disabled?");
    mCublas = cublasContext;
}

// Detach the plugin object from its execution context.
void AveragePlugin::detachFromContext() noexcept {}

int32_t AveragePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (inputDesc[0].type != nvinfer1::DataType::kFLOAT)
    {
        return -1;
    }

    CUBLAS_CALL(cublasSetStream(mCublas, stream));

    auto const* input_0 = static_cast<const float*>(inputs[0]);
    auto const* input_1 = static_cast<const float*>(inputs[1]);
    auto* result = static_cast<float*>(outputs[0]);

    // Make sure output is initialized to all 0's.
    // Later we will set the correct outputs to be 1's and not touch the rest.
    CUDA_CALL(cudaMemsetAsync(result, 0, mDimProductOuter * mDimProductInner * mAxisSize * sizeof(float), stream));

    // max_c(data, result, mDimProductOuter * mDimProductInner * mAxisSize);
    average_op(input_0, input_1, result, mDimProductOuter * mDimProductInner * mAxisSize);
   
    return cudaPeekAtLastError();
}

size_t AveragePlugin::getSerializationSize() const noexcept
{
    return 4 * sizeof(int32_t);
}

void AveragePlugin::serialize(void* buffer) const noexcept
{
    // Same order as in deserialize()
    uint8_t* d = static_cast<uint8_t*>(buffer);
    uint8_t* const a = d;

    writeToBuffer(d, mAxis);
    writeToBuffer(d, mAxisSize);
    writeToBuffer(d, mDimProductOuter);
    writeToBuffer(d, mDimProductInner);

    ASSERT(d == a + getSerializationSize());
}

bool AveragePlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    // No change of type allowed
    if (inOut[0].type != inOut[pos].type)
    {
        return false;
    }

    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
}

void AveragePlugin::terminate() noexcept {}

void AveragePlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* AveragePlugin::clone() const noexcept
{
    auto* plugin = new AveragePlugin(mAxis);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->mAxisSize = mAxisSize;
    plugin->mDimProductInner = mDimProductInner;
    plugin->mDimProductOuter = mDimProductOuter;
    plugin->mCublas = mCublas;
    return plugin;
}

void AveragePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    nvinfer1::Dims const& inDims_0 = in[0].desc.dims;
    nvinfer1::Dims const& inDims_1 = in[0].desc.dims;
    nvinfer1::Dims const& outDims = out[0].desc.dims;

    // Check that inputs and outputs have the same dimensions
    ASSERT(inDims_0.nbDims == outDims.nbDims);
    for (int32_t dim = 0; dim < inDims_0.nbDims; dim++)
    {
        ASSERT(inDims_0.d[dim] == outDims.d[dim]);
    }
    ASSERT(inDims_1.nbDims == outDims.nbDims);
    for (int32_t dim = 0; dim < inDims_1.nbDims; dim++)
    {
        ASSERT(inDims_1.d[dim] == outDims.d[dim]);
    }

    // Check that axis is valid
    if (mAxis < 0)
    {
        mAxis += inDims_0.nbDims;
        ASSERT(mAxis >= 0);
    }
    ASSERT(inDims_0.nbDims > mAxis);

    // samplesCommon::volume() requires that all dimensions are non-negative.
    // Even in the case of dynamic shapes, the plugin will be configured with
    // resolved shapes before enqueue() is called, so the below member variables
    // will be set correctly.
    if (std::all_of(inDims_0.d, inDims_0.d + inDims_0.nbDims, [](int32_t x) { return x >= 0; }))
    {
        mDimProductOuter = samplesCommon::volume(inDims_0, 0, mAxis);
        mAxisSize = inDims_0.d[mAxis];
        mDimProductInner = samplesCommon::volume(inDims_0, mAxis + 1, inDims_0.nbDims);
    }
}

nvinfer1::DataType AveragePlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    ASSERT(inputTypes && nbInputs == 2 && index == 0);
    return inputTypes[0];
}

size_t AveragePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // 1st array to store the contents of the working axis
    // 2nd array to store an array of 1's
    return 2 * inputs[0].dims.d[mAxis] * sizeof(float);
}

void AveragePlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    ASSERT(libNamespace != nullptr);
    mNamespace = libNamespace;
}

char const* AveragePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

AveragePluginCreator::AveragePluginCreator()
{
    mPluginAttributes.clear();

    // Consistent with the ONNX model attr fields
    static auto const axisField = PluginField("axis", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back(axisField);

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* AveragePluginCreator::getPluginName() const noexcept
{
    return kHARDMAX_NAME;
}

char const* AveragePluginCreator::getPluginVersion() const noexcept
{
    return kHARDMAX_VERSION;
}

PluginFieldCollection const* AveragePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

char const* AveragePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void AveragePluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    ASSERT(libNamespace != nullptr);
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* AveragePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    // Set default value
    int32_t axis = -1;

    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        if (!strcmp(fc->fields[i].name, "axis"))
        {
            ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
            axis = *static_cast<int32_t const*>(fc->fields[i].data);
        }
    }

    AveragePlugin* plugin = new AveragePlugin(axis);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* AveragePluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    AveragePlugin* plugin = new AveragePlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}
