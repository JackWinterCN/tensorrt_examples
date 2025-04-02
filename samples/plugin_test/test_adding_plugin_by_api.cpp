/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//!
//! SamplePluginTest.cpp
//! This file contains a sample demonstrating a plugin for NonZero.
//! It can be run with the following command line:
//! Command: ./sample_non_zero_plugin [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
// #include "nonZeroKernel.h"
#include "parserOnnxConfig.h"
#include "plugin/average_plugin.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const kSAMPLE_NAME = "TensorRT.sample_non_zero_plugin";

constexpr char const* kHARDMAX_NAME{"AveragePlugin"};
constexpr char const* kHARDMAX_VERSION{"1"};

using half = __half;

namespace
{

struct CustPluginParams : public samplesCommon::SampleParams
{
    int axis{0};
};

} // namespace

//! \brief  The SamplePluginTest class implements a NonZero plugin
//!
//! \details The plugin is able to output the non-zero indices in row major or column major order
//!
class SamplePluginTest
{
public:
    SamplePluginTest(CustPluginParams const& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
        // mSeed = static_cast<uint32_t>(time(nullptr));
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    bool infer_with_manual_buffer();

private:
    CustPluginParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    // uint32_t mSeed{};

    //!
    //! \brief Creates a TensorRT network and inserts a NonZero plugin
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(samplesCommon::BufferManager const& buffers);

    //!
    //! \brief Verifies the result
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates a network containing a NonZeroPlugin and builds
//!          the engine that will be used to run the plugin (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SamplePluginTest::build()
{
    sample::gLogInfo << "==================build=================" << std::endl; 
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    auto const explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    // cannot release the plugin creator before finish network construction
    auto pluginCreator = std::make_unique<AveragePluginCreator>();
    getPluginRegistry()->registerCreator(*pluginCreator.get(), "");

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 2);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 2);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Creates a network with a single custom layer containing the NonZero plugin and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the NonZero plugin
//!
//! \param builder Pointer to the engine builder
//!
bool SamplePluginTest::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    sample::gLogInfo << "=============constructNetwork===========" << std::endl;
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    int32_t const R = 1;
    int32_t const C = 4;
    auto* input0 = network->addInput(mParams.inputTensorNames[0].c_str(), DataType::kFLOAT, {2, {R, C}});
    ASSERT(input0 != nullptr);
    auto* input1 = network->addInput(mParams.inputTensorNames[1].c_str(), DataType::kFLOAT, {2, {R, C}});
    ASSERT(input1 != nullptr);

    std::vector<PluginField> const vecPF{{"axis", &mParams.axis, PluginFieldType::kINT32, 1}};
    PluginFieldCollection pfc{static_cast<int32_t>(vecPF.size()), vecPF.data()};

    auto pluginCreator = static_cast<IPluginCreator*>(getPluginRegistry()->getPluginCreator(kHARDMAX_NAME, kHARDMAX_VERSION, ""));
    ASSERT(pluginCreator != nullptr);
    auto plugin = std::unique_ptr<IPluginV2>(pluginCreator->createPlugin(kHARDMAX_NAME, &pfc));
    ASSERT(plugin != nullptr);
    std::vector<ITensor*> inputsVec{input0, input1};
    auto pluginLayer = network->addPluginV2(inputsVec.data(), inputsVec.size(), *plugin);
    ASSERT(pluginLayer != nullptr);
    ASSERT(pluginLayer->getInput(0) != nullptr);
    ASSERT(pluginLayer->getInput(1) != nullptr);
    ASSERT(pluginLayer->getOutput(0) != nullptr);
    sample::gLogInfo << "pluginLayer info: "
                     << pluginLayer->getInput(0)->getName() << ", "
                     << pluginLayer->getInput(1)->getName() << ", "
                     << pluginLayer->getOutput(0)->getName() << std::endl;

    pluginLayer->getOutput(0)->setName(mParams.outputTensorNames[0].c_str());

    network->markOutput(*(pluginLayer->getOutput(0)));
    sample::gLogInfo << "===========constructNetwork end=========" << std::endl;
    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SamplePluginTest::infer()
{
    sample::gLogInfo << "====================infer===============" << std::endl;
    // Since the data dependent output size cannot be inferred from the engine denote a sufficient size for the
    // corresponding output buffer (along with the rest of the I/O tensors)
    std::vector<int64_t> ioVolumes = {mInputDims.d[0] * mInputDims.d[1], mInputDims.d[0] * mInputDims.d[1] * 2, 1};

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, 0, context.get());

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 2);
    if (!processInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = context->enqueueV3(stream);
    if (!status)
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));

    // Release stream.
    CHECK(cudaStreamDestroy(stream));

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

bool SamplePluginTest::infer_with_manual_buffer()
{
    sample::gLogInfo << "=========infer_with_manual_buffer=======" << std::endl;
    float *h_input0 = nullptr, *h_input1 = nullptr, *h_output = nullptr;
    float *d_input0 = nullptr, *d_input1 = nullptr, *d_output = nullptr;
    size_t input_sz0 = 4, input_sz1 = 4, output_sz = 4;

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    sample::gLogInfo << "getBindingDimensions(0).nbDims: " << mEngine->getBindingDimensions(0).nbDims << std::endl;
    sample::gLogInfo << "getBindingDimensions(0).d[0]: " << mEngine->getBindingDimensions(0).d[0] << std::endl;
    sample::gLogInfo << "getBindingDimensions(1).nbDims: " << mEngine->getBindingDimensions(1).nbDims << std::endl;
    sample::gLogInfo << "getBindingDimensions(1).d[0]: " << mEngine->getBindingDimensions(1).d[0] << std::endl;
    sample::gLogInfo << "getBindingDimensions(2).nbDims: " << mEngine->getBindingDimensions(2).nbDims << std::endl;
    sample::gLogInfo << "getBindingDimensions(2).d[0]: " << mEngine->getBindingDimensions(2).d[0] << std::endl;

    Dims2 inputDims{1, 4};
    Dims2 outputDims;
    // Set the input size for the preprocessor
    if (!context->setBindingDimensions(0, inputDims)) {
      sample::gLogInfo << "[ERR]: setBindingDimensions(0) failed" << std::endl;
    }
    if (!context->setBindingDimensions(1, inputDims)) {
      sample::gLogInfo << "[ERR]: setBindingDimensions(1) failed" << std::endl;
    }
    // We can only run inference once all dynamic input shapes have been
    // specified.
    if (!context->allInputDimensionsSpecified()) {
        sample::gLogInfo << "[ERROR]: input dimension not specified" << std::endl;
        return false;
    }

    for (int i = 0; i < mEngine->getNbBindings(); i++) {
      auto dims = context->getBindingDimensions(i);
      nvinfer1::DataType type = mEngine->getBindingDataType(i);
      std::string binding_name(mEngine->getBindingName(i));
      size_t volume = 1;
      for (int i = 0; i < dims.nbDims; i++) {
        volume *= dims.d[i];
      }
      sample::gLogInfo << binding_name << ":  data num = " << volume
                       << std::endl;
      volume *= samplesCommon::getElementSize(type);
      if (mEngine->bindingIsInput(i)) {
        if (binding_name == std::string(mParams.inputTensorNames[0])) {
          h_input0 = (float *)malloc(volume);
          CHECK(cudaMalloc((void **)&d_input0, volume));
          input_sz0 = volume;
          sample::gLogInfo << "[info] get input0 " << std::endl;
        } else if (binding_name == std::string(mParams.inputTensorNames[1])) {
          h_input1 = (float *)malloc(volume);
          CHECK(cudaMalloc((void **)&d_input1, volume));
          input_sz1 = volume;
          sample::gLogInfo << "[info] get input1 " << std::endl;
        } else {
          sample::gLogInfo << "[ERROR] unkonwn input name: "
                           << mEngine->getBindingName(i) << std::endl;
        }
      } else {
        outputDims = Dims2{dims.d[0], dims.d[1]};
        h_output = (float *)malloc(volume);
        CHECK(cudaMalloc((void **)&d_output, volume));
        output_sz = volume;
      }
    }

    std::default_random_engine generator(static_cast<uint32_t>(time(nullptr)));
    std::uniform_real_distribution<float> unif_real_distr(-10., 10.);

    int32_t const input0H = inputDims.d[0];
    int32_t const input0W = inputDims.d[1];
    int32_t const input1H = inputDims.d[0];
    int32_t const input1W = inputDims.d[1];

    sample::gLogInfo << "Input0:\n";
    for (int32_t i = 0; i < input0H * input0W; i++)
    {
        h_input0[i] = unif_real_distr(generator);
        sample::gLogInfo << std::setw(10) << h_input0[i] << (((i + 1) % input0W) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    sample::gLogInfo << "Input1:\n";
    for (int32_t i = 0; i < input1H * input1W; i++)
    {
        h_input1[i] = unif_real_distr(generator);
        sample::gLogInfo << std::setw(10) << h_input1[i] << (((i + 1) % input1W) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    // Memcpy from host input buffers to device input buffers
    CHECK(cudaMemcpy(d_input0, h_input0, input_sz0, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_input1, h_input1, input_sz1, cudaMemcpyHostToDevice));
    std::vector<void *> data_v{static_cast<void *>(d_input0),
                               static_cast<void *>(d_input1),
                               static_cast<void *>(d_output)};
    bool status = context->executeV2(data_v.data());
    if (!status) {
      return false;
    }

    // Memcpy from device output buffers to host output buffers
    CHECK(cudaMemcpy(h_output, d_output, output_sz, cudaMemcpyDeviceToHost));

    int32_t const outputH = outputDims.d[0];
    int32_t const outputW = outputDims.d[1];
    int32_t const outputSize = outputH * outputW;

    sample::gLogInfo << "Output:\n";
    for (int32_t i = 0; i < outputSize; i++)
    {
        sample::gLogInfo << std::setw(10) << h_output[i] << (((i + 1) % outputW) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;
    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SamplePluginTest::processInput(samplesCommon::BufferManager const& buffers)
{
    float* hostDataBuffer_0 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float* hostDataBuffer_1 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));
    
    for (int32_t i = 0; i < 4; ++i)
    {
        hostDataBuffer_0[i] = i;
        hostDataBuffer_1[i] = i * i;
    }
    sample::gLogInfo << "Input0:";
    for (int32_t i = 0; i < 4; ++i)
    {
        sample::gLogInfo << hostDataBuffer_0[i] << ",";
    }
    sample::gLogInfo << std::endl;

    sample::gLogInfo << "Input1:";
    for (int32_t i = 0; i < 4; ++i)
    {
        sample::gLogInfo << hostDataBuffer_1[i] << ",";
    }
    sample::gLogInfo << std::endl;

    return true;
}

//!
//! \brief Verify result
//!
//! \return whether the output correctly identifies all (and only) non-zero elements
//!
bool SamplePluginTest::verifyOutput(samplesCommon::BufferManager const& buffers)
{

  float *output = static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
  sample::gLogInfo << "Output:";
  for (int32_t i = 0; i < 4; ++i) {
    sample::gLogInfo << output[i] << ",";
  }
  sample::gLogInfo << std::endl;


  return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
CustPluginParams initializeSampleParams(samplesCommon::Args const& args)
{
    CustPluginParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("models/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.inputTensorNames.push_back("inputA");
    params.inputTensorNames.push_back("inputB");
    params.outputTensorNames.push_back("output");
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
    std::cout << "--columnOrder   Run plugin in column major output mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SamplePluginTest sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for NonZero plugin" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer_with_manual_buffer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}
