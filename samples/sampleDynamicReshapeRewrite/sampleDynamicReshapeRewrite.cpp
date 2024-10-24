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

//!
//! sampleNamedDimensions.cpp
//! This file contains the implementation of the named dimensions sample. It creates the network using
//! a synthetic ONNX model with named input dimensions.
//! It can be run with the following command line:
//! Command: ./sample_named_dimensions [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const gSampleName = "TensorRT.sample_named_dimensions";

//! \brief  The SampleNamedDimensions class implements a sample with named input dimensions
//!
//! \details It creates the network using an ONNX model
//!
class SampleNamedDimensions
{
public:
    SampleNamedDimensions(samplesCommon::OnnxSampleParams const& params, int input_h, int input_w)
        : mParams(params)
        , mEngine(nullptr)
        , input_h_(input_h)
        , input_w_(input_w)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    //! Input Tensors.
    int input_h_{1};
    int input_w_{1};

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::IRuntime> runtime;

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool setInput(float *buffers1, Dims2 inputDims1, float *buffers2, Dims2 inputDims2);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool printOutput(float *output, Dims2 outputDims);

    void printNetworkDimsInfo(std::shared_ptr<nvinfer1::INetworkDefinition> network, std::string prompt_info) {
      std::cout << "-----INetworkDefinition info: " << prompt_info << "-----"
                << std::endl;
      std::cout << network->getInput(0)->getName() << ": "
                << network->getInput(0)->getDimensions().d[0] << ", "
                << network->getInput(0)->getDimensions().d[1] << std::endl;
      std::cout << network->getInput(1)->getName() << ": "
                << network->getInput(1)->getDimensions().d[0] << ", "
                << network->getInput(1)->getDimensions().d[1] << std::endl;
      std::cout << network->getOutput(0)->getName() << ": "
                << network->getOutput(0)->getDimensions().d[0] << ", "
                << network->getOutput(0)->getDimensions().d[1] << std::endl;
    }
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the network definition by parsing the Onnx model and builds
//!          the engine that will be used to run the model (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleNamedDimensions::build()
{
    auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto const explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int32_t>(sample::gLogger.getReportableSeverity()));
    if (!parsed) {
      return false;
    }

    ASSERT(network->getNbInputs() == 2);
    ASSERT(network->getInput(0)->getDimensions().nbDims == 2);
    ASSERT(network->getInput(1)->getDimensions().nbDims == 2);
    ASSERT(network->getNbOutputs() == 1);
    ASSERT(network->getOutput(0)->getDimensions().nbDims == 2);

    printNetworkDimsInfo(network, "after parse form file");

    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    // add optimization profile
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims2(1, 8));
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims2(1, 16));

    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kOPT, Dims2(1, 8));
    profile->setDimensions(network->getInput(1)->getName(), OptProfileSelector::kMAX, Dims2(1, 16));

    if(config->addOptimizationProfile(profile)< 0) {
      std::cout << "add optimization profile fail" << std::endl;
    }
    
    printNetworkDimsInfo(network, "after optimization profile");

    std::shared_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }
    printNetworkDimsInfo(network, "after build serialize network");

    runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        return false;
    }

    std::cout << "------------ ICudaEngine info-----------------" << std::endl;
    std::cout << "mEngine->getNbBindings(): " << mEngine->getNbBindings()
              << std::endl;
    for (int i = 0; i < mEngine->getNbBindings(); i++) {
      Dims bind_dims = mEngine->getBindingDimensions(i);
      std::cout << "bind_dims.nbDims: " << bind_dims.nbDims << ", "
                << "bind_dims.d[0]: " << bind_dims.d[0] << ", "
                << "bind_dims.d[1]: " << bind_dims.d[1] << std::endl;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleNamedDimensions::infer()
{
    float *h_input0 = nullptr, *h_input1 = nullptr, *h_output = nullptr;
    float *d_input0 = nullptr, *d_input1 = nullptr, *d_output = nullptr;
    size_t input_sz0 = 0, input_sz1 = 0, output_sz = 0;

    auto context = std::shared_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());
    if (!context) {
        return false;
    }

    Dims2 inputDims{input_h_, input_w_};
    Dims2 outputDims;
    // Set the input size for the preprocessor
    context->setBindingDimensions(0, inputDims);
    context->setBindingDimensions(1, inputDims);
    std::cout << "=================set input size: " << input_h_ << ", "
              << input_w_ << std::endl;
    // We can only run inference once all dynamic input shapes have been
    // specified.
    if (!context->allInputDimensionsSpecified()) {
        std::cout << "[ERROR]: input dimension not specified" << std::endl;
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
      sample::gLogInfo << binding_name << ":  data nun = " << volume << std::endl;
      volume *= samplesCommon::getElementSize(type);
      if (mEngine->bindingIsInput(i)) {
        if (binding_name == std::string("input0")) {
          h_input0 = (float *)malloc(volume);
          CHECK(cudaMalloc((void **)&d_input0, volume));
          input_sz0 = volume;
          std::cout << "[info] get input0 " << std::endl;
        } else if (binding_name == std::string("input1")) {
          h_input1 = (float *)malloc(volume);
          CHECK(cudaMalloc((void **)&d_input1, volume));
          input_sz1 = volume;
          std::cout << "[info] get input1 " << std::endl;
        } else {
          std::cout << "[ERROR] unkonwn input name: "
                    << mEngine->getBindingName(i) << std::endl;
        }
      } else {
        outputDims = Dims2{dims.d[0], dims.d[1]};
        h_output = (float *)malloc(volume);
        CHECK(cudaMalloc((void **)&d_output, volume));
        output_sz = volume;
      }
    }

    // Set the input data into the managed buffers
    if (!setInput(h_input0, inputDims, h_input1, inputDims))
    {
        return false;
    }

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

    // Verify results
    if (!printOutput(h_output, outputDims))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleNamedDimensions::setInput(float *buffers1, Dims2 inputDims1, float *buffers2, Dims2 inputDims2)
{
    int32_t const input0H = inputDims1.d[0];
    int32_t const input0W = inputDims1.d[1];
    int32_t const input1H = inputDims2.d[0];
    int32_t const input1W = inputDims2.d[1];

    std::default_random_engine generator(static_cast<uint32_t>(time(nullptr)));
    std::uniform_real_distribution<float> unif_real_distr(-10., 10.);

    sample::gLogInfo << "Input0:\n";
    for (int32_t i = 0; i < input0H * input0W; i++)
    {
        buffers1[i] = unif_real_distr(generator);
        sample::gLogInfo << std::setw(10) << buffers1[i] << (((i + 1) % input0W) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    sample::gLogInfo << "Input1:\n";
    for (int32_t i = 0; i < input1H * input1W; i++)
    {
        buffers2[i] = unif_real_distr(generator);
        sample::gLogInfo << std::setw(10) << buffers2[i] << (((i + 1) % input1W) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    return true;
}

//!
//! \brief Verify the result of concatenation
//!
//! \return whether the concatenated tesnor matches reference
//!
bool SampleNamedDimensions::printOutput(float *output, Dims2 outputDims)
{
    int32_t const outputH = outputDims.d[0];
    int32_t const outputW = outputDims.d[1];
    int32_t const outputSize = outputH * outputW;

    sample::gLogInfo << "Output:\n";
    for (int32_t i = 0; i < outputSize; i++)
    {
        sample::gLogInfo << std::setw(10) << output[i] << (((i + 1) % outputW) ? " " : "\n");
    }
    sample::gLogInfo << std::endl;

    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(samplesCommon::Args const& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("../samples/sampleDynamicReshapeRewrite/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "concat_layer.onnx";
    params.inputTensorNames.push_back("input0");
    params.inputTensorNames.push_back("input1");
    params.outputTensorNames.push_back("output");

    return params;
}

int32_t main(int32_t argc, char** argv)
{
    if (argc != 3) {
        std::cout << argv[0] << " height width" << std::endl;
        exit(1);
    }
    int height = std::stoi(argv[1]);
    int width = std::stoi(argv[2]);
    std::cout << "height: " << height << ", width: " << width << std::endl;

    samplesCommon::Args args;
    SampleNamedDimensions sample(initializeSampleParams(args), height, width);

    sample::gLogInfo << "Building and running a GPU inference engine for synthetic ONNX model" << std::endl;

    if (!sample.build())
    {
      sample::gLogInfo << "build failed";
      return -1;
    }
    if (!sample.infer())
    {
      sample::gLogInfo << "infer failed";
      return -1;
    }

    return 0;
}
