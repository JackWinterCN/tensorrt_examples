/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates
//! the network using the MNIST onnx model. It can be run with the following
//! command line: Command: ./sample_onnx_mnist [-h or --help]
//! [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "common.h"
#include "logger.h"

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace nvinfer1;

// const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST {
public:
  SampleOnnxMNIST(const samplesCommon::OnnxSampleParams &params)
      : mParams(params), mEngine(nullptr) {}

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

  nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
  nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
  int mNumber{0};             //!< The number to classify

  std::shared_ptr<nvinfer1::ICudaEngine>
      mEngine; //!< The TensorRT engine used to run the network

  //!
  //! \brief Reads the input  and stores the result in a managed buffer
  //!
  bool processInput(float *buffers);

  //!
  //! \brief Classifies digits and verify result
  //!
  bool verifyOutput(float *buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network
//! engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx
//! model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build() {
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder) {
    return false;
  }

  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    return false;
  }

  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  // Enable INT8 model. Required to set custom per-tensor dynamic range or INT8
  // Calibration
  config->setFlag(BuilderFlag::kINT8);

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
  if (!parser) {
    return false;
  }
  int verbosity = (int) nvinfer1::ILogger::Severity::kVERBOSE;
  auto parsed = parser->parseFromFile("../data/resnet50/quant_resnet50.onnx", verbosity);
  if (!parsed) {
    return false;
  }

  cudaStream_t profileStream;
  if (cudaStreamCreateWithFlags(&profileStream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    return false;
  }

  config->setProfileStream(profileStream);

  std::unique_ptr<IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }

  std::unique_ptr<IRuntime> runtime{
      createInferRuntime(sample::gLogger.getTRTLogger())};
  if (!runtime) {
    return false;
  }

  mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      samplesCommon::InferDeleter());
  if (!mEngine) {
    return false;
  }

  ASSERT(network->getNbInputs() == 1);
  mInputDims = network->getInput(0)->getDimensions();
  ASSERT(mInputDims.nbDims == 4);

  ASSERT(network->getNbOutputs() == 1);
  mOutputDims = network->getOutput(0)->getDimensions();
  ASSERT(mOutputDims.nbDims == 2);

  return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It
//! allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer() {
  float *h_input = nullptr, *h_output = nullptr;
  size_t input_sz = 0;
  float *d_input = nullptr, *d_output = nullptr;
  size_t output_sz = 0;
  size_t volume = 1;
  for (int i = 0; i < mEngine->getNbBindings(); i++) {
    auto bind_name = mEngine->getBindingName(i);
    sample::gLogInfo << "-------> bind_name = " << volume << std::endl;
    auto dims = mEngine->getBindingDimensions(i);
    nvinfer1::DataType type = mEngine->getBindingDataType(i);
    size_t volume = 1;
    for (int i = 0; i < dims.nbDims; i++) {
      volume *= dims.d[i];
    }
    sample::gLogInfo << "-------> data num = " << volume << std::endl;
    volume *= samplesCommon::getElementSize(type);
    sample::gLogInfo << "-------> data size = " << volume << std::endl;
    if (mEngine->bindingIsInput(i)) {
      h_input = (float *)malloc(volume);
      CHECK(cudaMalloc((void **)&d_input, volume));
      input_sz = volume;
    } else {
      h_output = (float *)malloc(volume);
      CHECK(cudaMalloc((void **)&d_output, volume));
      output_sz = volume;                  
    }
  }

  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext());
  if (!context) {
    return false;
  }

  // Read the input data into the managed buffers
  ASSERT(mParams.inputTensorNames.size() == 1);
  if (!processInput(h_input)) {
    return false;
  }
  CHECK(cudaMemcpy(d_input, h_input, input_sz, cudaMemcpyHostToDevice));

  // Memcpy from host input buffers to device input buffers
  std::vector<void *> data_v{static_cast<void *>(d_input),
                             static_cast<void *>(d_output)};
  bool status = context->executeV2(data_v.data());
  if (!status) {
    return false;
  }

  // Memcpy from device output buffers to host output buffers
  CHECK(cudaMemcpy(h_output, d_output, output_sz, cudaMemcpyDeviceToHost));

  // Verify results
  if (!verifyOutput(h_output)) {
    return false;
  }

  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(float *buffers) {
  int inputC = mInputDims.d[1];
  int inputH = mInputDims.d[2];
  int inputW = mInputDims.d[3];
  sample::gLogInfo << "processInput(): " << inputC << ", " << inputH << ", "
                   << inputW;
  // Read a random digit file
  srand(unsigned(time(nullptr)));
  std::vector<uint8_t> fileData(inputC * inputH * inputW);

  int max{0};
  std::string magic{""};

  std::ifstream infile(locateFile("airliner.ppm", mParams.dataDirs), std::ifstream::binary);
  ASSERT(infile.is_open() &&
         "Attempting to read from a file that is not open.");
  infile >> magic >> inputW >> inputH >> max;
  infile.seekg(1, infile.cur);
  infile.read(reinterpret_cast<char *>(fileData.data()),
              inputW * inputH * inputC);
  sample::gLogInfo << "processInput(): " << inputC << ", " << inputH << ", "
                   << inputW;
  // mNumber = rand() % 10;
  // readPGMFile(locateFile("airliner.pgm", mParams.dataDirs),
  //             fileData.data(), inputH, inputW);

  // Print an ascii representation
  // sample::gLogInfo << "Input:" << std::endl;
  // for (int i = 0; i < inputH * inputW; i++) {
  //   sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26])
  //                    << (((i + 1) % inputW) ? "" : "\n");
  // }
  // sample::gLogInfo << std::endl;



    // Convert HWC to CHW and Normalize
    for (int c = 0; c < inputC; ++c)
    {
        for (int h = 0; h < inputH; ++h)
        {
            for (int w = 0; w < inputW; ++w)
            {
                int dstIdx = c * inputH * inputW + h * inputW + w;
                int srcIdx = h * inputW * inputC + w * inputC + c;
                // This equation include 3 steps
                // 1. Scale Image to range [0.f, 1.0f]
                // 2. Normalize Image using per channel Mean and per channel Standard Deviation
                // 3. Shuffle HWC to CHW form
                buffers[dstIdx] = (2.0 / 255.0) * static_cast<float>(fileData[srcIdx]) - 1.0;
            }
        }
    }

  // for (int i = 0; i < inputH * inputW; i++) {
  //   buffers[i] = 1.0 - float(fileData[i] / 255.0);
  // }

  return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(float *output) {
  const int outputSize = mOutputDims.d[1];
   sample::gLogInfo << "outputSize = " << outputSize << std::endl;
  float val{0.0f};
  int idx{0};



    // auto inds = samplesCommon::argMagnitudeSort(output, output + outputSize);

  for (int i = 0; i < outputSize; i++) {
    //  output[i] /= sum;
    val = std::max(val, output[i]);
    if (val == output[i]) {
      idx = i;
    }
  }

    // read reference lables to generate prediction lables
    std::vector<std::string> referenceVector;
    if (!samplesCommon::readReferenceFile(locateFile("reference_labels.txt", mParams.dataDirs) , referenceVector))
    {
        sample::gLogError << "Unable to read reference file: " << std::endl;
        return false;
    }

    // std::vector<std::string> top5Result = samplesCommon::classify(referenceVector, output, 5);

    sample::gLogInfo << "SampleINT8API result: p[" << idx << "] = " << val << std::endl;
    sample::gLogInfo << "SampleINT8API result: Detected:" << referenceVector[idx] << std::endl;
    // for (int i = 1; i <= 5; ++i)
    // {
    //     sample::gLogInfo << "[" << i << "]  " << top5Result[i - 1] << std::endl;
    // }

    return true;

  // // Calculate Softmax
  // float sum{0.0f};
  // for (int i = 0; i < outputSize; i++) {
  //   output[i] = exp(output[i]);
  //   sum += output[i];
  // }

  // sample::gLogInfo << "Output:" << std::endl;
  // for (int i = 0; i < outputSize; i++) {
  //   // output[i] /= sum;
  //   val = std::max(val, output[i]);
  //   if (val == output[i]) {
  //     idx = i;
  //   }

  //   sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5)
  //                    << std::setprecision(4) << output[i] << " "
  //                    << "Class " << i << ": "
  //                    << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
  //                    << std::endl;
  // }
  // sample::gLogInfo << std::endl;

  // return idx == mNumber && val > 0.9f;
}

int main(int argc, char** argv)
{

    samplesCommon::OnnxSampleParams params;
    params.dataDirs.push_back("data/resnet50/");
    params.dataDirs.push_back("data/int8_api/");
    // params.onnxFileName = "ResNet50.onnx";
    // params.onnxFileName = "quant_resnet50.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = -1;
    params.int8 = 0;
    params.fp16 = 0;

    SampleOnnxMNIST sample(params);

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!sample.build()) {
      sample::gLogError << "build failed";
    }
    if (!sample.infer()) {
      sample::gLogError << "infer failed";
    }
    return 0;
}
