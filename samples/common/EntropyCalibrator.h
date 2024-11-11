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

#ifndef ENTROPY_CALIBRATOR_H
#define ENTROPY_CALIBRATOR_H

#include "BatchStream.h"
#include "NvInfer.h"

//! \class EntropyCalibratorImpl
//!
//! \brief Implements common functionality for Entropy calibrators.
//!
template <typename TBatchStream>
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(
        TBatchStream stream, int firstBatch, std::string networkName, const char* inputBlobName, bool readCache = true)
        : mStream{stream}
        , mCalibrationTableName("CalibrationTable" + networkName)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
      std::cout << "--------------- [info] EntropyCalibratorImpl: firstBatch = "
                << firstBatch << ", networkName = " << networkName
                << ", inputBlobName = " << inputBlobName << std::endl;
      nvinfer1::Dims dims = mStream.getDims();
      std::cout << "--------------- [info] EntropyCalibratorImpl: dims.nbDims = "
          << dims.nbDims << ", dims.d[3] = " << dims.d[3]
          << ", dims.d[2] = " << dims.d[2] << ", dims.d[1] = " << dims.d[1]
          << ", dims.d[0] = " << dims.d[0] << std::endl;
      mInputCount = samplesCommon::volume(dims);
      CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
      mStream.reset(firstBatch);
        
    }

    virtual ~EntropyCalibratorImpl()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const noexcept
    {
        std::cout << "--------------- [info] EntropyCalibratorImpl<...>::getBatchSize()" << std::endl;
        return mStream.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
    {
        std::cout << "--------------- [info] EntropyCalibratorImpl<...>::getBatch()" << std::endl;
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        ASSERT(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept
    {
        std::cout << "--------------- [info] EntropyCalibratorImpl<...>::getBatch()" << std::endl;
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept
    {
        std::cout << "--------------- [info] EntropyCalibratorImpl<...>::getBatch()" << std::endl;
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    TBatchStream mStream;
    size_t mInputCount;
    std::string mCalibrationTableName;
    const char* mInputBlobName;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//!
//! \brief Implements Entropy calibrator 2.
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
template <typename TBatchStream>
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(
        TBatchStream stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
        std::cout << "------------------- [info] Int8EntropyCalibrator2::Int8EntropyCalibrator2()" << std::endl;
    }

    int getBatchSize() const noexcept override
    {
        std::cout << "------------------- [info] Int8EntropyCalibrator2::getBatchSize()" << std::endl;
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        std::cout << "------------------- [info] Int8EntropyCalibrator2::getBatch()" << std::endl;
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        std::cout << "------------------- [info] Int8EntropyCalibrator2::readCalibrationCache()" << std::endl;
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        std::cout << "------------------- [info] Int8EntropyCalibrator2::writeCalibrationCache()" << std::endl;
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};

#endif // ENTROPY_CALIBRATOR_H
