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
#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <stdio.h>
#include <vector>

#define BATCH_STREAM_DEBUG
class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

class MNISTBatchStream : public IBatchStream
{
public:
    MNISTBatchStream(int batchSize, int maxBatches, const std::string& dataFile, const std::string& labelsFile,
        const std::vector<std::string>& directories)
        : mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{3, {1, 28, 28}} //!< We already know the dimensions of MNIST images.
    {
#ifdef BATCH_STREAM_DEBUG
      std::cout << "----------- [info] MNISTBatchStream::mBatchSize = "
                << mBatchSize << ", mMaxBatches = " << mMaxBatches << std::endl;
#endif        
        readDataFile(locateFile(dataFile, directories));
        readLabelsFile(locateFile(labelsFile, directories));
    }

    void reset(int firstBatch) override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::reset()" << std::endl;
#endif
        mBatchCount = firstBatch;
    }

    bool next() override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::next()" << std::endl;
#endif
       
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::skip()" << std::endl;
#endif
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::getBatch()" << std::endl;
#endif
        return mData.data() + (mBatchCount * mBatchSize * samplesCommon::volume(mDims));
    }

    float* getLabels() override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::getLabels()" << std::endl;
#endif
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::getBatchesRead()" << std::endl;
#endif
        return mBatchCount;
    }

    int getBatchSize() const override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::getBatchSize():  " <<  mBatchSize << std::endl;
#endif
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::getDims()" << std::endl;
#endif
        return nvinfer1::Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}};
    }

private:
    void readDataFile(const std::string& dataFilePath)
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::readDataFile()" << std::endl;
#endif
        std::ifstream file{dataFilePath.c_str(), std::ios::binary};

        int magicNumber, numImages, imageH, imageW;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = samplesCommon::swapEndianness(magicNumber);
        ASSERT(magicNumber == 2051 && "Magic Number does not match the expected value for an MNIST image set");

        // Read number of images and dimensions
        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        file.read(reinterpret_cast<char*>(&imageH), sizeof(imageH));
        file.read(reinterpret_cast<char*>(&imageW), sizeof(imageW));

        numImages = samplesCommon::swapEndianness(numImages);
        imageH = samplesCommon::swapEndianness(imageH);
        imageW = samplesCommon::swapEndianness(imageW);
        std::cout << "------- [info] numImages = " << numImages
                  << ", imageH = " << imageH << ", imageW = " << imageW
                  << std::endl;
        // The MNIST data is made up of unsigned bytes, so we need to cast to float and normalize.
        int numElements = numImages * imageH * imageW;
        std::vector<uint8_t> rawData(numElements);
        file.read(reinterpret_cast<char*>(rawData.data()), numElements * sizeof(uint8_t));
        mData.resize(numElements);
        std::transform(
            rawData.begin(), rawData.end(), mData.begin(), [](uint8_t val) { return static_cast<float>(val) / 255.f; });
    }

    void readLabelsFile(const std::string& labelsFilePath)
    {
#ifdef BATCH_STREAM_DEBUG
    std::cout << "----------- [info] MNISTBatchStream::readLabelsFile()" << std::endl;
#endif
       
        std::ifstream file{labelsFilePath.c_str(), std::ios::binary};
        int magicNumber, numImages;
        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        // All values in the MNIST files are big endian.
        magicNumber = samplesCommon::swapEndianness(magicNumber);
        ASSERT(magicNumber == 2049 && "Magic Number does not match the expected value for an MNIST labels file");

        file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        numImages = samplesCommon::swapEndianness(numImages);
        std::cout << "------- [info] numImages = " << numImages << std::endl;
        std::vector<uint8_t> rawLabels(numImages);
        file.read(reinterpret_cast<char*>(rawLabels.data()), numImages * sizeof(uint8_t));
        mLabels.resize(numImages);
        std::transform(
            rawLabels.begin(), rawLabels.end(), mLabels.begin(), [](uint8_t val) { return static_cast<float>(val); });
    }

    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    nvinfer1::Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
};

#endif
