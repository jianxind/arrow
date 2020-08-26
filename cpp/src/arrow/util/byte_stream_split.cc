// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/util/byte_stream_split.h"
#include "arrow/util/dispatch.h"
#include "arrow/util/simd.h"
#include "arrow/util/ubsan.h"

#if defined(ARROW_HAVE_RUNTIME_SSE4_2)
#include "arrow/util/byte_stream_split_sse4_2.h"
#endif
#if defined(ARROW_HAVE_RUNTIME_AVX2)
#include "arrow/util/byte_stream_split_avx2.h"
#endif
#if defined(ARROW_HAVE_RUNTIME_AVX512)
#include "arrow/util/byte_stream_split_avx512.h"
#endif

using arrow::internal::DispatchLevel;
using arrow::internal::DynamicDispatch;

namespace arrow {
namespace util {
namespace internal {

namespace {

template <typename T>
void ByteStreamSplitEncodeScalar(const uint8_t* raw_values, const size_t num_values,
                                 uint8_t* output_buffer_raw) {
  constexpr size_t kNumStreams = sizeof(T);
  for (size_t i = 0U; i < num_values; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byte_in_value = raw_values[i * kNumStreams + j];
      output_buffer_raw[j * num_values + i] = byte_in_value;
    }
  }
}

template <typename T>
void ByteStreamSplitDecodeScalar(const uint8_t* data, int64_t num_values, int64_t stride,
                                 T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  auto output_buffer_raw = reinterpret_cast<uint8_t*>(out);

  for (int64_t i = 0; i < num_values; ++i) {
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byte_index = b * stride + i;
      output_buffer_raw[i * kNumStreams + b] = data[byte_index];
    }
  }
}

template <typename T>
struct ByteStreamSplitEncodeDynamicFunction {
  using FunctionType = decltype(&ByteStreamSplitEncodeScalar<T>);

  static std::vector<std::pair<DispatchLevel, FunctionType>> implementations() {
    return {
      { DispatchLevel::NONE, ByteStreamSplitEncodeScalar<T> }
#if defined(ARROW_HAVE_RUNTIME_SSE4_2)
      , { DispatchLevel::SSE4_2, ByteStreamSplitEncodeSse42<T> }
#endif
#if defined(ARROW_HAVE_RUNTIME_AVX2)
      , { DispatchLevel::AVX2, ByteStreamSplitEncodeAvx2<T> }
#endif
#if defined(ARROW_HAVE_RUNTIME_AVX512)
      , { DispatchLevel::AVX512, ByteStreamSplitEncodeAvx512<T> }
#endif
    };
  }
};

template <typename T>
struct ByteStreamSplitDecodeDynamicFunction {
  using FunctionType = decltype(&ByteStreamSplitDecodeScalar<T>);

  static std::vector<std::pair<DispatchLevel, FunctionType>> implementations() {
    return {
      { DispatchLevel::NONE, ByteStreamSplitDecodeScalar<T> }
#if defined(ARROW_HAVE_RUNTIME_SSE4_2)
      , { DispatchLevel::SSE4_2, ByteStreamSplitDecodeSse42<T> }
#endif
#if defined(ARROW_HAVE_RUNTIME_AVX2)
      , { DispatchLevel::AVX2, ByteStreamSplitDecodeAvx2<T> }
#endif
#if defined(ARROW_HAVE_RUNTIME_AVX512)
      , { DispatchLevel::AVX512, ByteStreamSplitDecodeAvx512<T> }
#endif
    };
  }
};

}  // namespace

void ByteStreamSplitEncodeFloat(const uint8_t* raw_values, const size_t num_values,
                                uint8_t* output_buffer_raw) {
  static DynamicDispatch<ByteStreamSplitEncodeDynamicFunction<float>> dispatch;
  return dispatch.func(raw_values, num_values, output_buffer_raw);
}

void ByteStreamSplitEncodeDouble(const uint8_t* raw_values, const size_t num_values,
                                 uint8_t* output_buffer_raw) {
  static DynamicDispatch<ByteStreamSplitEncodeDynamicFunction<double>> dispatch;
  return dispatch.func(raw_values, num_values, output_buffer_raw);
}

void ByteStreamSplitDecodeFloat(const uint8_t* data, int64_t num_values, int64_t stride,
                                float* out) {
  static DynamicDispatch<ByteStreamSplitDecodeDynamicFunction<float>> dispatch;
  return dispatch.func(data, num_values, stride, out);
}

void ByteStreamSplitDecodeDouble(const uint8_t* data, int64_t num_values, int64_t stride,
                                 double* out) {
  static DynamicDispatch<ByteStreamSplitDecodeDynamicFunction<double>> dispatch;
  return dispatch.func(data, num_values, stride, out);
}

}  // namespace internal
}  // namespace util
}  // namespace arrow
