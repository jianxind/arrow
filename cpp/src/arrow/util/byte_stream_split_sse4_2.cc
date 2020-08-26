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

#include "arrow/util/byte_stream_split_sse4_2.h"
#include "arrow/util/simd.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace util {
namespace internal {

namespace {

template <typename T>
void ByteStreamSplitDecodeSse42Internal(const uint8_t* data, int64_t num_values,
                                        int64_t stride, T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);

  const int64_t size = num_values * sizeof(T);
  constexpr int64_t kBlockSize = sizeof(__m128i) * kNumStreams;
  const int64_t num_blocks = size / kBlockSize;
  uint8_t* output_data = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  // This helps catch if the simd-based processing overflows into the suffix
  // since almost surely a test would fail.
  const int64_t num_processed_elements = (num_blocks * kBlockSize) / kNumStreams;
  for (int64_t i = num_processed_elements; i < num_values; ++i) {
    uint8_t gathered_byte_data[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byte_index = b * stride + i;
      gathered_byte_data[b] = data[byte_index];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gathered_byte_data[0]);
  }

  // The blocks get processed hierahically using the unpack intrinsics.
  // Example with four streams:
  // Stage 1: AAAA BBBB CCCC DDDD
  // Stage 2: ACAC ACAC BDBD BDBD
  // Stage 3: ABCD ABCD ABCD ABCD
  __m128i stage[kNumStreamsLog2 + 1U][kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < num_blocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(&data[i * sizeof(__m128i) + j * stride]));
    }
    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            _mm_unpacklo_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            _mm_unpackhi_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }
    for (size_t j = 0; j < kNumStreams; ++j) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(
                           &output_data[(i * kNumStreams + j) * sizeof(__m128i)]),
                       stage[kNumStreamsLog2][j]);
    }
  }
}

template <typename T>
void ByteStreamSplitEncodeSse42Internal(const uint8_t* raw_values,
                                        const size_t num_values,
                                        uint8_t* output_buffer_raw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  __m128i stage[3][kNumStreams];
  __m128i final_result[kNumStreams];

  const size_t size = num_values * sizeof(T);
  constexpr size_t kBlockSize = sizeof(__m128i) * kNumStreams;
  const size_t num_blocks = size / kBlockSize;
  const __m128i* raw_values_sse = reinterpret_cast<const __m128i*>(raw_values);
  __m128i* output_buffer_streams[kNumStreams];
  for (size_t i = 0; i < kNumStreams; ++i) {
    output_buffer_streams[i] =
        reinterpret_cast<__m128i*>(&output_buffer_raw[num_values * i]);
  }

  // First handle suffix.
  const size_t num_processed_elements = (num_blocks * kBlockSize) / sizeof(T);
  for (size_t i = num_processed_elements; i < num_values; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byte_in_value = raw_values[i * kNumStreams + j];
      output_buffer_raw[j * num_values + i] = byte_in_value;
    }
  }
  // The current shuffling algorithm diverges for float and double types but the compiler
  // should be able to remove the branch since only one path is taken for each template
  // instantiation.
  // Example run for floats:
  // Step 0, copy:
  //   0: ABCD ABCD ABCD ABCD 1: ABCD ABCD ABCD ABCD ...
  // Step 1: _mm_unpacklo_epi8 and mm_unpackhi_epi8:
  //   0: AABB CCDD AABB CCDD 1: AABB CCDD AABB CCDD ...
  //   0: AAAA BBBB CCCC DDDD 1: AAAA BBBB CCCC DDDD ...
  // Step 3: __mm_unpacklo_epi8 and _mm_unpackhi_epi8:
  //   0: AAAA AAAA BBBB BBBB 1: CCCC CCCC DDDD DDDD ...
  // Step 4: __mm_unpacklo_epi64 and _mm_unpackhi_epi64:
  //   0: AAAA AAAA AAAA AAAA 1: BBBB BBBB BBBB BBBB ...
  for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
    // First copy the data to stage 0.
    for (size_t i = 0; i < kNumStreams; ++i) {
      stage[0][i] = _mm_loadu_si128(&raw_values_sse[block_index * kNumStreams + i]);
    }

    // The shuffling of bytes is performed through the unpack intrinsics.
    // In my measurements this gives better performance then an implementation
    // which uses the shuffle intrinsics.
    for (size_t stage_lvl = 0; stage_lvl < 2U; ++stage_lvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        stage[stage_lvl + 1][i * 2] =
            _mm_unpacklo_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
        stage[stage_lvl + 1][i * 2 + 1] =
            _mm_unpackhi_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
      }
    }
    if (kNumStreams == 8U) {
      // This is the path for double.
      __m128i tmp[8];
      for (size_t i = 0; i < 4; ++i) {
        tmp[i * 2] = _mm_unpacklo_epi32(stage[2][i], stage[2][i + 4]);
        tmp[i * 2 + 1] = _mm_unpackhi_epi32(stage[2][i], stage[2][i + 4]);
      }

      for (size_t i = 0; i < 4; ++i) {
        final_result[i * 2] = _mm_unpacklo_epi32(tmp[i], tmp[i + 4]);
        final_result[i * 2 + 1] = _mm_unpackhi_epi32(tmp[i], tmp[i + 4]);
      }
    } else {
      // this is the path for float.
      __m128i tmp[4];
      for (size_t i = 0; i < 2; ++i) {
        tmp[i * 2] = _mm_unpacklo_epi8(stage[2][i * 2], stage[2][i * 2 + 1]);
        tmp[i * 2 + 1] = _mm_unpackhi_epi8(stage[2][i * 2], stage[2][i * 2 + 1]);
      }
      for (size_t i = 0; i < 2; ++i) {
        final_result[i * 2] = _mm_unpacklo_epi64(tmp[i], tmp[i + 2]);
        final_result[i * 2 + 1] = _mm_unpackhi_epi64(tmp[i], tmp[i + 2]);
      }
    }
    for (size_t i = 0; i < kNumStreams; ++i) {
      _mm_storeu_si128(&output_buffer_streams[i][block_index], final_result[i]);
    }
  }
}

}  // namespace

void ByteStreamSplitEncodeSse42Float(const uint8_t* raw_values, const size_t num_values,
                                     uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeSse42Internal<float>(raw_values, num_values,
                                                   output_buffer_raw);
}

void ByteStreamSplitEncodeSse42Double(const uint8_t* raw_values, const size_t num_values,
                                      uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeSse42Internal<double>(raw_values, num_values,
                                                    output_buffer_raw);
}

void ByteStreamSplitDecodeSse42Float(const uint8_t* data, int64_t num_values,
                                     int64_t stride, float* out) {
  return ByteStreamSplitDecodeSse42Internal<float>(data, num_values, stride, out);
}

void ByteStreamSplitDecodeSse42Double(const uint8_t* data, int64_t num_values,
                                      int64_t stride, double* out) {
  return ByteStreamSplitDecodeSse42Internal<double>(data, num_values, stride, out);
}

}  // namespace internal
}  // namespace util
}  // namespace arrow
