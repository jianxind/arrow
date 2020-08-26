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

#include "arrow/util/byte_stream_split_avx2.h"
#include "arrow/util/byte_stream_split_sse4_2.h"
#include "arrow/util/simd.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace util {
namespace internal {

namespace {

template <typename T>
void ByteStreamSplitDecodeAvx2Internal(const uint8_t* data, int64_t num_values,
                                       int64_t stride, T* out) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  constexpr size_t kNumStreamsLog2 = (kNumStreams == 8U ? 3U : 2U);

  const int64_t size = num_values * sizeof(T);
  constexpr int64_t kBlockSize = sizeof(__m256i) * kNumStreams;
  if (size < kBlockSize)  // Back to SSE for small size
    return ByteStreamSplitDecodeSse42(data, num_values, stride, out);
  const int64_t num_blocks = size / kBlockSize;
  uint8_t* output_data = reinterpret_cast<uint8_t*>(out);

  // First handle suffix.
  const int64_t num_processed_elements = (num_blocks * kBlockSize) / kNumStreams;
  for (int64_t i = num_processed_elements; i < num_values; ++i) {
    uint8_t gathered_byte_data[kNumStreams];
    for (size_t b = 0; b < kNumStreams; ++b) {
      const size_t byte_index = b * stride + i;
      gathered_byte_data[b] = data[byte_index];
    }
    out[i] = arrow::util::SafeLoadAs<T>(&gathered_byte_data[0]);
  }

  // Processed hierahically using unpack intrinsics, then permute intrinsics.
  __m256i stage[kNumStreamsLog2 + 1U][kNumStreams];
  __m256i final_result[kNumStreams];
  constexpr size_t kNumStreamsHalf = kNumStreams / 2U;

  for (int64_t i = 0; i < num_blocks; ++i) {
    for (size_t j = 0; j < kNumStreams; ++j) {
      stage[0][j] = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(&data[i * sizeof(__m256i) + j * stride]));
    }

    for (size_t step = 0; step < kNumStreamsLog2; ++step) {
      for (size_t j = 0; j < kNumStreamsHalf; ++j) {
        stage[step + 1U][j * 2] =
            _mm256_unpacklo_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
        stage[step + 1U][j * 2 + 1U] =
            _mm256_unpackhi_epi8(stage[step][j], stage[step][kNumStreamsHalf + j]);
      }
    }

    if (kNumStreams == 8U) {
      // path for double, 128i index:
      //   {0x00, 0x08}, {0x01, 0x09}, {0x02, 0x0A}, {0x03, 0x0B},
      //   {0x04, 0x0C}, {0x05, 0x0D}, {0x06, 0x0E}, {0x07, 0x0F},
      final_result[0] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00100000);
      final_result[1] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00100000);
      final_result[2] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][4],
                                                  stage[kNumStreamsLog2][5], 0b00100000);
      final_result[3] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][6],
                                                  stage[kNumStreamsLog2][7], 0b00100000);
      final_result[4] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00110001);
      final_result[5] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00110001);
      final_result[6] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][4],
                                                  stage[kNumStreamsLog2][5], 0b00110001);
      final_result[7] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][6],
                                                  stage[kNumStreamsLog2][7], 0b00110001);
    } else {
      // path for float, 128i index:
      //   {0x00, 0x04}, {0x01, 0x05}, {0x02, 0x06}, {0x03, 0x07}
      final_result[0] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00100000);
      final_result[1] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00100000);
      final_result[2] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][0],
                                                  stage[kNumStreamsLog2][1], 0b00110001);
      final_result[3] = _mm256_permute2x128_si256(stage[kNumStreamsLog2][2],
                                                  stage[kNumStreamsLog2][3], 0b00110001);
    }

    for (size_t j = 0; j < kNumStreams; ++j) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(
                              &output_data[(i * kNumStreams + j) * sizeof(__m256i)]),
                          final_result[j]);
    }
  }
}

template <typename T>
void ByteStreamSplitEncodeAvx2Internal(const uint8_t* raw_values, const size_t num_values,
                                       uint8_t* output_buffer_raw) {
  constexpr size_t kNumStreams = sizeof(T);
  static_assert(kNumStreams == 4U || kNumStreams == 8U, "Invalid number of streams.");
  if (kNumStreams == 8U)  // Back to SSE, currently no path for double.
    return ByteStreamSplitEncodeSse42<T>(raw_values, num_values, output_buffer_raw);

  const size_t size = num_values * sizeof(T);
  constexpr size_t kBlockSize = sizeof(__m256i) * kNumStreams;
  if (size < kBlockSize)  // Back to SSE for small size
    return ByteStreamSplitEncodeSse42<T>(raw_values, num_values, output_buffer_raw);
  const size_t num_blocks = size / kBlockSize;
  const __m256i* raw_values_simd = reinterpret_cast<const __m256i*>(raw_values);
  __m256i* output_buffer_streams[kNumStreams];

  for (size_t i = 0; i < kNumStreams; ++i) {
    output_buffer_streams[i] =
        reinterpret_cast<__m256i*>(&output_buffer_raw[num_values * i]);
  }

  // First handle suffix.
  const size_t num_processed_elements = (num_blocks * kBlockSize) / sizeof(T);
  for (size_t i = num_processed_elements; i < num_values; ++i) {
    for (size_t j = 0U; j < kNumStreams; ++j) {
      const uint8_t byte_in_value = raw_values[i * kNumStreams + j];
      output_buffer_raw[j * num_values + i] = byte_in_value;
    }
  }

  // Path for float.
  // 1. Processed hierahically to 32i blcok using the unpack intrinsics.
  // 2. Pack 128i block using _mm256_permutevar8x32_epi32.
  // 3. Pack final 256i block with _mm256_permute2x128_si256.
  constexpr size_t kNumUnpack = 3U;
  __m256i stage[kNumUnpack + 1][kNumStreams];
  static const __m256i kPermuteMask =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);
  __m256i permute[kNumStreams];
  __m256i final_result[kNumStreams];

  for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
    for (size_t i = 0; i < kNumStreams; ++i) {
      stage[0][i] = _mm256_loadu_si256(&raw_values_simd[block_index * kNumStreams + i]);
    }

    for (size_t stage_lvl = 0; stage_lvl < kNumUnpack; ++stage_lvl) {
      for (size_t i = 0; i < kNumStreams / 2U; ++i) {
        stage[stage_lvl + 1][i * 2] =
            _mm256_unpacklo_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
        stage[stage_lvl + 1][i * 2 + 1] =
            _mm256_unpackhi_epi8(stage[stage_lvl][i * 2], stage[stage_lvl][i * 2 + 1]);
      }
    }

    for (size_t i = 0; i < kNumStreams; ++i) {
      permute[i] = _mm256_permutevar8x32_epi32(stage[kNumUnpack][i], kPermuteMask);
    }

    final_result[0] = _mm256_permute2x128_si256(permute[0], permute[2], 0b00100000);
    final_result[1] = _mm256_permute2x128_si256(permute[0], permute[2], 0b00110001);
    final_result[2] = _mm256_permute2x128_si256(permute[1], permute[3], 0b00100000);
    final_result[3] = _mm256_permute2x128_si256(permute[1], permute[3], 0b00110001);

    for (size_t i = 0; i < kNumStreams; ++i) {
      _mm256_storeu_si256(&output_buffer_streams[i][block_index], final_result[i]);
    }
  }
}
}  // namespace

void ByteStreamSplitEncodeAvx2Float(const uint8_t* raw_values, const size_t num_values,
                                    uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeAvx2Internal<float>(raw_values, num_values,
                                                  output_buffer_raw);
}

void ByteStreamSplitEncodeAvx2Double(const uint8_t* raw_values, const size_t num_values,
                                     uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeAvx2Internal<double>(raw_values, num_values,
                                                   output_buffer_raw);
}

void ByteStreamSplitDecodeAvx2Float(const uint8_t* data, int64_t num_values,
                                    int64_t stride, float* out) {
  return ByteStreamSplitDecodeAvx2Internal<float>(data, num_values, stride, out);
}

void ByteStreamSplitDecodeAvx2Double(const uint8_t* data, int64_t num_values,
                                     int64_t stride, double* out) {
  return ByteStreamSplitDecodeAvx2Internal<double>(data, num_values, stride, out);
}

}  // namespace internal
}  // namespace util
}  // namespace arrow
