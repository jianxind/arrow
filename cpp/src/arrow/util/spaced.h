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

#pragma once

#include "arrow/util/align_util.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/simd.h"
#include "arrow/util/spaced_avx2.h"
#include "arrow/util/spaced_sse.h"

namespace arrow {
namespace util {
namespace internal {

template <typename T>
int SpacedCompressScalar(const T* values, int num_values, const uint8_t* valid_bits,
                         int64_t valid_bits_offset, T* output) {
  int num_valid_values = 0;
  arrow::internal::BitmapReader valid_bits_reader(valid_bits, valid_bits_offset,
                                                  num_values);
  for (int i = 0; i < num_values; i++) {
    if (valid_bits_reader.IsSet()) {
      output[num_valid_values++] = values[i];
    }
    valid_bits_reader.Next();
  }
  return num_valid_values;
}

template <typename T>
int SpacedExpandScalar(T* buffer, int num_values, int null_count,
                       const uint8_t* valid_bits, int64_t valid_bits_offset) {
  const int values_read = num_values - null_count;

  // Depending on the number of nulls, some of the value slots in buffer may
  // be uninitialized, and this will cause valgrind warnings / potentially UB
  memset(static_cast<void*>(buffer + values_read), 0,
         (num_values - values_read) * sizeof(T));

  // Add spacing for null entries. As we have filled the buffer from the front,
  // we need to add the spacing from the back.
  int values_to_move = values_read - 1;
  // We stop early on one of two conditions:
  // 1. There are no more null values that need spacing.  Note we infer this
  //     backwards, when 'i' is equal to 'values_to_move' it indicates
  //    all nulls have been consumed.
  // 2. There are no more non-null values that need to move which indicates
  //    all remaining slots are null, so their exact value doesn't matter.
  for (int i = num_values - 1; (i > values_to_move) && (values_to_move >= 0); i--) {
    if (BitUtil::GetBit(valid_bits, valid_bits_offset + i)) {
      buffer[i] = buffer[values_to_move];
      values_to_move--;
    }
  }
  return num_values;
}

#if defined(ARROW_HAVE_SSE4_2)
// Compress the buffer to spaced with help of _mm_shuffle_epi8 API, the shuffle control
// mask achieved by a lookup table for better performance
template <typename T>
int SpacedCompressSseShuffle(const T* values, int num_values, const uint8_t* valid_bits,
                             int64_t valid_bits_offset, T* output) {
  assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 1);
  // [(2 128i block/8 epi32), (4 128i/8 epi64), (1 128i/16 epi8)] for each batch
  constexpr size_t kBatchSize = (sizeof(T) == 1) ? sizeof(__m128i) : 8;
  constexpr size_t kBatchByteSize = kBatchSize / 8;
  int num_valid_values = 0;
  // Loop the valid bits with batch bytes align way
  const auto p = arrow::internal::BitmapWordAlign<kBatchByteSize>(
      valid_bits, valid_bits_offset, num_values);
  int valid_values_batch;

  // First handle the leading bits
  const int leading_bits = static_cast<int>(p.leading_bits);
  if (leading_bits > 0) {
    valid_values_batch =
        SpacedCompressScalar(values, leading_bits, valid_bits, valid_bits_offset, output);
    // Step the count and idx
    output += valid_values_batch;
    num_valid_values += valid_values_batch;
    values += leading_bits;
  }

  // The aligned parts can fill into batches
  const uint8_t* aligned_bits = p.aligned_start;
  auto aligned_words = p.aligned_words;
  while (aligned_words-- > 0) {
    const uint8_t valid_byte_value = aligned_bits[0];

    // Compiler able to pick the path at instantiation time
    if (sizeof(T) == 1) {
      // Path for epi8, 16 epi8 one batch, two bytes in valid_bits
      const uint64_t* kMask64ThinTable =
          reinterpret_cast<const uint64_t*>(arrow::internal::kMask64SseCompressEpi8Thin);
      const __m128i* kMask128ThinCompactTable =
          reinterpret_cast<const __m128i*>(arrow::internal::kMask128SseEpi8ThinCompact);

      const uint8_t valid_count_low = BitUtil::kBytePopcount[valid_byte_value];
      const uint8_t valid_byte_value_high = aligned_bits[1];
      const uint8_t valid_count_high = BitUtil::kBytePopcount[valid_byte_value_high];

      // Thin table used to avoid the large full table(1M), the addtional step is it need
      // add back the offset of high and compact two parts
      const __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
      __m128i mask = _mm_set_epi64x(kMask64ThinTable[valid_byte_value_high],
                                    kMask64ThinTable[valid_byte_value]);
      mask = _mm_add_epi8(mask, _mm_set_epi32(0x08080808, 0x08080808, 0x0, 0x0));
      const __m128i pruned = _mm_shuffle_epi8(src, mask);
      // Compact the final result
      const __m128i result =
          _mm_shuffle_epi8(pruned, kMask128ThinCompactTable[valid_count_low]);

      // Safe to store the spare null values which will be covered next batch
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output), result);
      valid_values_batch = valid_count_low + valid_count_high;
    } else if (sizeof(T) == 4) {
      // Path for epi32, compress from low to high, 4 bits each time
      const __m128i* kMask128Table =
          reinterpret_cast<const __m128i*>(arrow::internal::kMask128SseCompressEpi32);
      valid_values_batch = 0;
      for (int i = 0; i < 2; i++) {
        const uint8_t valid_value = (valid_byte_value >> (4 * i)) & 0x0F;
        const uint8_t valid_count = BitUtil::kBytePopcount[valid_value];

        const __m128i src =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(&values[4 * i]));
        const __m128i mask = _mm_load_si128(&kMask128Table[valid_value]);

        // Safe to store the spare null values which will be covered next batch
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[valid_values_batch]),
                         _mm_shuffle_epi8(src, mask));
        valid_values_batch += valid_count;
      }
    } else {
      // Path for epi64, compress from low to high, 2 bits each time
      const __m128i* kMask128Table =
          reinterpret_cast<const __m128i*>(arrow::internal::kMask128SseCompressEpi64);
      valid_values_batch = 0;
      for (int i = 0; i < 4; i++) {
        const uint8_t valid_value = (valid_byte_value >> (2 * i)) & 0x03;
        const uint8_t valid_count = BitUtil::kBytePopcount[valid_value];

        const __m128i src =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(&values[2 * i]));
        const __m128i mask = _mm_load_si128(&kMask128Table[valid_value]);

        // Safe to store the spare null values which will be covered next batch
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[valid_values_batch]),
                         _mm_shuffle_epi8(src, mask));
        valid_values_batch += valid_count;
      }
    }

    // Step the count and idx
    num_valid_values += valid_values_batch;
    output += valid_values_batch;
    values += kBatchSize;
    aligned_bits += kBatchByteSize;
  }

  // The remaining trailing bits
  const int trailing_bits = static_cast<int>(p.trailing_bits);
  if (trailing_bits > 0) {
    valid_values_batch = SpacedCompressScalar(values, trailing_bits, valid_bits,
                                              p.trailing_bit_offset, output);
    // Step the count and idx
    output += valid_values_batch;
    num_valid_values += valid_values_batch;
    values += trailing_bits;
  }

  return num_valid_values;
}

// Expand the spaced buffer with _mm_shuffle_epi8 API, the shuffle control mask achieved
// by a lookup table for better performance
template <typename T>
int SpacedExpandSseShuffle(T* buffer, int num_values, int null_count,
                           const uint8_t* valid_bits, int64_t valid_bits_offset) {
  assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 1);

  // [(2 128i block/8 epi32), (4 128i/8 epi64), (1 128i/16 epi8)] for each batch
  constexpr size_t kBatchSize = (sizeof(T) == 1) ? sizeof(__m128i) : 8;
  constexpr size_t kBatchByteSize = kBatchSize / 8;
  // Point to end as we add the spacing from the back.
  int idx_decode = num_values - null_count - 1;
  int idx_spaced = num_values - 1;
  int64_t idx_valid_bits = valid_bits_offset + idx_spaced;
  // Loop the valid bits with batch bytes align way
  const auto p = arrow::internal::BitmapWordAlign<kBatchByteSize>(
      valid_bits, valid_bits_offset, num_values);

  // The trailing bits
  auto trailing_bits = p.trailing_bits;
  while (trailing_bits-- > 0) {
    if (BitUtil::GetBit(valid_bits, idx_valid_bits)) {
      buffer[idx_spaced] = buffer[idx_decode];
      idx_decode--;
    } else {
      memset(static_cast<void*>(buffer + idx_spaced), 0, sizeof(T));
    }
    idx_spaced--;
    idx_valid_bits--;
  }

  // The aligned parts can fill into batches
  auto aligned_words = p.aligned_words;
  int64_t idx_valid_bytes = BitUtil::BytesForBits(idx_valid_bits) - 1;
  while ((idx_decode < idx_spaced) && (aligned_words-- > 0)) {
    const uint8_t valid_byte_value = valid_bits[idx_valid_bytes];
    idx_valid_bytes--;

    // Compiler able to pick the path at instantiation time
    if (sizeof(T) == 1) {
      // Path for epi8, 16 epi8 one batch, two bytes in valid_bits
      const uint64_t* kMask64ThinTable =
          reinterpret_cast<const uint64_t*>(arrow::internal::kMask64SseExpandEpi8Thin);
      const uint8_t valid_count_high = BitUtil::kBytePopcount[valid_byte_value];
      const uint8_t valid_byte_value_low = valid_bits[idx_valid_bytes];
      idx_valid_bytes--;
      const uint8_t valid_count_low = BitUtil::kBytePopcount[valid_byte_value_low];

      idx_decode -= valid_count_low + valid_count_high;
      idx_spaced -= kBatchSize;
      idx_valid_bits -= kBatchSize;

      // Thin table used to avoid the large full table(1M), the addtional step is it need
      // add back the count of low to high part
      const __m128i src =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(buffer + idx_decode + 1));
      __m128i mask = _mm_set_epi64x(kMask64ThinTable[valid_byte_value],
                                    kMask64ThinTable[valid_byte_value_low]);
      const __m128i mask_offset =
          _mm_set_epi8(valid_count_low, valid_count_low, valid_count_low, valid_count_low,
                       valid_count_low, valid_count_low, valid_count_low, valid_count_low,
                       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);
      mask = _mm_add_epi8(mask, mask_offset);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer + idx_spaced + 1),
                       _mm_shuffle_epi8(src, mask));

    } else if (sizeof(T) == 4) {
      // Path for epi32, expand from high to low, 4 bits each time
      const __m128i* kMask128Table =
          reinterpret_cast<const __m128i*>(arrow::internal::kMask128SseExpandEpi32);
      for (int i = 0; i < 2; i++) {
        const uint8_t valid_value = (valid_byte_value >> (4 * (2 - i - 1))) & 0x0F;
        const uint8_t valid_count = BitUtil::kBytePopcount[valid_value];
        idx_spaced -= 4;
        idx_valid_bits -= 4;
        idx_decode -= valid_count;

        const __m128i src =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(buffer + idx_decode + 1));
        const __m128i mask = _mm_load_si128(&kMask128Table[valid_value]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer + idx_spaced + 1),
                         _mm_shuffle_epi8(src, mask));
      }
    } else {
      // Path for epi64, expand from high to low, 2 bits each time
      const __m128i* kMask128Table =
          reinterpret_cast<const __m128i*>(arrow::internal::kMask128SseExpandEpi64);
      for (int i = 0; i < 4; i++) {
        const uint8_t valid_value = (valid_byte_value >> (2 * (4 - i - 1))) & 0x03;
        const uint8_t valid_count = BitUtil::kBytePopcount[valid_value];
        idx_spaced -= 2;
        idx_valid_bits -= 2;
        idx_decode -= valid_count;

        const __m128i src =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(buffer + idx_decode + 1));
        const __m128i mask = _mm_load_si128(&kMask128Table[valid_value]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer + idx_spaced + 1),
                         _mm_shuffle_epi8(src, mask));
      }
    }
  }

  // The remaining leading bits
  auto leading_bits = p.leading_bits;
  while (leading_bits-- > 0) {
    if (BitUtil::GetBit(valid_bits, idx_valid_bits)) {
      buffer[idx_spaced] = buffer[idx_decode];
      idx_decode--;
    } else {
      memset(static_cast<void*>(buffer + idx_spaced), 0, sizeof(T));
    }
    idx_spaced--;
    idx_valid_bits--;
  }

  return num_values;
}

template <typename T>
inline int SpacedCompressSse(const T* src, int num_values, const uint8_t* valid_bits,
                             int64_t valid_bits_offset, T* output) {
  if (sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 1) {
    return SpacedCompressSseShuffle<T>(src, num_values, valid_bits, valid_bits_offset,
                                       output);
  } else {
    return SpacedCompressScalar<T>(src, num_values, valid_bits, valid_bits_offset,
                                   output);
  }
}

template <typename T>
inline int SpacedExpandSse(T* buffer, int num_values, int null_count,
                           const uint8_t* valid_bits, int64_t valid_bits_offset) {
  if (sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 1) {
    return SpacedExpandSseShuffle<T>(buffer, num_values, null_count, valid_bits,
                                     valid_bits_offset);
  } else {
    return SpacedExpandScalar<T>(buffer, num_values, null_count, valid_bits,
                                 valid_bits_offset);
  }
}
#endif

#if defined(ARROW_HAVE_AVX2)
// Compress the buffer to spaced with help of _mm256_permutevar8x32_epi32 API, the
// shuffle control mask achieved by a lookup table for better performance
template <typename T>
int SpacedCompressAvx2Epi32Epi64(const T* values, int num_values,
                                 const uint8_t* valid_bits, int64_t valid_bits_offset,
                                 T* output) {
  assert(sizeof(T) == 4 || sizeof(T) == 8);
  // [(1 256i block/8 epi32), (2 256i/8 epi64)] for each batch
  constexpr size_t kBatchSize = 8;
  constexpr size_t kBatchByteSize = kBatchSize / 8;
  int num_valid_values = 0;
  // Loop the valid bits with batch bytes align way
  const auto p = arrow::internal::BitmapWordAlign<kBatchByteSize>(
      valid_bits, valid_bits_offset, num_values);
  int valid_values_batch;

  // First handle the leading bits
  const int leading_bits = static_cast<int>(p.leading_bits);
  if (leading_bits > 0) {
    valid_values_batch =
        SpacedCompressScalar(values, leading_bits, valid_bits, valid_bits_offset, output);
    // Step the count and idx
    output += valid_values_batch;
    num_valid_values += valid_values_batch;
    values += leading_bits;
  }

  // The aligned parts can fill into batches
  const uint8_t* aligned_bits = p.aligned_start;
  auto aligned_words = p.aligned_words;
  while (aligned_words-- > 0) {
    const uint8_t valid_byte_value = aligned_bits[0];

    // Compiler able to pick the path at instantiation time
    if (sizeof(T) == 4) {
      // Path for epi32, 8 epi32 for one m256i block, one byte in valid_bits
      const __m256i* kMask256Table =
          reinterpret_cast<const __m256i*>(arrow::internal::kMask256Avx2CompressEpi32);
      valid_values_batch = BitUtil::kBytePopcount[valid_byte_value];

      const __m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(values));
      const __m256i mask = _mm256_load_si256(&kMask256Table[valid_byte_value]);

      // Safe to store the spare null values which will be covered next batch
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(output),
                          _mm256_permutevar8x32_epi32(src, mask));
    } else {
      // Path for epi64, compress from low to high, 4 bits each time
      const __m256i* kMask256Table =
          reinterpret_cast<const __m256i*>(arrow::internal::kMask256Avx2CompressEpi64);
      valid_values_batch = 0;
      for (int i = 0; i < 2; i++) {
        const uint8_t valid_value = (valid_byte_value >> (4 * i)) & 0x0F;
        const uint8_t valid_count = BitUtil::kBytePopcount[valid_value];

        const __m256i src =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&values[4 * i]));
        const __m256i mask = _mm256_load_si256(&kMask256Table[valid_value]);

        // Safe to store the spare null values which will be covered next batch
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[valid_values_batch]),
                            _mm256_permutevar8x32_epi32(src, mask));
        valid_values_batch += valid_count;
      }
    }

    // Step the count and idx
    num_valid_values += valid_values_batch;
    output += valid_values_batch;
    values += kBatchSize;
    aligned_bits += kBatchByteSize;
  }

  // The remaining trailing bits
  const int trailing_bits = static_cast<int>(p.trailing_bits);
  if (trailing_bits > 0) {
    valid_values_batch = SpacedCompressScalar(values, trailing_bits, valid_bits,
                                              p.trailing_bit_offset, output);
    // Step the count and idx
    output += valid_values_batch;
    num_valid_values += valid_values_batch;
    values += trailing_bits;
  }

  return num_valid_values;
}

// Expand the spaced buffer with _mm256_permutevar8x32_epi32 API, the shuffle control
// mask achieved by a lookup table for better performance
template <typename T>
int SpacedExpandAvx2Epi32Epi64(T* buffer, int num_values, int null_count,
                               const uint8_t* valid_bits, int64_t valid_bits_offset) {
  assert(sizeof(T) == 4 || sizeof(T) == 8);

  // [(1 256i block/8 epi32), (2 256i/8 epi64)] for each batch
  constexpr size_t kBatchSize = 8;
  constexpr size_t kBatchByteSize = kBatchSize / 8;
  // Point to end as we add the spacing from the back.
  int idx_decode = num_values - null_count - 1;
  int idx_spaced = num_values - 1;
  int64_t idx_valid_bits = valid_bits_offset + idx_spaced;
  // Loop the valid bits with batch bytes align way
  const auto p = arrow::internal::BitmapWordAlign<kBatchByteSize>(
      valid_bits, valid_bits_offset, num_values);

  // The trailing bits
  auto trailing_bits = p.trailing_bits;
  while (trailing_bits-- > 0) {
    if (BitUtil::GetBit(valid_bits, idx_valid_bits)) {
      buffer[idx_spaced] = buffer[idx_decode];
      idx_decode--;
    } else {
      memset(static_cast<void*>(buffer + idx_spaced), 0, sizeof(T));
    }
    idx_spaced--;
    idx_valid_bits--;
  }

  // The aligned parts can fill into batches
  auto aligned_words = p.aligned_words;
  int64_t idx_valid_bytes = BitUtil::BytesForBits(idx_valid_bits) - 1;
  while ((idx_decode < idx_spaced) && (aligned_words-- > 0)) {
    const uint8_t valid_byte_value = valid_bits[idx_valid_bytes];
    idx_valid_bytes--;

    // Compiler able to pick the path at instantiation time
    if (sizeof(T) == 4) {
      // Path for epi32, 8 epi32 for one m256i block, one byte in valid_bits
      const __m256i* kMask256Table =
          reinterpret_cast<const __m256i*>(arrow::internal::kMask256Avx2ExpandEpi32);
      const uint8_t valid_count = BitUtil::kBytePopcount[valid_byte_value];
      idx_spaced -= kBatchSize;
      idx_valid_bits -= kBatchSize;
      idx_decode -= valid_count;

      const __m256i src =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer + idx_decode + 1));
      const __m256i mask = _mm256_load_si256(&kMask256Table[valid_byte_value]);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer + idx_spaced + 1),
                          _mm256_permutevar8x32_epi32(src, mask));
    } else {
      // Path for epi64, expand from high to low, 4 bits each time
      const __m256i* kMask256Table =
          reinterpret_cast<const __m256i*>(arrow::internal::kMask256Avx2ExpandEpi64);
      for (int i = 0; i < 2; i++) {
        const uint8_t valid_value = (valid_byte_value >> (4 * (2 - i - 1))) & 0x0F;
        const uint8_t valid_count = BitUtil::kBytePopcount[valid_value];
        idx_spaced -= 4;
        idx_valid_bits -= 4;
        idx_decode -= valid_count;

        const __m256i src =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer + idx_decode + 1));
        const __m256i mask = _mm256_load_si256(&kMask256Table[valid_value]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer + idx_spaced + 1),
                            _mm256_permutevar8x32_epi32(src, mask));
      }
    }
  }

  // The remaining leading bits
  auto leading_bits = p.leading_bits;
  while (leading_bits-- > 0) {
    if (BitUtil::GetBit(valid_bits, idx_valid_bits)) {
      buffer[idx_spaced] = buffer[idx_decode];
      idx_decode--;
    } else {
      memset(static_cast<void*>(buffer + idx_spaced), 0, sizeof(T));
    }
    idx_spaced--;
    idx_valid_bits--;
  }

  return num_values;
}

template <typename T>
inline int SpacedCompressAvx2(const T* src, int num_values, const uint8_t* valid_bits,
                              int64_t valid_bits_offset, T* output) {
  if (sizeof(T) == 4 || sizeof(T) == 8) {
    // Based on _mm256_permutevar8x32_epi32
    return SpacedCompressAvx2Epi32Epi64<T>(src, num_values, valid_bits, valid_bits_offset,
                                           output);
  } else if (sizeof(T) == 1) {
    // Fall back to SSE for epi8
    return SpacedCompressSseShuffle<T>(src, num_values, valid_bits, valid_bits_offset,
                                       output);
  } else {
    return SpacedCompressScalar<T>(src, num_values, valid_bits, valid_bits_offset,
                                   output);
  }
}

template <typename T>
inline int SpacedExpandAvx2(T* buffer, int num_values, int null_count,
                            const uint8_t* valid_bits, int64_t valid_bits_offset) {
  if (sizeof(T) == 4 || sizeof(T) == 8) {
    // Based on _mm256_permutevar8x32_epi32
    return SpacedExpandAvx2Epi32Epi64<T>(buffer, num_values, null_count, valid_bits,
                                         valid_bits_offset);
  } else if (sizeof(T) == 1) {
    // Fall back to SSE for epi8
    return SpacedExpandSseShuffle<T>(buffer, num_values, null_count, valid_bits,
                                     valid_bits_offset);
  } else {
    return SpacedExpandScalar<T>(buffer, num_values, null_count, valid_bits,
                                 valid_bits_offset);
  }
}
#endif

#if defined(ARROW_HAVE_AVX512)
template <typename T>
int SpacedCompressAvx512Epi32Epi64(const T* values, int num_values,
                                   const uint8_t* valid_bits, int64_t valid_bits_offset,
                                   T* output) {
  assert(sizeof(T) == 4 || sizeof(T) == 8);  // Only support epi32 and epi64
  // [(1 512i block/16 epi32), (1 512i block/8 epi64)] for each batch
  constexpr int kBatchSize = sizeof(T) == 8 ? 8 : 16;
  constexpr int kBatchByteSize = kBatchSize / 8;
  int num_valid_values = 0;
  // Loop the valid bits with batch bytes align way
  const auto p = arrow::internal::BitmapWordAlign<kBatchByteSize>(
      valid_bits, valid_bits_offset, num_values);
  int valid_values_batch;

  // First handle the leading bits
  const int leading_bits = static_cast<int>(p.leading_bits);
  if (leading_bits > 0) {
    valid_values_batch =
        SpacedCompressScalar(values, leading_bits, valid_bits, valid_bits_offset, output);
    // Step the count and idx
    output += valid_values_batch;
    num_valid_values += valid_values_batch;
    values += leading_bits;
  }

  // The aligned parts can fill into batches
  const uint8_t* aligned_bits = p.aligned_start;
  auto aligned_words = p.aligned_words;
  while (aligned_words-- > 0) {
    // Count the valid numbers of one batch.
    valid_values_batch = BitUtil::kBytePopcount[aligned_bits[0]];
    if (kBatchByteSize > 1) {
      valid_values_batch += BitUtil::kBytePopcount[aligned_bits[1]];
    }

    const __m512i src = _mm512_loadu_si512(values);
    __m512i result;
    if (sizeof(T) == 4) {
      // 16 epi32 for one m512i block, two bytes in valid_bits
      const __mmask16 k = *(reinterpret_cast<const __mmask16*>(aligned_bits));
      result = _mm512_maskz_compress_epi32(k, src);
    } else {
      // 8 epi64 for one m512i block, one byte in valid_bits
      const __mmask8 k = *(aligned_bits);
      result = _mm512_maskz_compress_epi64(k, src);
    }
    // Safe to store the spare null values which will be covered next batch
    _mm512_storeu_si512(output, result);

    // Step the count and idx
    num_valid_values += valid_values_batch;
    output += valid_values_batch;
    values += kBatchSize;
    aligned_bits += kBatchByteSize;
  }

  // The remaining trailing bits
  const int trailing_bits = static_cast<int>(p.trailing_bits);
  if (trailing_bits > 0) {
    valid_values_batch = SpacedCompressScalar(values, trailing_bits, valid_bits,
                                              p.trailing_bit_offset, output);
    // Step the count and idx
    output += valid_values_batch;
    num_valid_values += valid_values_batch;
    values += trailing_bits;
  }

  return num_valid_values;
}

template <typename T>
int SpacedExpandAvx512Epi32Epi64(T* buffer, int num_values, int null_count,
                                 const uint8_t* valid_bits, int64_t valid_bits_offset) {
  assert(sizeof(T) == 4 || sizeof(T) == 8);  // Only support epi32 and epi64
  // [(1 512i block/16 epi32), (1 512i block/8 epi64)] for each batch
  constexpr int kBatchSize = sizeof(T) == 8 ? 8 : 16;
  constexpr int kBatchByteSize = kBatchSize / 8;
  // Point to end as we add the spacing from the back.
  int idx_decode = num_values - null_count - 1;
  int idx_spaced = num_values - 1;
  int idx_valid_bits = valid_bits_offset + idx_spaced;
  // Loop the valid bits with batch bytes align way
  const auto p = arrow::internal::BitmapWordAlign<kBatchByteSize>(
      valid_bits, valid_bits_offset, num_values);

  // The trailing bits
  auto trailing_bits = p.trailing_bits;
  while (trailing_bits-- > 0) {
    if (BitUtil::GetBit(valid_bits, idx_valid_bits)) {
      buffer[idx_spaced] = buffer[idx_decode];
      idx_decode--;
    } else {
      memset(static_cast<void*>(buffer + idx_spaced), 0, sizeof(T));
    }
    idx_spaced--;
    idx_valid_bits--;
  }

  // The aligned parts can fill into batches
  auto aligned_words = p.aligned_words;
  int64_t idx_valid_bytes = BitUtil::BytesForBits(idx_valid_bits) - 1;
  while ((idx_decode < idx_spaced) && (aligned_words-- > 0)) {
    // Count valid numbers of one batch and step the index
    uint8_t valid_count = BitUtil::kBytePopcount[valid_bits[idx_valid_bytes]];
    idx_valid_bytes--;
    if (kBatchByteSize > 1) {
      valid_count += BitUtil::kBytePopcount[valid_bits[idx_valid_bytes]];
      idx_valid_bytes--;
    }
    idx_decode -= valid_count;
    idx_spaced -= kBatchSize;
    idx_valid_bits -= kBatchSize;

    const __m512i src = _mm512_loadu_si512(buffer + idx_decode + 1);
    __m512i result;
    if (sizeof(T) == 4) {
      // 16 epi32 for one m512i block, two bytes in valid_bits
      const __mmask16 k =
          *(reinterpret_cast<const __mmask16*>(valid_bits + idx_valid_bytes + 1));
      result = _mm512_maskz_expand_epi32(k, src);
    } else {
      // 8 epi32 for one m512i block, one byte in valid_bits
      const __mmask8 k = *(valid_bits + idx_valid_bytes + 1);
      result = _mm512_maskz_expand_epi64(k, src);
    }
    _mm512_storeu_si512(buffer + idx_spaced + 1, result);
  }

  // The remaining leading bits
  auto leading_bits = p.leading_bits;
  while (leading_bits-- > 0) {
    if (BitUtil::GetBit(valid_bits, idx_valid_bits)) {
      buffer[idx_spaced] = buffer[idx_decode];
      idx_decode--;
    } else {
      memset(static_cast<void*>(buffer + idx_spaced), 0, sizeof(T));
    }
    idx_spaced--;
    idx_valid_bits--;
  }

  return num_values;
}

template <typename T>
inline int SpacedCompressAvx512(const T* src, int num_values, const uint8_t* valid_bits,
                                int64_t valid_bits_offset, T* output) {
  if (sizeof(T) == 4 || sizeof(T) == 8) {
    // Based on _mask_compress_epi32/64
    return SpacedCompressAvx512Epi32Epi64<T>(src, num_values, valid_bits,
                                             valid_bits_offset, output);
  } else if (sizeof(T) == 1) {
    // Fall back to SSE for epi8
    return SpacedCompressSseShuffle<T>(src, num_values, valid_bits, valid_bits_offset,
                                       output);
  } else {
    return SpacedCompressScalar<T>(src, num_values, valid_bits, valid_bits_offset,
                                   output);
  }
}

template <typename T>
inline int SpacedExpandAvx512(T* buffer, int num_values, int null_count,
                              const uint8_t* valid_bits, int64_t valid_bits_offset) {
  if (sizeof(T) == 4 || sizeof(T) == 8) {
    // Based on _mask_expand_epi32/64
    return SpacedExpandAvx512Epi32Epi64<T>(buffer, num_values, null_count, valid_bits,
                                           valid_bits_offset);
  } else if (sizeof(T) == 1) {
    // Fall back to SSE for epi8
    return SpacedExpandSseShuffle<T>(buffer, num_values, null_count, valid_bits,
                                     valid_bits_offset);
  } else {
    return SpacedExpandScalar<T>(buffer, num_values, null_count, valid_bits,
                                 valid_bits_offset);
  }
}
#endif

/// \brief Compress the buffer to spaced using the valid bits map, skip the null entries.
///
/// \param[in] src the source buffer
/// \param[in] num_values the size of source buffer
/// \param[in] valid_bits bitmap data indicating position of valid slots
/// \param[in] valid_bits_offset offset into valid_bits
/// \param[out] output the output buffer spaced
/// \return The size of spaced buffer.
template <typename T>
inline int SpacedCompress(const T* src, int num_values, const uint8_t* valid_bits,
                          int64_t valid_bits_offset, T* output) {
#if defined(ARROW_HAVE_AVX512)
  return SpacedCompressAvx512<T>(src, num_values, valid_bits, valid_bits_offset, output);
#elif defined(ARROW_HAVE_AVX2)
  return SpacedCompressAvx2<T>(src, num_values, valid_bits, valid_bits_offset, output);
#elif defined(ARROW_HAVE_SSE4_2)
  return SpacedCompressSse<T>(src, num_values, valid_bits, valid_bits_offset, output);
#else
  return SpacedCompressScalar<T>(src, num_values, valid_bits, valid_bits_offset, output);
#endif
}

/// \brief Expand the spaced buffer as the valid bits map, leave spaces for null entries.
///
/// \param[in, out] buffer the in-place buffer
/// \param[in] num_values total size of both spaced and null slots
/// \param[in] null_count number of null slots
/// \param[in] valid_bits bitmap data indicating position of valid slots
/// \param[in] valid_bits_offset offset into valid_bits
/// \return The number of values expanded, including nulls.
template <typename T>
inline int SpacedExpand(T* buffer, int num_values, int null_count,
                        const uint8_t* valid_bits, int64_t valid_bits_offset) {
#if defined(ARROW_HAVE_AVX512)
  return SpacedExpandAvx512<T>(buffer, num_values, null_count, valid_bits,
                               valid_bits_offset);
#elif defined(ARROW_HAVE_AVX2)
  return SpacedExpandAvx2<T>(buffer, num_values, null_count, valid_bits,
                             valid_bits_offset);
#elif defined(ARROW_HAVE_SSE4_2)
  return SpacedExpandSse<T>(buffer, num_values, null_count, valid_bits,
                            valid_bits_offset);
#else
  return SpacedExpandScalar<T>(buffer, num_values, null_count, valid_bits,
                               valid_bits_offset);
#endif
}

}  // namespace internal
}  // namespace util
}  // namespace arrow
