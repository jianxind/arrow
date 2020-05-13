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

#include <memory>
#include <type_traits>

#include "arrow/compute/kernel.h"
#include "arrow/compute/kernels/aggregate.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/logging.h"
#include "arrow/util/simd.h"

namespace arrow {

class Array;
class DataType;

namespace compute {

// Find the largest compatible primitive type for a primitive type.
template <typename I, typename Enable = void>
struct FindAccumulatorType {};

template <typename I>
struct FindAccumulatorType<I, enable_if_signed_integer<I>> {
  using Type = Int64Type;
};

template <typename I>
struct FindAccumulatorType<I, enable_if_unsigned_integer<I>> {
  using Type = UInt64Type;
};

template <typename I>
struct FindAccumulatorType<I, enable_if_floating_point<I>> {
  using Type = DoubleType;
};

template <typename ArrowType, typename StateType>
class SumAggregateFunction final : public AggregateFunctionStaticState<StateType> {
  using CType = typename TypeTraits<ArrowType>::CType;
  using ArrayType = typename TypeTraits<ArrowType>::ArrayType;

  // A small number of elements rounded to the next cacheline. This should
  // amount to a maximum of 4 cachelines when dealing with 8 bytes elements.
  static constexpr int64_t kTinyThreshold = 32;
  static_assert(kTinyThreshold >= (2 * CHAR_BIT) + 1,
                "ConsumeSparse requires 3 bytes of null bitmap, and 17 is the"
                "required minimum number of bits/elements to cover 3 bytes.");

 public:
  Status Consume(const Array& input, StateType* state) const override {
    const ArrayType& array = static_cast<const ArrayType&>(input);

    if (input.null_count() == 0) {
      *state = ConsumeDense(array);
    } else if (input.length() <= kTinyThreshold) {
      // In order to simplify ConsumeSparse implementation (requires at least 3
      // bytes of bitmap data), small arrays are handled differently.
      *state = ConsumeTiny(array);
    } else {
      *state = ConsumeSparse(array);
    }

    return Status::OK();
  }

  Status Merge(const StateType& src, StateType* dst) const override {
    *dst += src;
    return Status::OK();
  }

  Status Finalize(const StateType& src, Datum* output) const override {
    *output = src.Finalize();
    return Status::OK();
  }

  std::shared_ptr<DataType> out_type() const override { return StateType::out_type(); }

 private:
  template <typename T>
  inline int BatchSize() const {
#if defined(ARROW_HAVE_AVX512)
    return 8;  // 8 epi64/double(accumulator type) for one m512 block
#else
    return 8;  // Always 8 for scalar
#endif
  }

  template <typename T>
  // Scalar version for dense batch
  inline StateType SumDenseBatch(const T* values, int64_t num_batch) const {
    const auto kBatchSize = BatchSize<T>();
    StateType local;

    for (auto idx_batch = 0; idx_batch < num_batch; idx_batch++) {
      for (auto i = 0; i < kBatchSize; i++) {
        local.sum += values[kBatchSize * idx_batch + i];
      }
    }

    local.count = num_batch * kBatchSize;

    return local;
  }

  template <typename T>
  // Scalar version for sparse batch
  inline StateType SumSparseBatch(const uint8_t* bitmap, const T* values,
                                  int64_t num_batch) const {
    const auto kBatchSize = BatchSize<T>();
    const auto kBatchByteSize = kBatchSize / 8;
    StateType local;

    for (auto idx_batch = 0; idx_batch < num_batch; idx_batch++) {
      for (auto i = 0; i < kBatchByteSize; i++) {
        auto idx_bitmap = idx_batch * kBatchByteSize + i;
        local += UnrolledSum(bitmap[idx_bitmap], &values[idx_bitmap * 8]);
      }
    }

    return local;
  }

#if defined(ARROW_HAVE_AVX512)

// Dense helper for accumulator type is same to data type
#define BATCH_FUNC_DENSE_DIRECT(Type, SimdSetFn, SimdLoadFn, SimdAddFn)         \
  inline StateType SumDenseBatch(const Type* values, int64_t num_batch) const { \
    StateType local;                                                            \
    const auto kBatchSize = BatchSize<Type>();                                  \
                                                                                \
    auto result_simd = SimdSetFn(0);                                            \
    /* Directly aggregate the results with vectorize way */                     \
    for (auto i = 0; i < num_batch; i++) {                                      \
      auto src_simd = SimdLoadFn(&values[i * kBatchSize]);                      \
      result_simd = SimdAddFn(src_simd, result_simd);                           \
    }                                                                           \
                                                                                \
    /* Aggregate the final result on the vectorize results */                   \
    auto result_scalar = reinterpret_cast<Type*>(&result_simd);                 \
    for (auto i = 0; i < kBatchSize; i++) {                                     \
      local.sum += result_scalar[i];                                            \
    }                                                                           \
    local.count += kBatchSize * num_batch;                                      \
    return local;                                                               \
  }

  // AVX512 override version for dense double
  BATCH_FUNC_DENSE_DIRECT(double, _mm512_set1_pd, _mm512_loadu_pd, _mm512_add_pd)
  // AVX512 override version for dense int64_t
  BATCH_FUNC_DENSE_DIRECT(int64_t, _mm512_set1_epi64, _mm512_loadu_si512,
                          _mm512_add_epi64)
  // AVX512 override version for dense uint64_t
  BATCH_FUNC_DENSE_DIRECT(uint64_t, _mm512_set1_epi64, _mm512_loadu_si512,
                          _mm512_add_epi64)

// Dense helper for which need a converter from data type to accumulator type
#define BATCH_FUNC_DENSE_CVT(Type, SumType, SimdSetFn, SimdLoadFn, SimdLoadType, \
                             SimdCvtFn, SimdAddFn)                               \
  inline StateType SumDenseBatch(const Type* values, int64_t num_batch) const {  \
    StateType local;                                                             \
    const auto kBatchSize = BatchSize<Type>();                                   \
                                                                                 \
    auto result_simd = SimdSetFn(0);                                             \
    for (auto i = 0; i < num_batch; i++) {                                       \
      auto src_simd =                                                            \
          SimdLoadFn(reinterpret_cast<SimdLoadType>(&values[i * kBatchSize]));   \
      /* Convert to the accumulator type */                                      \
      auto acc_simd = SimdCvtFn(src_simd);                                       \
      /* Vectorize aggregate the result on the accumulator type*/                \
      result_simd = SimdAddFn(acc_simd, result_simd);                            \
    }                                                                            \
                                                                                 \
    /* Aggregate the final result on the vectorize results */                    \
    auto result_scalar = reinterpret_cast<SumType*>(&result_simd);               \
    for (auto i = 0; i < kBatchSize; i++) {                                      \
      local.sum += result_scalar[i];                                             \
    }                                                                            \
    local.count += kBatchSize * num_batch;                                       \
    return local;                                                                \
  }

  // AVX512 override version for dense float
  BATCH_FUNC_DENSE_CVT(float, double, _mm512_set1_pd, _mm256_loadu_ps, const float*,
                       _mm512_cvtps_pd, _mm512_add_pd)
  // AVX512 override version for dense int32_t
  BATCH_FUNC_DENSE_CVT(int32_t, int64_t, _mm512_set1_epi64, _mm256_loadu_si256,
                       const __m256i*, _mm512_cvtepi32_epi64, _mm512_add_epi64)
  // AVX512 override version for dense uint32_t
  BATCH_FUNC_DENSE_CVT(uint32_t, uint64_t, _mm512_set1_epi64, _mm256_loadu_si256,
                       const __m256i*, _mm512_cvtepi32_epi64, _mm512_add_epi64)
  // AVX512 override version for dense int16_t
  BATCH_FUNC_DENSE_CVT(int16_t, int64_t, _mm512_set1_epi64, _mm_loadu_si128,
                       const __m128i*, _mm512_cvtepi16_epi64, _mm512_add_epi64)
  // AVX512 override version for dense uint16_t
  BATCH_FUNC_DENSE_CVT(uint16_t, uint64_t, _mm512_set1_epi64, _mm_loadu_si128,
                       const __m128i*, _mm512_cvtepi16_epi64, _mm512_add_epi64)
  // AVX512 override version for dense int8_t
  BATCH_FUNC_DENSE_CVT(int8_t, int64_t, _mm512_set1_epi64, _mm_loadu_si128,
                       const __m128i*, _mm512_cvtepi8_epi64, _mm512_add_epi64)
  // AVX512 override version for dense uint8_t
  BATCH_FUNC_DENSE_CVT(uint8_t, uint64_t, _mm512_set1_epi64, _mm_loadu_si128,
                       const __m128i*, _mm512_cvtepi8_epi64, _mm512_add_epi64)

// Sparse helper for accumulator type is same to data type
#define BATCH_FUNC_SPARSE_DIRECT(Type, SimdSetFn, SimdLoadFn, SimdMaskAddFn)      \
  inline StateType SumSparseBatch(const uint8_t* bitmap, const Type* values,      \
                                  int64_t num_batch) const {                      \
    StateType local;                                                              \
    const auto kBatchSize = BatchSize<Type>();                                    \
                                                                                  \
    auto result_simd = SimdSetFn(0);                                              \
    /* Directly mask aggregate the results with vectorize way */                  \
    for (auto i = 0; i < num_batch; i++) {                                        \
      auto src_simd = SimdLoadFn(&values[i * kBatchSize]);                        \
      result_simd = SimdMaskAddFn(result_simd, bitmap[i], src_simd, result_simd); \
      local.count += BitUtil::kBytePopcount[bitmap[i]];                           \
    }                                                                             \
                                                                                  \
    /* Aggregate the final result on the vectorize results */                     \
    auto result_scalar = reinterpret_cast<Type*>(&result_simd);                   \
    for (auto i = 0; i < kBatchSize; i++) {                                       \
      local.sum += result_scalar[i];                                              \
    }                                                                             \
    return local;                                                                 \
  }

  // AVX512 override version for sparse double
  BATCH_FUNC_SPARSE_DIRECT(double, _mm512_set1_pd, _mm512_loadu_pd, _mm512_mask_add_pd)
  // AVX512 override version for sparse int64_t
  BATCH_FUNC_SPARSE_DIRECT(int64_t, _mm512_set1_epi64, _mm512_loadu_si512,
                           _mm512_mask_add_epi64)
  // AVX512 override version for sparse uint64_t
  BATCH_FUNC_SPARSE_DIRECT(uint64_t, _mm512_set1_epi64, _mm512_loadu_si512,
                           _mm512_mask_add_epi64)

// Sparse helper for which need a converter from data type to accumulator type
#define BATCH_FUNC_SPARSE_CVT(Type, SumType, SimdSetFn, SimdLoadFn, SimdLoadType, \
                              SimdCvtFn, SimdMaskAddFn)                           \
  inline StateType SumSparseBatch(const uint8_t* bitmap, const Type* values,      \
                                  int64_t num_batch) const {                      \
    StateType local;                                                              \
    const auto kBatchSize = BatchSize<Type>();                                    \
                                                                                  \
    auto result_simd = SimdSetFn(0);                                              \
    for (auto i = 0; i < num_batch; i++) {                                        \
      auto src_simd =                                                             \
          SimdLoadFn(reinterpret_cast<SimdLoadType>(&values[i * kBatchSize]));    \
      /* Convert to the accumulator type */                                       \
      auto acc_simd = SimdCvtFn(src_simd);                                        \
      /* Mask vectorize aggregate the result on the accumulator type*/            \
      result_simd = SimdMaskAddFn(result_simd, bitmap[i], acc_simd, result_simd); \
      local.count += BitUtil::kBytePopcount[bitmap[i]];                           \
    }                                                                             \
                                                                                  \
    /* Aggregate the final result on the vectorize results */                     \
    auto result_scalar = reinterpret_cast<SumType*>(&result_simd);                \
    for (auto i = 0; i < kBatchSize; i++) {                                       \
      local.sum += result_scalar[i];                                              \
    }                                                                             \
    return local;                                                                 \
  }

  // AVX512 override version for sparse float
  BATCH_FUNC_SPARSE_CVT(float, double, _mm512_set1_pd, _mm256_loadu_ps, const float*,
                        _mm512_cvtps_pd, _mm512_mask_add_pd)
  // AVX512 override version for sparse int32_t
  BATCH_FUNC_SPARSE_CVT(int32_t, int64_t, _mm512_set1_epi64, _mm256_loadu_si256,
                        const __m256i*, _mm512_cvtepi32_epi64, _mm512_mask_add_epi64)
  // AVX512 override version for sparse uint32_t
  BATCH_FUNC_SPARSE_CVT(uint32_t, uint64_t, _mm512_set1_epi64, _mm256_loadu_si256,
                        const __m256i*, _mm512_cvtepi32_epi64, _mm512_mask_add_epi64)
  // AVX512 override version for sparse int16_t
  BATCH_FUNC_SPARSE_CVT(int16_t, int64_t, _mm512_set1_epi64, _mm_loadu_si128,
                        const __m128i*, _mm512_cvtepi16_epi64, _mm512_mask_add_epi64)
  // AVX512 override version for sparse uint16_t
  BATCH_FUNC_SPARSE_CVT(uint16_t, uint64_t, _mm512_set1_epi64, _mm_loadu_si128,
                        const __m128i*, _mm512_cvtepi16_epi64, _mm512_mask_add_epi64)
  // AVX512 override version for sparse int8_t
  BATCH_FUNC_SPARSE_CVT(int8_t, int64_t, _mm512_set1_epi64, _mm_loadu_si128,
                        const __m128i*, _mm512_cvtepi8_epi64, _mm512_mask_add_epi64)
  // AVX512 override version for sparse uint8_t
  BATCH_FUNC_SPARSE_CVT(uint8_t, uint64_t, _mm512_set1_epi64, _mm_loadu_si128,
                        const __m128i*, _mm512_cvtepi8_epi64, _mm512_mask_add_epi64)
#endif

  // Handle the sparse as byte sequence
  inline StateType SumSparseBytes(const uint8_t* bitmap, const CType* values,
                                  int64_t num_byte) const {
    const auto kBatchSize = BatchSize<CType>();
    const auto kBatchByteSize = kBatchSize / 8;
    const auto num_batch = num_byte / kBatchByteSize;
    const auto num_suffix_byte = num_byte % kBatchByteSize;
    StateType local;

    local += SumSparseBatch(bitmap, values, num_batch);

    auto start_idx_suffix = num_batch * kBatchByteSize;
    for (auto i = 0; i < num_suffix_byte; i++) {
      local +=
          UnrolledSum(bitmap[start_idx_suffix + i], &values[(start_idx_suffix + i) * 8]);
    }

    return local;
  }

  StateType ConsumeDense(const ArrayType& array) const {
    const auto kBatchSize = BatchSize<CType>();
    const auto values = array.raw_values();
    const int64_t length = array.length();
    StateType local;
    const auto num_batch = length / kBatchSize;
    const auto num_suffix = length % kBatchSize;

    local += SumDenseBatch(values, num_batch);

    for (auto i = 0; i < num_suffix; i++) {
      local.sum += values[kBatchSize * num_batch + i];
    }
    local.count += num_suffix;

    return local;
  }

  StateType ConsumeTiny(const ArrayType& array) const {
    StateType local;

    internal::BitmapReader reader(array.null_bitmap_data(), array.offset(),
                                  array.length());
    const auto values = array.raw_values();
    for (int64_t i = 0; i < array.length(); i++) {
      if (reader.IsSet()) {
        local.sum += values[i];
        local.count++;
      }
      reader.Next();
    }

    return local;
  }

  // While this is not branchless, gcc needs this to be in a different function
  // for it to generate cmov which ends to be slightly faster than
  // multiplication but safe for handling NaN with doubles.
  inline CType MaskedValue(bool valid, CType value) const { return valid ? value : 0; }

  inline StateType UnrolledSum(uint8_t bits, const CType* values) const {
    StateType local;

    if (bits < 0xFF) {
      // Some nulls
      for (size_t i = 0; i < 8; i++) {
        local.sum += MaskedValue(bits & (1U << i), values[i]);
      }
      local.count += BitUtil::kBytePopcount[bits];
    } else {
      // No nulls
      for (size_t i = 0; i < 8; i++) {
        local.sum += values[i];
      }
      local.count += 8;
    }

    return local;
  }

  StateType ConsumeSparse(const ArrayType& array) const {
    StateType local;

    // Sliced bitmaps on non-byte positions induce problem with the branchless
    // unrolled technique. Thus extra padding is added on both left and right
    // side of the slice such that both ends are byte-aligned. The first and
    // last bitmap are properly masked to ignore extra values induced by
    // padding.
    //
    // The execution is divided in 3 sections.
    //
    // 1. Compute the sum of the first masked byte.
    // 2. Compute the sum of the middle bytes
    // 3. Compute the sum of the last masked byte.

    const int64_t length = array.length();
    const int64_t offset = array.offset();

    // The number of bytes covering the range, this includes partial bytes.
    // This number bounded by `<= (length / 8) + 2`, e.g. a possible extra byte
    // on the left, and on the right.
    const int64_t covering_bytes = BitUtil::CoveringBytes(offset, length);
    DCHECK_GE(covering_bytes, 3);

    // Align values to the first batch of 8 elements. Note that raw_values() is
    // already adjusted with the offset, thus we rewind a little to align to
    // the closest 8-batch offset.
    const auto values = array.raw_values() - (offset % 8);

    // Align bitmap at the first consumable byte.
    const auto bitmap = array.null_bitmap_data() + BitUtil::RoundDown(offset, 8) / 8;

    // Consume the first (potentially partial) byte.
    const uint8_t first_mask = BitUtil::kTrailingBitmask[offset % 8];
    local += UnrolledSum(bitmap[0] & first_mask, values);

    // Consume the (full) middle bytes. The loop iterates in unit of
    // batches of 8 values and 1 byte of bitmap.
    local += SumSparseBytes(&bitmap[1], &values[8], covering_bytes - 2);

    // Consume the last (potentially partial) byte.
    const int64_t last_idx = covering_bytes - 1;
    const uint8_t last_mask = BitUtil::kPrecedingWrappingBitmask[(offset + length) % 8];
    local += UnrolledSum(bitmap[last_idx] & last_mask, &values[last_idx * 8]);

    return local;
  }
};  // namespace compute

}  // namespace compute
}  // namespace arrow
