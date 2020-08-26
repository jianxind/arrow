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

#include <stddef.h>
#include <stdint.h>

namespace arrow {
namespace util {
namespace internal {

void ByteStreamSplitEncodeAvx512Float(const uint8_t* raw_values, const size_t num_values,
                                      uint8_t* output_buffer_raw);
void ByteStreamSplitEncodeAvx512Double(const uint8_t* raw_values, const size_t num_values,
                                       uint8_t* output_buffer_raw);

void ByteStreamSplitDecodeAvx512Float(const uint8_t* data, int64_t num_values,
                                      int64_t stride, float* out);
void ByteStreamSplitDecodeAvx512Double(const uint8_t* data, int64_t num_values,
                                       int64_t stride, double* out);

template <typename T>
void inline ByteStreamSplitEncodeAvx512(const uint8_t* raw_values,
                                        const size_t num_values,
                                        uint8_t* output_buffer_raw) {
  static_assert("ByteStreamSplitEncodeAvx512 not implemented");
}

template <>
void inline ByteStreamSplitEncodeAvx512<float>(const uint8_t* raw_values,
                                               const size_t num_values,
                                               uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeAvx512Float(raw_values, num_values, output_buffer_raw);
}

template <>
void inline ByteStreamSplitEncodeAvx512<double>(const uint8_t* raw_values,
                                                const size_t num_values,
                                                uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeAvx512Double(raw_values, num_values, output_buffer_raw);
}

template <typename T>
void inline ByteStreamSplitDecodeAvx512(const uint8_t* data, int64_t num_values,
                                        int64_t stride, T* out) {
  static_assert("ByteStreamSplitDecodeAvx512 not implemented");
}

template <>
void inline ByteStreamSplitDecodeAvx512<float>(const uint8_t* data, int64_t num_values,
                                               int64_t stride, float* out) {
  return ByteStreamSplitDecodeAvx512Float(data, num_values, stride, out);
}

template <>
void inline ByteStreamSplitDecodeAvx512<double>(const uint8_t* data, int64_t num_values,
                                                int64_t stride, double* out) {
  return ByteStreamSplitDecodeAvx512Double(data, num_values, stride, out);
}

}  // namespace internal
}  // namespace util
}  // namespace arrow
