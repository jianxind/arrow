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

void ByteStreamSplitEncodeSse42Float(const uint8_t* raw_values, const size_t num_values,
                                     uint8_t* output_buffer_raw);
void ByteStreamSplitEncodeSse42Double(const uint8_t* raw_values, const size_t num_values,
                                      uint8_t* output_buffer_raw);

void ByteStreamSplitDecodeSse42Float(const uint8_t* data, int64_t num_values,
                                     int64_t stride, float* out);
void ByteStreamSplitDecodeSse42Double(const uint8_t* data, int64_t num_values,
                                      int64_t stride, double* out);

template <typename T>
void inline ByteStreamSplitEncodeSse42(const uint8_t* raw_values, const size_t num_values,
                                       uint8_t* output_buffer_raw) {
  static_assert(true, "ByteStreamSplitEncodeSse42 not implemented");
}

template <>
void inline ByteStreamSplitEncodeSse42<float>(const uint8_t* raw_values,
                                              const size_t num_values,
                                              uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeSse42Float(raw_values, num_values, output_buffer_raw);
}

template <>
void inline ByteStreamSplitEncodeSse42<double>(const uint8_t* raw_values,
                                               const size_t num_values,
                                               uint8_t* output_buffer_raw) {
  return ByteStreamSplitEncodeSse42Double(raw_values, num_values, output_buffer_raw);
}

template <typename T>
void inline ByteStreamSplitDecodeSse42(const uint8_t* data, int64_t num_values,
                                       int64_t stride, T* out) {
  static_assert(true, "ByteStreamSplitDecodeSse42 not implemented");
}

template <>
void inline ByteStreamSplitDecodeSse42<float>(const uint8_t* data, int64_t num_values,
                                              int64_t stride, float* out) {
  return ByteStreamSplitDecodeSse42Float(data, num_values, stride, out);
}

template <>
void inline ByteStreamSplitDecodeSse42<double>(const uint8_t* data, int64_t num_values,
                                               int64_t stride, double* out) {
  return ByteStreamSplitDecodeSse42Double(data, num_values, stride, out);
}

}  // namespace internal
}  // namespace util
}  // namespace arrow
