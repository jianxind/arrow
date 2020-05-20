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
//
// Automatically generated file; DO NOT EDIT.

#include <cstdint>
#include "arrow/util/spaced_avx2.h"

namespace arrow {
namespace internal {

#if defined(ARROW_HAVE_AVX2)
extern const uint32_t kMask256Avx2CompressEpi32[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x05, 0x00, 0x00, 0x00,
  0x04, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x05, 0x00, 0x00, 0x00,
  0x03, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x06, 0x00, 0x00, 0x00,
  0x04, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x06, 0x00, 0x00, 0x00,
  0x03, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x06, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x06, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x00, 0x00,
  0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x03, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x00, 0x00,
  0x04, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x02, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x00, 0x00,
  0x03, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x07, 0x00, 0x00, 0x00,
  0x04, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x07, 0x00, 0x00, 0x00,
  0x03, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x07, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x07, 0x00, 0x00,
  0x05, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x03, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x05, 0x07, 0x00, 0x00,
  0x04, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x02, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x05, 0x07, 0x00, 0x00,
  0x03, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x05, 0x07, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x05, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x05, 0x07, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x00,
  0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x03, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x00, 0x00,
  0x04, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x02, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x06, 0x07, 0x00, 0x00,
  0x03, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x06, 0x07, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x06, 0x07, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x00,
  0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x02, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x02, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x03, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x03, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x02, 0x03, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x00,
  0x04, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x01, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x02, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x02, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x00,
  0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00,
  0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00,
  0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
};

extern const uint32_t kMask256Avx2CompressEpi64[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00,
  0x04, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x05, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00,
  0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x03, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x00, 0x00,
  0x04, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
};

extern const uint32_t kMask256Avx2ExpandEpi32[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x04, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x04, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x05, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x04, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x04, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x05, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x04, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x00,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x00,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x05, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x03,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x03,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x03,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x03,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x03,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x00, 0x04,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x03,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x04,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x04,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x03,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x03,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x00, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x04,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x04,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x00, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x00, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x04,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x04,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x00, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x04,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x04,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x00, 0x05,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x04,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x05,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x05,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x06,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x04,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x04,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x04,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x04,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x04,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x00, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x04,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x05,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x05,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x00, 0x05, 0x06,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x04,
  0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x04,
  0x00, 0x01, 0x02, 0x00, 0x00, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x04,
  0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x04,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x05,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x04, 0x05, 0x06,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x06,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x01, 0x00, 0x02, 0x03, 0x04, 0x05, 0x06,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
  0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
};

extern const uint32_t kMask256Avx2ExpandEpi64[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x00, 0x00,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
  0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x03,
  0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
  0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
};

#endif // ARROW_HAVE_AVX2

}  // namespace internal
}  // namespace arrow