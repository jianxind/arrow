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

#include <cstdint>
#include "arrow/util/visibility.h"

namespace arrow {
namespace internal {

#if defined(ARROW_HAVE_AVX2)
ARROW_EXPORT extern const uint32_t kMask256Avx2CompressEpi32[];
ARROW_EXPORT extern const uint32_t kMask256Avx2CompressEpi64[];

ARROW_EXPORT extern const uint32_t kMask256Avx2ExpandEpi32[];
ARROW_EXPORT extern const uint32_t kMask256Avx2ExpandEpi64[];
#endif  // ARROW_HAVE_AVX2

}  // namespace internal
}  // namespace arrow
