#!/bin/python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Generate the mask of avx2 lookup table for spaced compress and expand
# Modified from original source:
# https://github.com/lemire/simdprune/blob/master/scripts/avx32.py
# The original copyright notice follows.

# This code is released under the
# Apache License Version 2.0 http://www.apache.org/licenses/.
# (c) Daniel Lemire

# Usage: python3 spaced_avx2_table_codegen.py > spaced_avx2_table_generated.cc
# We use below API to achieve the expand/compress function, and a generated
# lookup table to avoid the slow loop over the valit bits mask.
#
#  __m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx)
#  Description
#    Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
#  Operation
#    FOR j := 0 to 7
#      i := j*32
#      id := idx[i+2:i]*32
#      dst[i+31:i] := a[id+31:id]
#    ENDFOR


NullMask = 0x00  # Just set to zero position


def print_mask_expand_table(dt_dword_width, item_dword_width):
    """
    Generate the lookup table for AVX2 expand shuffle control mask.

    Parameters
    ----------
    dt_dword_width : int
        Dword width of the data type
    item_dword_width:
        Dword width of each shuffle control item in the table
    """
    loop = int(item_dword_width / dt_dword_width)
    for index in range(1 << loop):
        maps = []
        lastbit = 0x00

        for bit in range(loop):
            if (index & (1 << bit)):
                for n in range(dt_dword_width):
                    maps.append(dt_dword_width * lastbit + n)
                lastbit += 1
            else:
                for n in range(dt_dword_width):
                    maps.append(NullMask)

        out = ""
        for item in range(item_dword_width):
            out += " 0x" + format(maps[item], '02x') + ","
        print(" " + out)


def print_mask_expand_epi32():
    """Generate the mask table for sse epi32 spaced expand"""
    print("extern const uint32_t kMask256Avx2ExpandEpi32[] = {")
    print_mask_expand_table(1, 8)
    print("};")


def print_mask_expand_epi64():
    """Generate the mask table for sse epi64 spaced expand"""
    print("extern const uint32_t kMask256Avx2ExpandEpi64[] = {")
    print_mask_expand_table(2, 8)
    print("};")


def print_mask_compress_table(dt_dword_width, item_dword_width):
    """
    Generate the lookup table for AVX2 compress shuffle control mask.

    Parameters
    ----------
    dt_dword_width : int
        Dword width of the data type
    item_dword_width:
        Dword width of each shuffle control item in the table
    """
    loop = int(item_dword_width / dt_dword_width)
    for index in range(1 << loop):
        maps = []

        for bit in range(loop):
            if (index & (1 << bit)):
                for n in range(dt_dword_width):
                    maps.append(dt_dword_width * bit + n)
        while(len(maps) < item_dword_width):
            maps.append(NullMask)

        out = ""
        for item in range(item_dword_width):
            out += " 0x" + format(maps[item], '02x') + ","
        print(" " + out)


def print_mask_compress_epi32():
    """Generate the mask table for avx2 epi32 spaced compress"""
    print("extern const uint32_t kMask256Avx2CompressEpi32[] = {")
    print_mask_compress_table(1, 8)
    print("};")


def print_mask_compress_epi64():
    """Generate the mask table for avx2 epi64 spaced compress"""
    print("extern const uint32_t kMask256Avx2CompressEpi64[] = {")
    print_mask_compress_table(2, 8)
    print("};")


def print_copyright():
    print(
        """// Licensed to the Apache Software Foundation (ASF) under one
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
// under the License.""")


def print_note():
    print("//")
    print("// Automatically generated file; DO NOT EDIT.")


if __name__ == '__main__':
    print_copyright()
    print_note()
    print("")
    print("#include <cstdint>")
    print('#include "arrow/util/spaced_avx2.h"')
    print("")
    print("namespace arrow {")
    print("namespace internal {")
    print("")
    print("#if defined(ARROW_HAVE_AVX2)")
    print_mask_compress_epi32()
    print("")
    print_mask_compress_epi64()
    print("")
    print_mask_expand_epi32()
    print("")
    print_mask_expand_epi64()
    print("")
    print("#endif // ARROW_HAVE_AVX2")
    print("")
    print("}  // namespace internal")
    print("}  // namespace arrow")
