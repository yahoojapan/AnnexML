//
// Copyright (C) 2017 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

namespace {

const size_t align_byte = 32;

inline size_t calc_aligned_float_size(size_t n) {
  size_t m = align_byte / sizeof(float);
  if (n % m == 0) { return n; }
  return m * ((n / m) + 1);
}

inline float dense_dot(const float *v1, const float *v2, size_t n) {
#ifdef USEFMA
  __m256 YMMacc = _mm256_setzero_ps();

  for (size_t i = 0; i < n; i += 8) {
    __m256 YMM1 = _mm256_load_ps(v1 + i);
    __m256 YMM2 = _mm256_load_ps(v2 + i);
    YMMacc = _mm256_fmadd_ps(YMM1, YMM2, YMMacc);
  }
  __attribute__((aligned(32))) float t[8] = {0};
  _mm256_store_ps(t, YMMacc);

  return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#else
  float ret = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    ret += v1[i] * v2[i];
  }
  return ret;
#endif
}

inline float edist(const float *v1, const float *v2, size_t n) {
#ifdef USEFMA
  __m256 YMMacc = _mm256_setzero_ps();

  for (size_t i = 0; i < n; i += 8) {
    __m256 YMM1 = _mm256_load_ps(v1 + i);
    __m256 YMM2 = _mm256_load_ps(v2 + i);
    YMM1 = _mm256_sub_ps(YMM1, YMM2);
    YMMacc = _mm256_fmadd_ps(YMM1, YMM1, YMMacc);
  }
  __attribute__((aligned(32))) float t[8] = {0};
  _mm256_store_ps(t, YMMacc);

  return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
#else
  float ret = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float diff = v1[i] - v2[i];
    ret += diff * diff;
  }
  return ret;
#endif
}

inline float calc_l2norm(const float *v, size_t n) {
  return std::sqrt(dense_dot(v, v, n));
}

} // namespace
