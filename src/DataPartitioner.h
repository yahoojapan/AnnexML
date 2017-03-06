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

#include <cstdio>
#include <vector>

namespace yj {
namespace xmlc {

class DataPartitioner {
  public:
    DataPartitioner();
    ~DataPartitioner();
    void Clear();
    float RunKmeans(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                   size_t K, size_t max_iter, int seed, int verbose);
    float RunPairwise(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                      const std::vector<std::vector<int> > &labels_vec,
                      size_t K, size_t max_iter,
                      size_t num_nn, int label_normalize,
                      float eta0, float lambda, float gamma,
                      int seed, int verbose);
    size_t GetNearestCluster(const std::vector<std::pair<int, float> > &datum) const;
    float GetNearestClusters(const std::vector<std::pair<int, float> > &datum,
                             std::vector<size_t> *centers) const;
    int NormalizeData(std::vector<std::vector<std::pair<int, float> > > *data_vec) const;
    void CopiedFrom(const DataPartitioner &that);
    int WriteToStream(FILE *stream) const;
    int ReadFromStream(FILE *stream);
    size_t K() const { return K_; }

  private:
    size_t K_;
    std::vector<std::vector<std::pair<int, float> > > w_index_;

};


} // namespace xmlc
} // namespace yj 
