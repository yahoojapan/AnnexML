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

#include <vector>
#include <utility>

#include "NGT.h"

namespace yj {
namespace xmlc {

class LLEmbedding {
  public:
    LLEmbedding();
    ~LLEmbedding();
    int Init(size_t num_data, size_t num_cluster, size_t embed_size, int verbose);
    void Learn(const std::vector<std::vector<std::pair<int, float> > > &data,
               const std::vector<std::vector<int> > &labels,
               const std::vector<std::vector<size_t> > &cluster_assign,
               size_t cluster, size_t num_nn, int label_normalize,
               float eta0, float gamma, size_t max_iter, int seed,
               std::vector<double> *prec_vec);
    int Search(const std::vector<std::pair<int, float> > &datum,
               size_t cluster, size_t nearestK, float eps,
               std::vector<std::pair<size_t, float> > *result) const;
    int InitSearchIndex();
    int BuildSearchIndex(size_t cluster, size_t max_in_leaf, size_t num_edge, int seed);
    int WriteToStream(FILE *stream) const;
    int ReadFromStream(FILE *stream);
    size_t Size() const;
    size_t GetMatrixSize() const;
    size_t GetEmbeddingSize() const;
    size_t GetSearchIndexSize() const;
    size_t num_cluster() const { return cluster_assign_.size(); };

  private:
    size_t embed_size_;
    size_t num_data_;
    int seed_;
    int verbose_;
    std::vector<std::vector<std::pair<int, std::vector<float> > > > w_mat_vec_;
    float *embeddings_;
    std::vector<std::vector<size_t> > cluster_assign_;
    std::vector<NGT> search_index_vec_;
};


} // namespace xmlc
} // namespace yj 
