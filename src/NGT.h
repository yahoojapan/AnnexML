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

#include <cfloat>
#include <cstdio>
#include <vector>
#include <random>

namespace yj {
namespace xmlc {

class NGT {
  public:
    NGT();
    ~NGT();

    void BuildIndex(const float *data_vec,
                    size_t vec_size,
                    const std::vector<size_t> &indices,
                    size_t max_in_leaf, size_t num_edge, int seed);
    int Search(const float *data_vec, const float *query,
               size_t K, float eta,
               std::vector<std::pair<size_t, float> > *result) const;
    void Clear();
    size_t Size() const;


  private:
    struct Node {
      int lid;
      int rid;
      float radius;
      std::vector<size_t> indices;
      float *center;
      Node() : center(NULL) {};
      ~Node() { if (center != NULL) { free(center); center = NULL; } };
    };

    int GrowTree(const float *data_vec,
                 const std::vector<size_t> &indices,
                 size_t max_in_leaf,
                 std::mt19937 *rnd_gen);
    int ProcessLeaf(const std::vector<size_t> &indices);
    int SplitNode(const float *data_vec,
                  const std::vector<size_t> &indices,
                  std::mt19937 *rnd_gen,
                  std::vector<size_t> *left_indices,
                  std::vector<size_t> *right_indices);
    int MakeBall(const float *data_vec,
                 const std::vector<size_t> &indices,
                 float *center, float *radius);
    int BuildGraph(const float *data_vec,
                   const std::vector<size_t> &indices,
                   size_t K);
    int LineSearch(const float *data_vec, const float *query,
                   size_t K, const std::vector<size_t> &indices,
                   std::vector<std::pair<size_t, float> > *heap) const;
    int TreeSearch(const float *data_vec, const float *query,
                   size_t K, size_t node_id,
                   std::vector<std::pair<size_t, float> > *heap) const;
    int GraphSearch(const float *data_vec, const float *query,
                    size_t K, float eps,
                    std::vector<std::pair<size_t, float> > *heap) const;

    size_t vec_size_;
    std::vector<size_t> indices_;
    std::vector<std::vector<size_t> > nn_vec_;
    std::vector<Node> node_vec_;

};


} // namespace xmlc
} // namespace yj 
