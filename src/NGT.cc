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

#include "NGT.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
#include <random>
#include <unordered_map>

#include "Utils.h"
#include "vectorize.h"

namespace {

union Entry {
  int flag;
  float val;
  Entry() : flag(-1) {};
  explicit Entry(float v) : val(v) {};
};

auto comp = [](const std::pair<size_t,float> &a,const std::pair<size_t,float> &b){return a.second<b.second;};

} // namespace

namespace yj {
namespace xmlc {

NGT::NGT() {
}

NGT::~NGT() {
}

void NGT::BuildIndex(const float *data_vec,
                     size_t vec_size,
                     const std::vector<size_t> &indices,
                     size_t max_in_leaf, size_t num_edge, int seed) {
  Clear();

  vec_size_ = vec_size;

  std::mt19937 rnd_gen(seed);
  GrowTree(data_vec, indices, max_in_leaf, &rnd_gen);

  indices_.assign(indices.begin(), indices.end());
  std::unordered_map<size_t, size_t> inv_map;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    inv_map[idx] = i;
  }

  for (size_t i = 0; i < node_vec_.size(); ++i) {
    Node &node = node_vec_[i];
    int status = posix_memalign((void**)&(node.center), align_byte, vec_size_ * sizeof(float));
    assert(status == 0);
    std::fill_n(node.center, vec_size_, 0.0f);
    MakeBall(data_vec, node.indices, node.center, &(node.radius));

    for (size_t j = 0; j < node.indices.size(); ++j) {
      size_t idx = node.indices[j];
      node.indices[j] = inv_map[idx];
    }
    if (node.lid > 0) { node.indices.clear(); }
  }

  BuildGraph(data_vec, indices, num_edge);
}

int NGT::Search(const float *data_vec, const float *query,
                size_t K, float eps,
                std::vector<std::pair<size_t, float> > *result) const {

  result->clear(); result->reserve(K);

  TreeSearch(data_vec, query, K, 0, result);
  GraphSearch(data_vec, query, K, eps, result);
  std::sort_heap(result->begin(), result->end(), comp);

  // convert Euclidean distance to cosine
  for (size_t i = 0; i < result->size(); ++i) {
    size_t orig_idx = indices_[(*result)[i].first];
    (*result)[i].first = orig_idx;
    (*result)[i].second = 1.0f - 0.5 * (*result)[i].second;
  }

  return 1;
}

void NGT::Clear() {
  indices_.clear();
  nn_vec_.clear();
  node_vec_.clear(); 
}


int NGT::GrowTree(const float *data_vec,
                  const std::vector<size_t> &indices,
                  size_t max_in_leaf,
                  std::mt19937 *rnd_gen) {

  if (indices.size() <= max_in_leaf) { return ProcessLeaf(indices); }

  std::vector<size_t> left_indices, right_indices;
  SplitNode(data_vec, indices, rnd_gen, &left_indices, &right_indices);
  if (left_indices.size() == 0 || right_indices.size() == 0) { return ProcessLeaf(indices); }

  int node_id = node_vec_.size();
  node_vec_.resize(node_id + 1);

  int lid = GrowTree(data_vec, left_indices, max_in_leaf, rnd_gen);
  int rid = GrowTree(data_vec, right_indices, max_in_leaf, rnd_gen);

  Node &node = node_vec_[node_id];
  node.lid = lid;
  node.rid = rid;
  node.indices.assign(indices.begin(), indices.end());

  return node_id;
}

int NGT::ProcessLeaf(const std::vector<size_t> &indices) {
  int node_id = node_vec_.size();
  node_vec_.resize(node_id + 1);

  Node &node = node_vec_[node_id];
  node.lid = -1;
  node.rid = -1;
  node.indices.assign(indices.begin(), indices.end());

  return node_id;
}

int NGT::SplitNode(const float *data_vec,
                   const std::vector<size_t> &indices,
                   std::mt19937 *rnd_gen,
                   std::vector<size_t> *left_indices,
                   std::vector<size_t> *right_indices) {

  std::uniform_int_distribution<size_t> idist(0, indices.size()-1);
  size_t base_idx = indices[idist(*rnd_gen)];
  const float *base_v = data_vec + vec_size_ * base_idx;

  float max_dist = 0.0f;
  size_t a_idx = base_idx;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    if (idx == base_idx) { continue; }
    const float *v = data_vec + vec_size_ * idx;

    float dist = edist(base_v, v, vec_size_);
    if (dist > max_dist) { a_idx = idx; max_dist = dist; }
  }

  max_dist = 0.0f;
  size_t b_idx = a_idx;
  const float *a_v = data_vec + vec_size_ * a_idx;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    if (idx == base_idx || idx == a_idx) { continue; }
    const float *v = data_vec + vec_size_ * idx;

    float dist = edist(a_v, v, vec_size_);
    if (dist > max_dist) { b_idx = idx; max_dist = dist; }
  }

  const float *b_v = data_vec + vec_size_ * b_idx;
  std::vector<float> w(vec_size_);
  for (size_t i = 0; i < vec_size_; ++i) { w[i] = b_v[i] - a_v[i]; }

  float a_snorm = 0.0f, b_snorm = 0.0f;
  for (size_t i = 0; i < w.size(); ++i) {
    a_snorm += a_v[i] * a_v[i];
    b_snorm += b_v[i] * b_v[i];
  }
  float bias = -0.5 * (b_snorm - a_snorm);

  left_indices->clear(); right_indices->clear();
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    const float *v = data_vec + vec_size_ * idx;

    float z = bias;
    for (size_t j = 0; j < vec_size_; ++j) { z += w[j] * v[j]; }

    if (z <= 0.0f) { left_indices->push_back(idx); }
    else           { right_indices->push_back(idx); }
  }

  return 1;
}

int NGT::MakeBall(const float *data_vec,
                  const std::vector<size_t> &indices,
                  float *center, float *radius) {
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    const float *v = data_vec + vec_size_ * idx;
    for (size_t j = 0; j < vec_size_; ++j) { center[j] += v[j]; }
  }
  for (size_t j = 0; j < vec_size_; ++j) { center[j] /= indices.size(); }

  *radius = 0.0f;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    const float *v = data_vec + vec_size_ * idx;

    float dist = edist(center, v, vec_size_);
    if (dist > *radius) { *radius = dist; }
  }
  *radius = std::sqrt(*radius);

  return 1;
}

int NGT::BuildGraph(const float *data_vec,
                    const std::vector<size_t> &indices,
                    size_t num_edge) {
  nn_vec_.clear();
  nn_vec_.resize(indices.size());
  std::vector<std::vector<std::pair<size_t, float> > > heap_vec(indices.size());

  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx1 = indices[i];
    const float *v1 = data_vec + vec_size_ * idx1;
    auto &heap_i = heap_vec[i];

    for (size_t j = i+1; j < indices.size(); ++j) {
      size_t idx2 = indices[j];
      const float *v2 = data_vec + vec_size_ * idx2;
      float dist = edist(v1, v2, vec_size_);

      if (heap_i.size() < num_edge) {
        heap_i.push_back(std::make_pair(j, dist));
        std::push_heap(heap_i.begin(), heap_i.end(), comp);
      } else if (dist < heap_i.front().second) {
        std::pop_heap(heap_i.begin(), heap_i.end(), comp);
        heap_i.pop_back();
        heap_i.push_back(std::make_pair(j, dist));
        std::push_heap(heap_i.begin(), heap_i.end(), comp);
      }

      auto &heap_j = heap_vec[j];
      if (heap_j.size() < num_edge) {
        heap_j.push_back(std::make_pair(i, dist));
        std::push_heap(heap_j.begin(), heap_j.end(), comp);
      } else if (dist < heap_j.front().second) {
        std::pop_heap(heap_j.begin(), heap_j.end(), comp);
        heap_j.pop_back();
        heap_j.push_back(std::make_pair(i, dist));
        std::push_heap(heap_j.begin(), heap_j.end(), comp);
      }
    }

    for (size_t j = 0; j < heap_i.size(); ++j) {
      nn_vec_[i].push_back(heap_i[j].first);
    }
    std::vector<std::pair<size_t, float> >().swap(heap_i);
  }

  return 1;
}

int NGT::LineSearch(const float *data_vec, const float *query,
                    size_t K, const std::vector<size_t> &indices,
                    std::vector<std::pair<size_t, float> > *heap) const {

  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];
    size_t orig_idx = indices_[idx];
    const float *v = data_vec + vec_size_ * orig_idx;

    float dist = edist(query, v, vec_size_);

    if (heap->size() >= K) {
      if (dist >= heap->front().second) { continue; }
      std::pop_heap(heap->begin(), heap->end(), comp);
      heap->pop_back();
    }
    heap->push_back(std::make_pair(idx, dist));
    std::push_heap(heap->begin(), heap->end(), comp);
  }

  return 1;
}

int NGT::TreeSearch(const float *data_vec, const float *query,
                    size_t K, size_t node_id,
                    std::vector<std::pair<size_t, float> > *heap) const {

  const Node &node = node_vec_[node_id];
  int lid = node.lid;
  int rid = node.rid;

  // leaf node
  if (lid < 0) { LineSearch(data_vec, query, K, node.indices, heap); return 1; }

  const auto &lc = node_vec_[lid].center;
  float ldist = edist(query, lc, vec_size_);

  const auto &rc = node_vec_[rid].center;
  float rdist = edist(query, rc, vec_size_);

  if (ldist < rdist) { TreeSearch(data_vec, query, K, lid, heap); }
  else               { TreeSearch(data_vec, query, K, rid, heap); }

  return 1;
}


int NGT::GraphSearch(const float *data_vec, const float *query,
                     size_t K, float eps,
                     std::vector<std::pair<size_t, float> > *heap) const {

  std::vector<Entry> done_vec(indices_.size());
  std::vector<size_t> candidate;

  for (size_t i = 0; i < heap->size(); ++i) {
    size_t idx = (*heap)[i].first;
    done_vec[idx].val = (*heap)[i].second;

    for (size_t j = 0; j < nn_vec_[idx].size(); ++j) {
      size_t nn_idx  = nn_vec_[idx][j];
      candidate.push_back(nn_idx);
    }
  }

  while (candidate.size() > 0) {
    size_t idx = candidate.back();
    candidate.pop_back();
    if (done_vec[idx].flag != -1) { continue; }

    size_t orig_idx = indices_[idx];
    const float *v = data_vec + vec_size_ * orig_idx;
    float dist = edist(query, v, vec_size_);
    done_vec[idx].val = dist;

    if (heap->size() >= K) {
      if (dist < heap->front().second) {
        std::pop_heap(heap->begin(), heap->end(), comp);
        heap->pop_back();
        heap->push_back(std::make_pair(idx, dist));
        std::push_heap(heap->begin(), heap->end(), comp);
      }
    } else {
      heap->push_back(std::make_pair(idx, dist));
      std::push_heap(heap->begin(), heap->end(), comp);
    }

    float r = (1.0f + eps) * heap->front().second;
    if (dist < r) {
      for (size_t j = 0; j < nn_vec_[idx].size(); ++j) {
        size_t nn_idx  = nn_vec_[idx][j];
        if (done_vec[nn_idx].flag == -1) { candidate.push_back(nn_idx); }
      }
    }
  }

  return 1;
}

size_t NGT::Size() const {
  size_t index_size = 0;
  index_size += sizeof(vec_size_);
  index_size += sizeof(size_t) * indices_.size();
  for (size_t i = 0; i < nn_vec_.size(); ++i) {
    index_size += sizeof(size_t) * nn_vec_[i].size();
  }
  index_size += sizeof(Node) * node_vec_.size();

  return index_size;
}


} // namespace xmlc
} // namespace yj 

