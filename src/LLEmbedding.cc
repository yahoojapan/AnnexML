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

#include "LLEmbedding.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>

#include "Utils.h"
#include "Reindexer.h"
#include "vectorize.h"

namespace {

inline void sparse_dot(const std::vector<std::pair<int, float> > &vec,
                       const std::vector<std::vector<float> > &mat,
                       std::vector<float> *z) {
  std::fill_n(z->begin(), z->size(), 0.0);
  for (size_t f = 0; f < vec.size(); ++f) {
    int k   = vec[f].first;
    float v = vec[f].second;
    for (size_t i = 0; i < mat[k].size(); ++i) { (*z)[i] += v * mat[k][i]; }
  }
}

inline float dense_dot(const std::vector<float> &v1, const std::vector<float> &v2) {
  float z = 0.0;
  for (size_t i = 0; i < v1.size(); ++i) { z += v1[i] * v2[i]; }
  return z;
}

inline float calc_l2norm(const std::vector<float> &v) {
  return std::sqrt(dense_dot(v, v));
}


inline void sparse_dot(const std::vector<std::pair<int, float> > &vec,
                       const std::vector<std::pair<int, std::vector<float> > > &mat,
                       size_t n, float *z) {
  std::fill_n(z, n, 0.0f);

  auto comparator = [](const std::pair<int, std::vector<float> > &p, int v) -> bool { return p.first < v; };

  auto vitr = vec.begin();
  auto mitr = mat.begin();
  while (vitr != vec.end()) {
    mitr = std::lower_bound(mitr, mat.end(), vitr->first, comparator);
    if (mitr == mat.end())  { break; }
    if (mitr->first == vitr->first) {
      const auto &row = mitr->second;
      for (size_t i = 0; i < row.size(); ++i) { z[i] += vitr->second * row[i]; }
    }
    ++vitr;
  }
}


inline void add_mat_adagrad(const std::vector<std::pair<int, float> > &x1,
                            const std::vector<float> &h1,
                            const std::vector<float> &h2,
                            float ip, float norm1, float norm2,
                            float eta0, float coef,
                            std::vector<std::vector<float> > *w_mat,
                            std::vector<std::vector<float> > *g_mat) {

  float bc = 1.0f / norm1 / norm2;
  float ab2 = ip / norm1 / norm1;

  for (size_t f = 0; f < x1.size(); ++f) {
    int k   = x1[f].first;
    float c = coef * bc * x1[f].second;
    for (size_t i = 0; i < h1.size(); ++i) {
      float g = c * (h2[i] - ab2 * h1[i]);
      (*g_mat)[k][i] += g * g;
      float eta = eta0 / std::sqrt((*g_mat)[k][i]);
      (*w_mat)[k][i] += eta * g;
    }
  }
}

void update_w_mat_dssm(
    const std::vector<std::vector<std::pair<int, float> > > &data_vec,
    size_t base_idx,
    const std::vector<std::pair<size_t, float> > &pos_vec,
    const std::vector<std::pair<size_t, float> > &neg_vec,
    size_t embed_size, float eta0, float gamma,
    std::vector<std::vector<float> > *w_mat,
    std::vector<std::vector<float> > *g_mat) {

  size_t num_pos = pos_vec.size();
  size_t num_neg = pos_vec.size();
  size_t num_samples = num_pos + num_neg;
  if (num_pos == 0) { return; }

  std::vector<float> bz(embed_size);
  sparse_dot(data_vec[base_idx], *w_mat, &bz);
  float bnorm = calc_l2norm(bz);

  std::vector<std::vector<float> > zs(num_samples);
  std::vector<float> zzs(num_samples);
  std::vector<float> norms(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    size_t idx = (i < num_pos) ? pos_vec[i].first : neg_vec[i-num_pos].first;
    zs[i].resize(embed_size);
    sparse_dot(data_vec[idx], *w_mat, &(zs[i]));
    zzs[i] = dense_dot(bz, zs[i]);
    norms[i] = calc_l2norm(zs[i]);
  }

  std::vector<double> as(num_samples);
  std::vector<double> ts(num_samples);
  for (size_t i = 0; i < num_pos; ++i) {
    double cosi = zzs[i] / norms[i] / bnorm;

    double denom = 1.0;
    for (size_t j = num_pos; j < num_samples; ++j) {
      double cosj = zzs[j] / norms[j] / bnorm;

      double tmp = std::exp(-gamma * (cosi - cosj));
      ts[j] = tmp;
      denom += tmp;
    }

    for (size_t j = num_pos; j < num_samples; ++j) {
      double tmp = gamma * ts[j] / denom;
      as[i] += tmp;
      as[j] -= tmp;
    }
  }

  for (size_t i = 0; i < num_samples; ++i) {
    size_t idx = (i < num_pos) ? pos_vec[i].first : neg_vec[i-num_pos].first;
    add_mat_adagrad(data_vec[base_idx], bz, zs[i], zzs[i], bnorm, norms[i], eta0, as[i], w_mat, g_mat);
    add_mat_adagrad(data_vec[idx], zs[i], bz, zzs[i], norms[i], bnorm, eta0, as[i], w_mat, g_mat);
  }
}

void calc_precision(
    const std::vector<std::vector<std::pair<int, float> > > &data_vec,
    const std::vector<std::vector<int> > &labels_vec,
    size_t embed_size, size_t num_nn,
    const std::vector<std::vector<float> > &w_mat,
    std::vector<double> *prec_vec) {

  prec_vec->clear();
  prec_vec->resize(5);

  auto comp1 = [](const std::pair<size_t,float> &a, const std::pair<size_t,float> &b) {return a.second>b.second;};
  auto comp2 = [](const std::pair<int,int> &a, const std::pair<int,int> &b) {
    return (a.second != b.second) ? a.second > b.second : a.first < b.first;
  };

  std::vector<std::vector<float> > zs(data_vec.size());
  std::vector<float> norms(data_vec.size());
  for (size_t i = 0; i < data_vec.size(); ++i) {
    zs[i].resize(embed_size);
    sparse_dot(data_vec[i], w_mat, &(zs[i]));
    norms[i] = calc_l2norm(zs[i]);
  }

  std::vector<std::pair<size_t, float> > heap;
  heap.reserve(num_nn);

  for (size_t i = 0; i < data_vec.size(); ++i) {
    heap.clear();
    for (size_t j = 0; j < data_vec.size(); ++j) {
      if (i == j) { continue; }
      float v = dense_dot(zs[i], zs[j]) / norms[j];

      if (heap.size() >= num_nn) {
        if (v <= heap.front().second) { continue; }
        std::pop_heap(heap.begin(), heap.end(), comp1);
        heap.pop_back();
      }
      heap.push_back(std::make_pair(j, v));
      std::push_heap(heap.begin(), heap.end(), comp1);
    }


    std::unordered_map<int, int> p_map;
    for (size_t k = 0; k < heap.size(); ++k) {
      size_t idx = heap[k].first;
      for (size_t j = 0; j < labels_vec[idx].size(); ++j) {
        int l = labels_vec[idx][j];
        if (p_map.count(l) > 0) { p_map[l] += 1; }
        else                    { p_map[l]  = 1; }
      }
    }
    std::vector<std::pair<int, int> > p_vec(p_map.begin(), p_map.end());
    std::sort(p_vec.begin(), p_vec.end(), comp2);

    size_t max_k = std::min(p_vec.size(), prec_vec->size());

    for (size_t k = 0; k < max_k; ++k) {
      int l = p_vec[k].first;
      auto litr = std::lower_bound(labels_vec[i].begin(), labels_vec[i].end(), l);
      if (litr != labels_vec[i].end() && *litr == l) { (*prec_vec)[k] += 1.0; }
    }
  }

  double sum_acc = 0.0;
  for (size_t k = 0; k < prec_vec->size(); ++k) {
    sum_acc += (*prec_vec)[k];
    (*prec_vec)[k] = sum_acc / data_vec.size() / (k+1);
  }
}

void calc_achievable_precision(
    const std::vector<std::vector<int> > &labels_vec,
    const std::vector<std::vector<std::pair<size_t, float> > > &pos_vec,
    std::vector<double> *prec_vec) {

  auto comp = [](const std::pair<int,int> &a, const std::pair<int,int> &b) {
    return (a.second != b.second) ? a.second > b.second : a.first < b.first;
  };

  prec_vec->clear();
  prec_vec->resize(5);

  for (size_t i = 0; i < labels_vec.size(); ++i) {

    std::unordered_map<int, int> p_map;
    for (size_t k = 0; k < pos_vec[i].size(); ++k) {
      size_t idx = pos_vec[i][k].first;
      for (size_t j = 0; j < labels_vec[idx].size(); ++j) {
        int l = labels_vec[idx][j];
        if (p_map.count(l) > 0) { p_map[l] += 1; }
        else                    { p_map[l]  = 1; }
      }
    }
    std::vector<std::pair<int, int> > p_vec(p_map.begin(), p_map.end());
    std::sort(p_vec.begin(), p_vec.end(), comp);

    size_t max_k = std::min(p_vec.size(), prec_vec->size());

    // binary search, assuming labels_vec is sorted
    for (size_t k = 0; k < max_k; ++k) {
      int l = p_vec[k].first;
      auto litr = std::lower_bound(labels_vec[i].begin(), labels_vec[i].end(), l);
      if (litr != labels_vec[i].end() && *litr == l) { (*prec_vec)[k] += 1.0; }
    }
  }

  double sum_acc = 0.0;
  for (size_t k = 0; k < prec_vec->size(); ++k) {
    sum_acc += (*prec_vec)[k];
    (*prec_vec)[k] = sum_acc / labels_vec.size() / (k+1);
  }
}


void get_positives(
    size_t num_pos,
    const std::vector<std::vector<int> > &labels_vec,
    size_t max_lid, int label_normalize,
    std::vector<std::vector<std::pair<size_t, float> > > *pos_vec) {

  auto comp = [](const std::pair<size_t,float> &a, const std::pair<size_t,float> &b) {return a.second>b.second;};

  // build inverted index of labels
  std::vector<std::vector<std::pair<size_t, float> > > l_inv_idx(max_lid + 1);
  for (size_t i = 0; i < labels_vec.size(); ++i) {
    float v = (label_normalize > 0) ? 1.0f / labels_vec[i].size() : 1.0f;
    for (size_t l = 0; l < labels_vec[i].size(); ++l) {
      l_inv_idx[labels_vec[i][l]].push_back(std::make_pair(i, v));
    }
  }

  pos_vec->resize(labels_vec.size());

  std::vector<std::pair<size_t, float> > score_vec;
  std::unordered_map<size_t, float> score_map;

  for (size_t i = 0; i < labels_vec.size(); ++i) {
    score_map.clear();

    for (size_t l = 0; l < labels_vec[i].size(); ++l) {
      size_t lid = labels_vec[i][l];
      size_t list_size = l_inv_idx[lid].size();
      for (size_t j = 0; j < list_size; ++j) {
        size_t idx = l_inv_idx[lid][j].first;
        if (i == idx) { continue; }
        float v = l_inv_idx[lid][j].second;

        auto itr = score_map.find(idx);
        if (itr != score_map.end()) { itr->second += v; }
        else                        { score_map[idx] = v; }
      }
    }

    score_vec.clear();
    for (auto itr = score_map.begin(); itr != score_map.end(); ++itr) {
      if (score_vec.size() >= num_pos) {
        if (itr->second <= score_vec.front().second) { continue; }
        std::pop_heap(score_vec.begin(), score_vec.end(), comp);
        score_vec.pop_back();
      }
      score_vec.push_back(*itr);
      std::push_heap(score_vec.begin(), score_vec.end(), comp);
    }
    std::sort_heap(score_vec.begin(), score_vec.end(), comp);

    (*pos_vec)[i].clear();
    if (score_vec.size() > 0) { // if not sufficient, amplify positives
      for (size_t j = 0; j < num_pos; ++j) {
        (*pos_vec)[i].push_back(score_vec[j % score_vec.size()]);
      }
    }
  }

}


void sampling_negatives_uniform(
    size_t num_neg,
    size_t num_data,
    std::mt19937 *rnd_gen,
    std::vector<std::pair<size_t, float> > *neg_vec) {

  std::uniform_int_distribution<size_t> idist(0, num_data-1);

  neg_vec->clear();
  for (size_t i = 0; i < num_neg; ++i) {
    size_t idx = idist(*rnd_gen);
    neg_vec->push_back(std::make_pair(idx, 0.0f));
  }
}


} // namespace


namespace yj {
namespace xmlc {

LLEmbedding::LLEmbedding()
: embed_size_(10), num_data_(), seed_(), verbose_(0),
  w_mat_vec_(), embeddings_(NULL), cluster_assign_()
{}

LLEmbedding::~LLEmbedding() {
  if (embeddings_ != NULL) {
    free(embeddings_); embeddings_ = NULL;
  }
}

int LLEmbedding::Init(size_t num_data, size_t num_cluster, size_t embed_size, int verbose) {
  embed_size_ = embed_size;
  num_data_ = num_data;
  verbose_ = verbose;

  size_t aligned_embed_size = calc_aligned_float_size(embed_size_);
  int status = posix_memalign((void**)&embeddings_, align_byte, num_data_ * aligned_embed_size * sizeof(float));
  std::fill_n(embeddings_, num_data * aligned_embed_size, 0.0f);
  assert(status == 0);

  w_mat_vec_.resize(num_cluster); 

  cluster_assign_.resize(num_cluster);

  return 1;
}


int LLEmbedding::Search(
    const std::vector<std::pair<int, float> > &datum,
    size_t cluster, size_t num_nn, float eps,
    std::vector<std::pair<size_t, float> > *result
    ) const {
  if (result == NULL) { return -1; }
  result->clear();

  auto comp = [](const std::pair<size_t,float> &a, const std::pair<size_t,float> &b) {return a.second>b.second;};

  size_t aligned_embed_size = calc_aligned_float_size(embed_size_);

  float *z;
  int status = posix_memalign((void**)&z, align_byte, aligned_embed_size * sizeof(float));
  std::fill_n(z, aligned_embed_size, 0.0f);
  assert(status == 0);

  sparse_dot(datum, w_mat_vec_[cluster], aligned_embed_size, z);

  if (search_index_vec_.size() > cluster) {
    float norm = calc_l2norm(z, aligned_embed_size);
    if (norm != 0.0f) { for (size_t i = 0; i < embed_size_; ++i) { z[i] /= norm; } }
    search_index_vec_[cluster].Search(embeddings_, z, num_nn, eps, result);
    return 2;
  }

  auto &heap = *result;
  const auto &index = cluster_assign_[cluster];

  for (size_t i = 0; i < index.size(); ++i) {
    size_t idx = index[i];
#ifndef SLEEC
    float v = dense_dot(z, embeddings_ + idx * aligned_embed_size, aligned_embed_size);
#else
    // for SLEEC prediction, use negative Euclidean distance
    float v = -edist(z, embeddings_ + idx * aligned_embed_size, aligned_embed_size);
#endif

    if (heap.size() >= num_nn) {
      if (v <= heap.front().second) { continue; }
      std::pop_heap(heap.begin(), heap.end(), comp);
      heap.pop_back();
    }
    heap.push_back(std::make_pair(idx, v));
    std::push_heap(heap.begin(), heap.end(), comp);
  }
  std::sort_heap(heap.begin(), heap.end(), comp);

  free(z); z = NULL;

  return 1;
}


int LLEmbedding::InitSearchIndex() {
  search_index_vec_.clear();
  search_index_vec_.resize(cluster_assign_.size());
  return 1;
}

int LLEmbedding::BuildSearchIndex(size_t cluster, size_t max_in_leaf, size_t num_edge, int seed) {
  if (num_data_ == 0 || cluster_assign_.size() == 0) { return 0; }

  size_t aligned_embed_size = calc_aligned_float_size(embed_size_);
  search_index_vec_[cluster].BuildIndex(embeddings_, aligned_embed_size, cluster_assign_[cluster], max_in_leaf, num_edge, seed);

  return 1;
}


int LLEmbedding::WriteToStream(FILE *stream) const {
  Utils::WriteNumToStream(embed_size_, stream);
  Utils::WriteNumToStream(seed_, stream);
  Utils::WriteNumToStream(verbose_, stream);

  Utils::WriteNumToStream(w_mat_vec_.size(), stream);
  for (size_t i = 0; i < w_mat_vec_.size(); ++i) {
    Utils::WriteNumToStream(w_mat_vec_[i].size(), stream);
    for (size_t j = 0; j < w_mat_vec_[i].size(); ++j) {
      Utils::WriteNumToStream(w_mat_vec_[i][j].first, stream);
      Utils::WriteNumToStream(w_mat_vec_[i][j].second.size(), stream);
      for (size_t k = 0; k < w_mat_vec_[i][j].second.size(); ++k) {
        Utils::WriteNumToStream(w_mat_vec_[i][j].second[k], stream);
      }
    }
  }

  size_t aligned_embed_size = calc_aligned_float_size(embed_size_);
  Utils::WriteNumToStream(num_data_, stream);
  for (size_t i = 0; i < num_data_; ++i) {
    Utils::WriteNumToStream(embed_size_, stream);
    float *evec = embeddings_ + aligned_embed_size * i;
    for (size_t j = 0; j < embed_size_; ++j) {
      Utils::WriteNumToStream(evec[j], stream);
    }
  }

  Utils::WriteNumToStream(cluster_assign_.size(), stream);
  for (size_t i = 0; i < cluster_assign_.size(); ++i) {
    Utils::WriteNumToStream(cluster_assign_[i].size(), stream);
    for (size_t j = 0; j < cluster_assign_[i].size(); ++j) {
      Utils::WriteNumToStream(cluster_assign_[i][j], stream);
    }
  }

  return 1;
};

int LLEmbedding::ReadFromStream(FILE *stream) {
  Utils::ReadNumFromStream(stream, &embed_size_);
  Utils::ReadNumFromStream(stream, &seed_);
  Utils::ReadNumFromStream(stream, &verbose_);

  size_t vec_size = 0;

  Utils::ReadNumFromStream(stream, &vec_size);
  w_mat_vec_.resize(vec_size);
  for (size_t i = 0; i < w_mat_vec_.size(); ++i) {
    Utils::ReadNumFromStream(stream, &vec_size);
    w_mat_vec_[i].resize(vec_size);
    for (size_t j = 0; j < w_mat_vec_[i].size(); ++j) {
      Utils::ReadNumFromStream(stream, &(w_mat_vec_[i][j].first));
      Utils::ReadNumFromStream(stream, &vec_size);
      w_mat_vec_[i][j].second.resize(vec_size);
      for (size_t k = 0; k < w_mat_vec_[i][j].second.size(); ++k) {
        Utils::ReadNumFromStream(stream, &(w_mat_vec_[i][j].second[k]));
      }
    }
  }

  if (embeddings_ != NULL) { free(embeddings_); embeddings_ = NULL; }

  size_t aligned_embed_size = calc_aligned_float_size(embed_size_);
  Utils::ReadNumFromStream(stream, &num_data_);
  int status = posix_memalign((void**)&embeddings_, align_byte, num_data_ * aligned_embed_size * sizeof(float));
  std::fill_n(embeddings_, num_data_ * aligned_embed_size, 0.0f);
  assert(status == 0);
  for (size_t i = 0; i < num_data_; ++i) {
    Utils::ReadNumFromStream(stream, &vec_size);
    float *evec = embeddings_ + aligned_embed_size * i;
    for (size_t j = 0; j < embed_size_; ++j) {
      Utils::ReadNumFromStream(stream, &(evec[j]));
    }
    for (size_t j = embed_size_; j < aligned_embed_size; ++j) { evec[j] = 0.0f; }
  }

  Utils::ReadNumFromStream(stream, &vec_size);
  cluster_assign_.resize(vec_size);
  for (size_t i = 0; i < cluster_assign_.size(); ++i) {
    Utils::ReadNumFromStream(stream, &vec_size);
    cluster_assign_[i].resize(vec_size);
    for (size_t j = 0; j < cluster_assign_[i].size(); ++j) {
      Utils::ReadNumFromStream(stream, &(cluster_assign_[i][j]));
    }
  }


  return 1;
};


void LLEmbedding::Learn(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                        const std::vector<std::vector<int> > &labels_vec,
                        const std::vector<std::vector<size_t> > &cluster_assign,
                        size_t cluster, size_t num_nn, int label_normalize,
                        float eta0, float gamma, size_t max_iter, int seed,
                        std::vector<double> *prec_vec) {
  if (cluster_assign[cluster].size() == 0) { return; }

  std::mt19937 rnd_gen(seed);

  Reindexer d_idxer;
  Reindexer l_idxer;
  std::vector<std::vector<std::pair<int, float> > > r_data_vec;
  std::vector<std::vector<int> > r_labels_vec;
  int max_fid = d_idxer.IndexData(data_vec, cluster_assign[cluster], &r_data_vec);
  int max_lid = l_idxer.IndexLabels(labels_vec, cluster_assign[cluster], &r_labels_vec);
  for (size_t i = 0; i < r_labels_vec.size(); ++i) {
    std::sort(r_labels_vec[i].begin(), r_labels_vec[i].end()); // sort for binary search
  }


  float avg_num_features = 0.0f, avg_num_labels = 0.0f;
  for (size_t i = 0; i < r_data_vec.size(); ++i) {
    avg_num_features += r_data_vec[i].size(); avg_num_labels += r_labels_vec[i].size();
  }
  avg_num_features /= r_data_vec.size(); avg_num_labels /= r_labels_vec.size();

  if (verbose_ > 0) {
    fprintf(stderr, "cluster: %3lu, #data: %lu, max_fid: %d, max_lid: %d, avg_num_features: %.2f, avg_num_label: %.2f\n", cluster, r_data_vec.size(), max_fid, max_lid, avg_num_features, avg_num_labels);
  }

  // init w_mat and g_mat
  float norm_ratio = std::sqrt(1.0f / avg_num_features);
  std::uniform_real_distribution<float> fdist(-norm_ratio, norm_ratio);
  std::vector<std::vector<float> > w_mat(max_fid + 1);
  std::vector<std::vector<float> > g_mat(max_fid + 1);
  for (size_t i = 0; i < w_mat.size(); ++i) {
    w_mat[i].resize(embed_size_);
    g_mat[i].resize(embed_size_);
    for (size_t j = 0; j < w_mat[i].size(); ++j) { w_mat[i][j] = fdist(rnd_gen); }
    std::fill_n(g_mat[i].begin(), g_mat[i].size(), 1.0f); // set initial value
  }

  std::vector<std::vector<std::pair<size_t, float> > > pos_vec(r_data_vec.size());
  get_positives(num_nn, r_labels_vec, max_lid, label_normalize, &pos_vec);

  if (verbose_ > 0) {
    calc_achievable_precision(r_labels_vec, pos_vec, prec_vec);
    fprintf(stderr, "Achievable:   ");
    for (size_t k = 0; k < prec_vec->size(); ++k) { fprintf(stderr, "P@%lu: %.4f, ", k+1, (*prec_vec)[k]); }
    fprintf(stderr, "\n");
  }

  std::vector<size_t> indices(r_data_vec.size());
  for (size_t i = 0; i < r_data_vec.size(); ++i) { indices[i] = i; }
  std::vector<std::pair<size_t, float> > neg_vec(num_nn);

  for (size_t iter = 0; iter < max_iter; ++iter) {
    if (verbose_ > 0) { fprintf(stderr, "\rIter: %4lu", iter); }

    std::shuffle(indices.begin(), indices.end(), rnd_gen);
    for (size_t i = 0; i < indices.size(); ++i) {
      size_t idx = indices[i];
      sampling_negatives_uniform(num_nn, r_data_vec.size(), &rnd_gen, &neg_vec);

      update_w_mat_dssm(r_data_vec, idx, pos_vec[idx], neg_vec, embed_size_, eta0, gamma, &w_mat, &g_mat);
    }
  }
  calc_precision(r_data_vec, r_labels_vec, embed_size_, num_nn, w_mat, prec_vec);

  if (verbose_ > 0) {
    fprintf(stderr, "\r");

    fprintf(stderr, "cluster: %3lu, ", cluster);
    for (size_t k = 0; k < prec_vec->size(); ++k) { fprintf(stderr, "P@%lu: %.4f, ", k+1, (*prec_vec)[k]); }
    fprintf(stderr, "\n");
  }

  w_mat_vec_[cluster].resize(w_mat.size());
  for (size_t i = 0; i < w_mat.size(); ++i) {
    int old_i = d_idxer.N2O(i);
    w_mat_vec_[cluster][i].first = old_i;
    w_mat_vec_[cluster][i].second.assign(w_mat[i].begin(), w_mat[i].end());
  }

  size_t aligned_embed_size = calc_aligned_float_size(embed_size_);
  for (size_t i = 0; i < cluster_assign[cluster].size(); ++i) {
    size_t idx = cluster_assign[cluster][i];
    float *evec = embeddings_ + aligned_embed_size * idx;
    sparse_dot(data_vec[idx], w_mat_vec_[cluster], aligned_embed_size, evec);

    float norm = calc_l2norm(evec, aligned_embed_size);
    for (size_t j = 0; j < embed_size_; ++j) { evec[j] /= norm; }
    for (size_t j = embed_size_; j < aligned_embed_size; ++j) { evec[j] = 0.0f; }
  }

  cluster_assign_[cluster].assign(cluster_assign[cluster].begin(), cluster_assign[cluster].end());
}

size_t LLEmbedding::Size() const {
  size_t total_size = 0;

  total_size += sizeof(embed_size_);
  total_size += sizeof(num_data_);
  total_size += sizeof(seed_);
  total_size += sizeof(verbose_);

  total_size += GetMatrixSize();
  total_size += GetEmbeddingSize();

  total_size += sizeof(cluster_assign_.size());
  for (size_t i = 0; i < cluster_assign_.size(); ++i) {
    total_size += sizeof(cluster_assign_[i].size());
    for (size_t j = 0; j < cluster_assign_[i].size(); ++j) {
      total_size += sizeof(cluster_assign_[i][j]);
    }
  }

  total_size += GetSearchIndexSize();

  return total_size;
}

size_t LLEmbedding::GetMatrixSize() const {
  size_t mat_size = 0;

  mat_size += sizeof(w_mat_vec_.size());
  for (size_t i = 0; i < w_mat_vec_.size(); ++i) {
    mat_size += sizeof(w_mat_vec_[i].size());
    for (size_t j = 0; j < w_mat_vec_[i].size(); ++j) {
      mat_size += sizeof(int);
      mat_size += sizeof(w_mat_vec_[i][j].second.size());
      mat_size += sizeof(float) * w_mat_vec_[i][j].second.size();
    }
  }

  return mat_size;
}

size_t LLEmbedding::GetEmbeddingSize() const {
  size_t aligned_embed_size = calc_aligned_float_size(embed_size_);
  return sizeof(*embeddings_) * num_data_ * aligned_embed_size;
}

size_t LLEmbedding::GetSearchIndexSize() const {
  size_t index_size = 0;
  for (size_t i = 0; i < search_index_vec_.size(); ++i) {
    index_size += search_index_vec_[i].Size();
  }
  return index_size;
}

} // namespace xmlc
} // namespace yj 

