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

#include "AnnexML.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_map>

#include <omp.h>

#include "Utils.h"
#include "FileReader.h"
#include "DataConverter.h"


namespace yj {
namespace xmlc {

AnnexML::AnnexML()
  : param_(), 
    partitioning_vec_(), embedding_vec_(), labels_()
{}

AnnexML::~AnnexML() {
}

int AnnexML::Init(const AnnexMLParameter &param, bool load_model) {
  param_ = param;
  if (CheckParam() < 0) { return -1; }

  if (load_model) {
    FILE *fp = fopen(param_.model_file().c_str(), "rb");
    if (fp == NULL) { return -1; }

    AnnexMLParameter prm;
    prm.ReadFromStream(fp);
    MergeParam(prm);

    size_t num_vec = 0;
    size_t start_pos = 0;

    start_pos = ftell(fp);
    Utils::ReadNumFromStream(fp, &num_vec);
    labels_.resize(num_vec);
    for (size_t i = 0; i < labels_.size(); ++i) {
      Utils::ReadNumFromStream(fp, &num_vec);
      labels_[i].resize(num_vec);
      for (size_t j = 0; j < labels_[i].size(); ++j) {
        Utils::ReadNumFromStream(fp, &(labels_[i][j]));
      }
    }
    size_t labels_size = ftell(fp) - start_pos;

    Utils::ReadNumFromStream(fp, &num_vec);

    size_t num_used_learner = param_.num_learner();
    if (num_used_learner > num_vec) { num_used_learner = num_vec; }

    size_t partitioning_vec_size = 0, embedding_vec_size = 0, w_size = 0, e_size = 0;
    partitioning_vec_.resize(num_used_learner);
    embedding_vec_.resize(num_used_learner);
    for (size_t i = 0; i < num_used_learner; ++i) {
      start_pos = ftell(fp);
      partitioning_vec_[i].ReadFromStream(fp);
      partitioning_vec_size += ftell(fp) - start_pos;

      embedding_vec_[i].ReadFromStream(fp);
      embedding_vec_size += embedding_vec_[i].Size();

      w_size += embedding_vec_[i].GetMatrixSize();
      e_size += embedding_vec_[i].GetEmbeddingSize();
    }

    size_t total_size = sizeof(param_) + partitioning_vec_size + embedding_vec_size + labels_size;

    fclose(fp);
    float mb = 1000 * 1000;
    fprintf(stderr, "Loaded model from %s\n", param_.model_file().c_str());
    fprintf(stderr, "Total: %.2f MB => labels: %.2f MB, partitioning: %.2f MB, embedding: %.2f MB (matrix: %.2f MB, embed: %.2f MB) \n", total_size / mb, labels_size / mb, partitioning_vec_size / mb, embedding_vec_size / mb, w_size / mb, e_size / mb);

    if (param_.pred_type() > 0) {
      size_t num_thread = (param_.num_thread() > 0) ? param_.num_thread() : 1;
      omp_set_num_threads(num_thread);

      std::mt19937 rnd_gen(param_.seed());
      std::uniform_int_distribution<int> idist(0, INT_MAX);

      std::vector<std::pair<size_t, size_t> > task_vec;
      std::vector<int> seed_vec;
      for (size_t i = 0; i < embedding_vec_.size(); ++i) {
        embedding_vec_[i].InitSearchIndex();
        size_t num_cluster = embedding_vec_[i].num_cluster();
        for (size_t cluster = 0; cluster < num_cluster; ++cluster) {
          int seed = idist(rnd_gen);
          task_vec.push_back(std::make_pair(i, cluster));
          seed_vec.push_back(seed);
        }
      }

      fprintf(stderr, "num_edge: %d, search_eps: %f\n", param_.num_edge(), param_.search_eps());
      fprintf(stderr, "Build SearchIndex...");
      size_t max_in_leaf = param_.num_nn(); // tentative...
      double start = Utils::GetTimeOfDaySec();
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < task_vec.size(); ++i) {
        size_t lid = task_vec[i].first;
        size_t cid = task_vec[i].second;
        embedding_vec_[lid].BuildSearchIndex(cid, max_in_leaf, param_.num_edge(), seed_vec[i]);
      }
      double elapsed = Utils::GetTimeOfDaySec() - start;
      size_t index_size = 0;
      for (size_t i = 0; i < embedding_vec_.size(); ++i) {
        index_size += embedding_vec_[i].GetSearchIndexSize();
      }
      float mb = 1000 * 1000;

      fprintf(stderr, "\rBuild SearchIndex done! (%.2f sec, %.2f MB)\n", elapsed, index_size / mb);
    }

  } else {

    if (! param_.has_train_file()) { fprintf(stderr, "train_file is not specified\n"); return -1; }
  }

  return 1;
}


int AnnexML::Train() {

  size_t num_thread = (param_.num_thread() > 0) ? param_.num_thread() : 1;
  int verbose = (num_thread > 1) ? 0 : param_.verbose();
  omp_set_num_threads(num_thread);

  std::vector<std::vector<std::pair<int, float> > > data_vec;
  int max_fid, max_lid;

  int ret = DataConverter::ParseForFile(
                param_.train_file(), &data_vec, &labels_, &max_fid, &max_lid);

  if (ret != 1) { fprintf(stderr, "Data load error!: '%s'\n", param_.train_file().c_str()); return -1; }

  size_t num_data = data_vec.size();
  fprintf(stderr, "Load Train Data: #data=%lu #feature=%d #label=%d\n", num_data, max_fid, max_lid);

  size_t num_cluster = std::max(data_vec.size() / 6000, 1LU);
  size_t num_learner = param_.num_learner();

  std::mt19937 rnd_gen(param_.seed());
  std::uniform_int_distribution<int> sdist(0, INT_MAX);

  std::vector<int> c_seeds(num_learner);
  std::vector<std::vector<int> > e_seeds(num_learner);
  for (size_t i = 0; i < num_learner; ++i) {
    c_seeds[i] = sdist(rnd_gen);

    std::mt19937 local_rnd_gen(sdist(rnd_gen)); // for backward compatibility
    e_seeds[i].resize(num_cluster);
    for (size_t k = 0; k < num_cluster; ++k) { e_seeds[i][k] = sdist(local_rnd_gen); }
  }

  DataPartitioner().NormalizeData(&data_vec);

  std::vector<std::vector<std::vector<size_t> > > cluster_assign_vec(num_learner);
  std::vector<std::vector<std::vector<double> > > cluster_prec_vec(num_learner);
  for (size_t i = 0; i < num_learner; ++i) {
    cluster_assign_vec[i].resize(num_cluster);
    cluster_prec_vec[i].resize(num_cluster);
    for (size_t k = 0; k < num_cluster; ++k) { cluster_prec_vec[i][k].resize(5); }
  }


  LearnPartitioning(data_vec, num_learner, num_cluster, c_seeds, verbose, &cluster_assign_vec);

  LearnEmbedding(data_vec, num_learner, num_cluster, cluster_assign_vec, e_seeds, verbose, &cluster_prec_vec);


  for (size_t i = 0; i < num_learner; ++i) {
    std::vector<double> total_prec_vec(5);
    for (size_t k = 0; k < num_cluster; ++k) {
      for (size_t j = 0; j < cluster_prec_vec[i][k].size(); ++j) {
        total_prec_vec[j] += cluster_prec_vec[i][k][j] * cluster_assign_vec[i][k].size();
      }
    }

    fprintf(stderr, "Learner: %2lu (#data=%lu)", i, data_vec.size());
    for (size_t j = 0; j < total_prec_vec.size(); ++j) {
      fprintf(stderr, ", P@%lu: %.4f", j+1, total_prec_vec[j] / data_vec.size());
    }
    fprintf(stderr, "\n");
  }

  // save model
  FILE *fp = fopen(param_.model_file().c_str(), "wb");
  if (fp == NULL) { return -1; }
  param_.WriteToStream(fp);
  Utils::WriteNumToStream(labels_.size(), fp);
  for (size_t i = 0; i < labels_.size(); ++i) {
    Utils::WriteNumToStream(labels_[i].size(), fp);
    for (size_t j = 0; j < labels_[i].size(); ++j) { Utils::WriteNumToStream(labels_[i][j], fp); }
  }
  Utils::WriteNumToStream(num_learner, fp);
  for (size_t i = 0; i < num_learner; ++i) {
    partitioning_vec_[i].WriteToStream(fp);
    embedding_vec_[i].WriteToStream(fp);
  }
  fclose(fp);

  return 1;
}


int AnnexML::Predict() const {
  FILE *fp = fopen(param_.result_file().c_str(), "w");
  if (fp == NULL) { return -1; }

  size_t num_thread = (param_.num_thread() > 0) ? param_.num_thread() : 1;
  omp_set_num_threads(num_thread);

  std::vector<std::vector<std::pair<int, float> > > data_vec;
  std::vector<std::vector<int> > label_vec;
  int max_fid, max_lid;

  int ret = DataConverter::ParseForFile(
                param_.predict_file(), &data_vec, &label_vec, &max_fid, &max_lid);

  if (ret != 1) { fprintf(stderr, "Fail to open predict_file: '%s'\n", param_.predict_file().c_str()); return -1; }

  size_t num_data = data_vec.size();
  fprintf(stderr, "Load Predict Data: #data=%lu #feature=%d #label=%d\n", num_data, max_fid, max_lid);

  DataPartitioner().NormalizeData(&data_vec);

  size_t num_used_learner = param_.num_learner();
  if (num_used_learner > partitioning_vec_.size()) { num_used_learner = partitioning_vec_.size(); }

  std::vector<size_t> cls_results(num_data);
  std::vector<std::unordered_map<size_t, int> > emb_results(num_data);
  std::vector<std::vector<std::pair<int, float> > > results(num_data);

  size_t max_k = 10;
  auto comparator = [](const std::pair<int,float> &a,const std::pair<int,float> &b){return a.second>b.second;};

  double start = Utils::GetTimeOfDaySec();
  double c_elapsed = 0.0;
  double e_elapsed = 0.0;
  double a_elapsed = 0.0;
  double tmp_start = 0.0;
 
  for (size_t l = 0; l < num_used_learner; ++l) {
    cls_results.clear();

    // partitioning
    tmp_start = Utils::GetTimeOfDaySec();
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_data; ++i) {
      size_t cluster = partitioning_vec_[l].GetNearestCluster(data_vec[i]);
      cls_results[i] = cluster;
    }

    // for cache efficiency, group tasks by each cluster
    std::vector<std::vector<size_t> > cluster_assign(partitioning_vec_[l].K());
    for (size_t i = 0; i < num_data; ++i) {
      size_t cluster = cls_results[i];
      cluster_assign[cluster].push_back(i);
    }
    std::vector<size_t> flatten_cluster_assign;
    for (size_t k = 0; k < cluster_assign.size(); ++k) {
      for (size_t i = 0; i < cluster_assign[k].size(); ++i) {
        flatten_cluster_assign.push_back(cluster_assign[k][i]);
      }
    }
    c_elapsed += Utils::GetTimeOfDaySec() - tmp_start;

    // embedding
    tmp_start = Utils::GetTimeOfDaySec();
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_data; ++i) {
      size_t index = flatten_cluster_assign[i];
      size_t cluster = cls_results[index];
      auto &s_map = emb_results[index];
      std::vector<std::pair<size_t, float> > idx_vec;

      embedding_vec_[l].Search(data_vec[index], cluster, param_.num_nn(), param_.search_eps(), &idx_vec);
      for (size_t k = 0; k < idx_vec.size(); ++k) {
        size_t idx = idx_vec[k].first;
        auto itr = s_map.find(idx);
        if (itr == s_map.end()) { s_map[idx]   = 1; }
        else                    { itr->second += 1; }
      }
    }
    e_elapsed += Utils::GetTimeOfDaySec() - tmp_start;
  }

  // aggregating results
  tmp_start = Utils::GetTimeOfDaySec();
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < num_data; ++i) {
    const auto &s_map = emb_results[i];

    std::unordered_map<int, float> l_map;
    for (auto itr = s_map.cbegin(); itr != s_map.cend(); ++itr) {
      size_t idx = itr->first;
      for (size_t j = 0; j < labels_[idx].size(); ++j) {
        int l = labels_[idx][j];
        auto itr2 = l_map.find(l);
        if (itr2 == l_map.end()) { l_map[l]      = itr->second; }
        else                     { itr2->second += itr->second; }
      }
    }

    auto &l_vec = results[i];
    l_vec.assign(l_map.begin(), l_map.end());
    std::sort(l_vec.begin(), l_vec.end(), comparator);

    if (l_vec.size() > max_k) { l_vec.resize(max_k); }
  }
  a_elapsed += Utils::GetTimeOfDaySec() - tmp_start;

  fprintf(stderr, "Done!\n");


  double elapsed = Utils::GetTimeOfDaySec() - start;
  double elapsed_per_sample = 1000.0 * elapsed / num_data;
  fprintf(stderr, "Elapsed: %.4f sec, %.4f msec/sample (* %lu threads = %.4f msec/sample)\n", elapsed, elapsed_per_sample, num_thread, num_thread * elapsed_per_sample);
  fprintf(stderr, "Partitioning: %.4f sec, Embedding: %.4f sec, Aggregating: %.4f sec\n", c_elapsed, e_elapsed, a_elapsed);


  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < label_vec[i].size(); ++j) {
      if (j > 0) { fprintf(fp, ","); }
      fprintf(fp, "%d", label_vec[i][j]);
    }
    fprintf(fp, "\t");
    for (size_t k = 0; k < results[i].size(); ++k) {
      if (k > 0) { fprintf(fp, ","); }
      fprintf(fp, "%d:%.1f", results[i][k].first, results[i][k].second);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);

  return 1;
}

int AnnexML::CheckParam() {
  // TODO check param_
  return 1;
}

int AnnexML::MergeParam(const AnnexMLParameter &param) {
  // TODO
  return 1;
}

int AnnexML::LearnPartitioning(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                               size_t num_learner, size_t num_cluster,
                               const std::vector<int> &seed_vec, int verbose,
                               std::vector<std::vector<std::vector<size_t> > > *cluster_assign_vec) {
  omp_lock_t lock;
  omp_init_lock(&lock);

  size_t num_done_task = 0;
  if (verbose == 0) {
    fprintf(stderr, "\rPartitioning progress: %6.2f%% (#task=%lu/%lu)", 0.0f, num_done_task, num_learner);
  }

  std::vector<DataPartitioner> c_vec(num_learner);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < num_learner; ++i) {

    if (param_.cls_type() == 0) {
      c_vec[i].RunKmeans(data_vec, num_cluster, param_.cls_iter(), seed_vec[i], verbose);
    } else {
      c_vec[i].RunPairwise(data_vec, labels_, num_cluster, param_.cls_iter(),
                                     param_.num_nn(), param_.label_normalize(),
                                     param_.eta0(), param_.lambda(), param_.gamma(),
                                     seed_vec[i], verbose);
    }

    if (verbose == 0) {
      omp_set_lock(&lock);
      ++num_done_task;
      float progress = 100.0f * num_done_task / num_learner;
      fprintf(stderr, "\rPartitioning progress: %6.2f%% (#task=%lu/%lu)", progress, num_done_task, num_learner);
      omp_unset_lock(&lock);
    }
  }
  if (verbose == 0) { fprintf(stderr, "\rPartitioning done!                                    \n"); }

  // Force memory deallocation (something wrong in my computational environment...)
  // This is hacky
  partitioning_vec_.resize(num_learner);
  for (size_t i = 0; i < partitioning_vec_.size(); ++i) {
    partitioning_vec_[i].CopiedFrom(c_vec[i]);
  }
  std::vector<DataPartitioner>().swap(c_vec);

  std::vector<size_t> cluster_vec(data_vec.size());
  for (size_t i = 0; i < num_learner; ++i) {
#pragma omp parallel for schedule(dynamic)
    for (size_t j = 0; j < data_vec.size(); ++j) {
      cluster_vec[j] = partitioning_vec_[i].GetNearestCluster(data_vec[j]);
    }

    for (size_t j = 0; j < data_vec.size(); ++j) {
      size_t k = cluster_vec[j];
      (*cluster_assign_vec)[i][k].push_back(j);
    }
  }

  omp_destroy_lock(&lock);

  return 1;
}

int AnnexML::LearnEmbedding(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                            size_t num_learner, size_t num_cluster,
                            const std::vector<std::vector<std::vector<size_t> > > &cluster_assign_vec,
                            const std::vector<std::vector<int> > &seed_vec, int verbose,
                            std::vector<std::vector<std::vector<double> > > *cluster_prec_vec) {

  omp_lock_t lock;
  omp_init_lock(&lock);

  embedding_vec_.clear(); embedding_vec_.resize(num_learner);

  // prepare task queue
  std::vector<std::pair<int, int> > etask_vec;
  for (size_t i = 0; i < num_learner; ++i) {
    embedding_vec_[i].Init(data_vec.size(), num_cluster, param_.emb_size(), verbose);
    for (size_t j = 0; j < num_cluster; ++j) { etask_vec.push_back(std::make_pair(i, j)); }
  }

  size_t num_done_task = 0;
  if (verbose == 0) {
    fprintf(stderr, "\rEmbedding progress: %6.2f%% (#task=%lu/%lu)", 0.0f, num_done_task, etask_vec.size());
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < etask_vec.size(); ++i) {
    int lidx = etask_vec[i].first;
    int cidx = etask_vec[i].second;
    embedding_vec_[lidx].Learn(data_vec, labels_, cluster_assign_vec[lidx], cidx,
                               param_.num_nn(), param_.label_normalize(),
                               param_.eta0(), param_.gamma(), param_.emb_iter(),
                               seed_vec[lidx][cidx],
                               &((*cluster_prec_vec)[lidx][cidx]));

    if (verbose == 0) {
      omp_set_lock(&lock);
      ++num_done_task;
      float progress = 100.0f * num_done_task / etask_vec.size();
      fprintf(stderr, "\rEmbedding progress: %6.2f%% (#task=%lu/%lu)", progress, num_done_task, etask_vec.size());
      omp_unset_lock(&lock);
    }
  }
  if (verbose == 0) { fprintf(stderr, "\rEmbedding done!                                     \n"); }

  omp_destroy_lock(&lock);

  return 1;
}


} // namespace xmlc
} // namespace yj 
