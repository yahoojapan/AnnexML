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
#include <set>

namespace yj {
namespace xmlc {

class Reindexer {
  public:
    Reindexer() : o2n_vec_(1024), n2o_vec_(1024) {};
    Reindexer(size_t max_id) : o2n_vec_(max_id + 1), n2o_vec_(max_id + 1) {};
    ~Reindexer() {};

    void Clear() {
      std::fill_n(o2n_vec_.begin(), o2n_vec_.size(), 0);
      std::fill_n(n2o_vec_.begin(), n2o_vec_.size(), 0);
    };

    int IndexLabels(const std::vector<std::vector<int> > &labels,
                    const std::vector<size_t> &index,
                    std::vector<std::vector<int> > *r_labels) {
      Clear();
      r_labels->resize(index.size());

      std::set<int> id_set;
      for (size_t i = 0; i < index.size(); ++i) {
        size_t idx = index[i];
        (*r_labels)[i].resize(labels[idx].size());
        for (size_t j = 0; j < labels[idx].size(); ++j) { id_set.insert(labels[idx][j]); }
      }

      int new_id = 0;
      for (auto itr = id_set.begin(); itr != id_set.end(); ++itr) {
        int old_id = *itr;
        ++new_id;

        AddToO2NVec(old_id, new_id);
        AddToN2OVec(new_id, old_id);
      }

      for (size_t i = 0; i < index.size(); ++i) {
        size_t idx = index[i];
        for (size_t j = 0; j < labels[idx].size(); ++j) {
          (*r_labels)[i][j] = O2N(labels[idx][j]);
        }
      }

      return new_id;
    };

    int IndexData(const std::vector<std::vector<std::pair<int, float> > > &data,
                  const std::vector<size_t> &index,
                  std::vector<std::vector<std::pair<int, float> > > *r_data) {
      Clear();
      r_data->resize(index.size());

      std::set<int> id_set;
      for (size_t i = 0; i < index.size(); ++i) {
        size_t idx = index[i];
        (*r_data)[i].resize(data[idx].size());
        for (size_t j = 0; j < data[idx].size(); ++j) { id_set.insert(data[idx][j].first); }
      }

      int new_id = 0;
      for (auto itr = id_set.begin(); itr != id_set.end(); ++itr) {
        int old_id = *itr;
        ++new_id;

        AddToO2NVec(old_id, new_id);
        AddToN2OVec(new_id, old_id);
      }

      for (size_t i = 0; i < index.size(); ++i) {
        size_t idx = index[i];
        for (size_t j = 0; j < data[idx].size(); ++j) {
          (*r_data)[i][j].first  = O2N(data[idx][j].first);
          (*r_data)[i][j].second = data[idx][j].second;
        }
      }

      return new_id;
    };

    int O2N(int idx) { return o2n_vec_[idx]; };
    int N2O(int idx) { return n2o_vec_[idx]; };

  private:
    void AddToO2NVec(size_t old_id, int new_id) {
        if (old_id >= o2n_vec_.size()) { o2n_vec_.resize(old_id + 1, 0); }
        o2n_vec_[old_id] = new_id;
    };

    void AddToN2OVec(size_t new_id, int old_id) {
        if (new_id >= n2o_vec_.size()) { n2o_vec_.resize(new_id + 1, 0); }
        n2o_vec_[new_id] = old_id;
    };

    std::vector<int> o2n_vec_;
    std::vector<int> n2o_vec_;

};


} // namespace xmlc
} // namespace yj 
