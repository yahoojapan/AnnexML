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

#include "FileReader.h"
#include "AnnexMLParameter.h"
#include "DataPartitioner.h"
#include "LLEmbedding.h"

namespace yj {
namespace xmlc {

class AnnexML {
  public:
    AnnexML();
    virtual ~AnnexML();
    int Init(const AnnexMLParameter &param, bool load_model);
    int Train();
    int Predict() const;

  private:
    int CheckParam();
    int MergeParam(const AnnexMLParameter &param);
    int LearnPartitioning(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                          size_t num_learner, size_t num_cluster,
                          const std::vector<int> &seed_vec, int verbose,
                          std::vector<std::vector<std::vector<size_t> > > *cluster_assign_vec);
    int LearnEmbedding(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                       size_t num_learner, size_t num_cluster,
                       const std::vector<std::vector<std::vector<size_t> > > &cluster_assign_vec,
                       const std::vector<std::vector<int> > &seed_vec, int verbose,
                       std::vector<std::vector<std::vector<double> > > *cluster_prec_vec);

    AnnexMLParameter param_;
    std::vector<DataPartitioner> partitioning_vec_;
    std::vector<LLEmbedding> embedding_vec_;
    std::vector<std::vector<int> > labels_;
};

} // namespace xmlc
} // namespace yj 
