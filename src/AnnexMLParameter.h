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

#include <string>

#include "picojson.h"

#include "Utils.h"

namespace yj {
namespace xmlc {

class AnnexMLParameter {
  public:
    AnnexMLParameter()
      : emb_size_(50),
        num_learner_(15),
        num_nn_(10),
        cls_type_(1),
        cls_iter_(10),
        emb_iter_(10),
        label_normalize_(1),
        eta0_(0.1f),
        lambda_(4.0f),
        gamma_(10.0f),
        pred_type_(1),
        num_edge_(50),
        search_eps_(0.0f),
        num_thread_(1),
        seed_(0x5EED),
        verbose_(1),
        train_file_(),
        predict_file_(),
        model_file_(),
        result_file_()
    {};
    virtual ~AnnexMLParameter() {};

    inline int emb_size() const { return emb_size_; };
    inline void set_emb_size(int value) { emb_size_ = value; };

    inline int num_learner() const { return num_learner_; };
    inline void set_num_learner(int value) { num_learner_ = value; };

    inline int num_nn() const { return num_nn_; };
    inline void set_num_nn(int value) { num_nn_ = value; };

    inline int cls_type() const { return cls_type_; };
    inline void set_cls_type(int value) { cls_type_ = value; };

    inline int cls_iter() const { return cls_iter_; };
    inline void set_cls_iter(int value) { cls_iter_ = value; };

    inline int emb_iter() const { return emb_iter_; };
    inline void set_emb_iter(int value) { emb_iter_ = value; };

    inline int label_normalize() const { return label_normalize_; };
    inline void set_label_normalize(int value) { label_normalize_ = value; };

    inline float eta0() const { return eta0_; };
    inline void set_eta0(float value) { eta0_ = value; };

    inline float lambda() const { return lambda_; };
    inline void set_lambda(float value) { lambda_ = value; };

    inline float gamma() const { return gamma_; };
    inline void set_gamma(float value) { gamma_ = value; };

    inline int pred_type() const { return pred_type_; };
    inline void set_pred_type(float value) { pred_type_ = value; };

    inline int num_edge() const { return num_edge_; };
    inline void set_num_edge(float value) { num_edge_ = value; };

    inline float search_eps() const { return search_eps_; };
    inline void set_search_eps(float value) { search_eps_ = value; };

    inline int num_thread() const { return num_thread_; };
    inline void set_num_thread(int value) { num_thread_ = value; };

    inline int seed() const { return seed_; };
    inline void set_seed(int value) { seed_ = value; };

    inline int verbose() const { return verbose_; };
    inline void set_verbose(int value) { verbose_ = value; };

    inline bool has_train_file() const { return !(train_file_.empty()); };
    inline void clear_train_file() { train_file_.clear(); };
    inline const std::string& train_file() const { return train_file_; };
    inline void set_train_file(const std::string &value) { train_file_ = value; };

    inline bool has_predict_file() const { return !(predict_file_.empty()); };
    inline void clear_predict_file() { predict_file_.clear(); };
    inline const std::string& predict_file() const { return predict_file_; };
    inline void set_predict_file(const std::string &value) { predict_file_ = value; };

    inline bool has_model_file() const { return !(model_file_.empty()); };
    inline void clear_model_file() { model_file_.clear(); };
    inline const std::string& model_file() const { return model_file_; };
    inline void set_model_file(const std::string &value) { model_file_ = value; };

    inline bool has_result_file() const { return !(result_file_.empty()); };
    inline void clear_result_file() { result_file_.clear(); };
    inline const std::string& result_file() const { return result_file_; };
    inline void set_result_file(const std::string &value) { result_file_ = value; };

    int ReadFromJSONFile(const std::string &json_f) {
      std::ifstream ifs(json_f);
      if (! ifs.is_open()) { return -1; }

      std::string err;
      picojson::value v;
      err = picojson::parse(v, ifs);
      if (! err.empty()) { fprintf(stderr, "ERROR: %s\n", err.c_str()); return -1; }

      if (! v.is<picojson::object>()) { fprintf(stderr, "ERROR: File format error\n"); return -1; }
      const picojson::value::object& obj = v.get<picojson::object>();

      for (picojson::value::object::const_iterator itr = obj.begin(); itr != obj.end(); ++itr) {
        if (itr->first == "emb_size")        { emb_size_ = itr->second.get<double>(); }
        if (itr->first == "num_learner")     { num_learner_ = itr->second.get<double>(); }
        if (itr->first == "num_nn")          { num_nn_ = itr->second.get<double>(); }
        if (itr->first == "cls_type")        { cls_type_ = itr->second.get<double>(); }
        if (itr->first == "cls_iter")        { cls_iter_ = itr->second.get<double>(); }
        if (itr->first == "emb_iter")        { emb_iter_ = itr->second.get<double>(); }
        if (itr->first == "label_normalize") { label_normalize_ = itr->second.get<double>(); }
        if (itr->first == "eta0")            { eta0_ = itr->second.get<double>(); }
        if (itr->first == "lambda")          { lambda_ = itr->second.get<double>(); }
        if (itr->first == "gamma")           { gamma_ = itr->second.get<double>(); }
        if (itr->first == "pred_type")       { pred_type_ = itr->second.get<double>(); }
        if (itr->first == "num_edge")        { num_edge_ = itr->second.get<double>(); }
        if (itr->first == "search_eps")      { search_eps_ = itr->second.get<double>(); }
        if (itr->first == "num_thread")      { num_thread_ = itr->second.get<double>(); }
        if (itr->first == "seed")            { seed_ = itr->second.get<double>(); }
        if (itr->first == "verbose")         { verbose_ = itr->second.get<double>(); }
        if (itr->first == "train_file")      { train_file_   = itr->second.get<std::string>(); }
        if (itr->first == "predict_file")    { predict_file_ = itr->second.get<std::string>(); }
        if (itr->first == "model_file")      { model_file_   = itr->second.get<std::string>(); }
        if (itr->first == "result_file")     { result_file_  = itr->second.get<std::string>(); }
      }

      return 1;
    }

    int UpdateFromArgs(const std::vector<std::string> &args) {
      for (size_t i = 0; i < args.size(); ++i) {
        const std::string &kv = args[i];
        if (kv.empty()) { continue; }
        size_t pos = kv.find('=');

        size_t val_len = kv.size() - (pos + 1);
        const std::string key = kv.substr(0, pos);
        const std::string val = kv.substr(pos+1, val_len);

        if (key == "emb_size")        { emb_size_ = std::stoi(val); }
        if (key == "num_learner")     { num_learner_ = std::stoi(val); }
        if (key == "num_nn")          { num_nn_ = std::stoi(val); }
        if (key == "cls_type")        { cls_type_ = std::stoi(val); }
        if (key == "cls_iter")        { cls_iter_ = std::stoi(val); }
        if (key == "emb_iter")        { emb_iter_ = std::stoi(val); }
        if (key == "label_normalize") { label_normalize_ = std::stoi(val); }
        if (key == "eta0")            { eta0_ = std::stof(val); }
        if (key == "lambda")          { lambda_ = std::stof(val); }
        if (key == "gamma")           { gamma_ = std::stof(val); }
        if (key == "pred_type")       { pred_type_ = std::stoi(val); }
        if (key == "num_edge")        { num_edge_ = std::stoi(val); }
        if (key == "search_eps")      { search_eps_ = std::stof(val); }
        if (key == "num_thread")      { num_thread_ = std::stoi(val); }
        if (key == "seed")            { seed_ = std::stoi(val); }
        if (key == "verbose")         { verbose_ = std::stoi(val); }
        if (key == "train_file")      { train_file_   = val; }
        if (key == "predict_file")    { predict_file_ = val; }
        if (key == "model_file")      { model_file_   = val; }
        if (key == "result_file")     { result_file_  = val; }
      }
      return 1;
    }

    int WriteToStream(FILE *stream) const {
      if (stream == NULL) { return -1; }

      yj::xmlc::Utils::WriteNumToStream(emb_size_, stream);
      yj::xmlc::Utils::WriteNumToStream(num_learner_, stream);
      yj::xmlc::Utils::WriteNumToStream(num_nn_, stream);
      yj::xmlc::Utils::WriteNumToStream(cls_type_, stream);
      yj::xmlc::Utils::WriteNumToStream(cls_iter_, stream);
      yj::xmlc::Utils::WriteNumToStream(emb_iter_, stream);
      yj::xmlc::Utils::WriteNumToStream(label_normalize_, stream);
      yj::xmlc::Utils::WriteNumToStream(eta0_, stream);
      yj::xmlc::Utils::WriteNumToStream(lambda_, stream);
      yj::xmlc::Utils::WriteNumToStream(gamma_, stream);
      yj::xmlc::Utils::WriteNumToStream(pred_type_, stream);
      yj::xmlc::Utils::WriteNumToStream(num_edge_, stream);
      yj::xmlc::Utils::WriteNumToStream(search_eps_, stream);
      yj::xmlc::Utils::WriteNumToStream(num_thread_, stream);
      yj::xmlc::Utils::WriteNumToStream(seed_, stream);
      yj::xmlc::Utils::WriteNumToStream(verbose_, stream);
      yj::xmlc::Utils::WriteStringToStream(train_file_, stream);
      yj::xmlc::Utils::WriteStringToStream(predict_file_, stream);
      yj::xmlc::Utils::WriteStringToStream(model_file_, stream);
      yj::xmlc::Utils::WriteStringToStream(result_file_, stream);

      return 1;
    };

    int ReadFromStream(FILE *stream) {
      if (stream == NULL) { return -1; }

      yj::xmlc::Utils::ReadNumFromStream(stream, &emb_size_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &num_learner_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &num_nn_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &cls_type_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &cls_iter_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &emb_iter_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &label_normalize_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &eta0_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &lambda_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &gamma_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &pred_type_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &num_edge_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &search_eps_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &num_thread_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &seed_);
      yj::xmlc::Utils::ReadNumFromStream(stream, &verbose_);
      yj::xmlc::Utils::ReadStringFromStream(stream, &train_file_);
      yj::xmlc::Utils::ReadStringFromStream(stream, &predict_file_);
      yj::xmlc::Utils::ReadStringFromStream(stream, &model_file_);
      yj::xmlc::Utils::ReadStringFromStream(stream, &result_file_);

      return 1;
    };

  private:
    int emb_size_;
    int num_learner_;
    int num_nn_;
    int cls_type_;
    int cls_iter_;
    int emb_iter_;
    int label_normalize_;
    float eta0_;
    float lambda_;
    float gamma_;
    int pred_type_;
    int num_edge_;
    float search_eps_;
    int num_thread_;
    int seed_;
    int verbose_;
    std::string train_file_;
    std::string predict_file_;
    std::string model_file_;
    std::string result_file_;
};


} // namespace xmlc
} // namespace yj 
