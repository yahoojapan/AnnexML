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

#include <cstdio>

#include "AnnexML.h"
#include "AnnexMLParameter.h"


int main(int argc, char **argv) {

  if (argc < 3) {
    fprintf(stderr, "Usage: %s <train or predict> <JSON file>\n", argv[0]);
    return -1;
  }

  const std::string mode(argv[1]);
  char *config_json_f = argv[2];

  std::vector<std::string> args;
  for (int i = 3; i < argc; ++i) { args.push_back(argv[i]); }

  if (mode != "train" && mode != "predict") {
    fprintf(stderr, "Mode should be 'train' or 'predict', unknown mode: '%s'\n", mode.c_str());
    return 1;
  }
  bool load_model = (mode == "predict") ? true : false;

  yj::xmlc::AnnexMLParameter param;
  if (param.ReadFromJSONFile(config_json_f) < 0) {
    fprintf(stderr, "Fail to read JSON config file\n");
    return 1;
  }
  if (param.UpdateFromArgs(args) < 0) {
    fprintf(stderr, "Fail to read args\n");
    return 1;
  }

  yj::xmlc::AnnexML classifier;

  int ret = classifier.Init(param, load_model);
  if (ret <= 0) { fprintf(stderr, "Init fail!\n"); }

  if (mode == "train") { classifier.Train(); }

  if (mode == "predict")  { classifier.Predict(); }

  return 0;
}
