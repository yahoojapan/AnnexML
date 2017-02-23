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
#include <vector>

namespace yj {
namespace xmlc {

class DataConverter {
  public:

    static int ParseForFile(
        const std::string &file_name,
        std::vector<std::vector<std::pair<int, float> > > *dvec,
        std::vector<std::vector<int> > *lvec,
        int *max_fid, int *max_lid) {
      if (dvec == NULL || lvec == NULL) { return -1; }
      dvec->clear(); lvec->clear();

      yj::xmlc::FileReader reader;
      int ret = reader.Open(file_name);
      if (ret != 1) { fprintf(stderr, "Fail to open file: '%s'\n", file_name.c_str()); return -1; }

      std::vector<std::pair<int, float> > dv;
      std::vector<int> lv;
      int mfid = 0, mlid = 0;
      int mf = 0, ml = 0;

      std::string line;
      while (! reader.isEOF()) {
        reader.ReadLine(&line);
        if (line.empty()) { continue; }

        int ret = ParseForLine(line, &dv, &lv, &mf, &ml);
        if (ret != 1) { continue; }

        dvec->push_back(dv);
        lvec->push_back(lv);
        if (mf > mfid) { mfid = mf; }
        if (ml > mlid) { mlid = ml; }
      }

      if (max_fid != NULL) { *max_fid = mfid; }
      if (max_lid != NULL) { *max_lid = mlid; }

      return 1;
    };

    static int ParseForLine(
        const std::string &line,
        std::vector<std::pair<int, float> > *dvec,
        std::vector<int> *lvec,
        int *max_fid, int *max_lid) {
      if (dvec == NULL || lvec == NULL) { return -1; }

      dvec->clear(); lvec->clear();
      int mfid = 0, mlid = 0;

      char c_buff[line.size() + 1];
      strncpy(c_buff, line.c_str(), line.size() + 1);

      char *save_ptr;
      char *p = c_buff;

      // format: l1,l2,l3 <space or TAB> f1:v1 <space or TAB> f2:v2 <space or TAB> f3:v3,...
      //
      char *l_pos = strtok_r(p, " \t\n", &save_ptr); 
      if (l_pos == NULL) { return -1; }

      while (true) {
        char *f = strtok_r(NULL, ":", &save_ptr);
        char *v = strtok_r(NULL, " \t", &save_ptr);
        if (v == NULL) { break; }

        int   fid = static_cast<int>(strtol(f, NULL, 10));
        float val = static_cast<float>(strtod(v, NULL));
        dvec->push_back(std::make_pair(fid, val));

        if (fid > mfid) { mfid = fid; }
      }
      if (dvec->size() == 0) { return 0; }

      p = l_pos;
      while (true) {
        char *l = strtok_r(p, ",", &save_ptr);
        if (l == NULL) { break; }

        int lid = static_cast<int>(strtol(l, NULL, 10));
        lvec->push_back(lid);

        if (lid > mlid) { mlid = lid; }
        p = NULL;
      }
      if (lvec->size() == 0) { return 0; }

      if (max_fid != NULL) { *max_fid = mfid; }
      if (max_lid != NULL) { *max_lid = mlid; }

      return 1;
    };

};


} // namespace xmlc
} // namespace yj 
