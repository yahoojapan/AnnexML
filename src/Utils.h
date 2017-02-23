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

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>


namespace yj {
namespace xmlc {

class Utils {
  public:

    static double GetTimeOfDaySec() {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    template <typename NumType>
    static void WriteNumToStream(NumType val, FILE *stream) {
      fwrite(&val, sizeof(val), 1, stream);
    }

    template <typename NumType>
    static void ReadNumFromStream(FILE *stream, NumType *val) {
      assert(fread(val, sizeof(*val), 1, stream) == 1);
    }

    static void WriteStringToStream(const std::string &str, FILE *stream) {
      int size = static_cast<int>(str.size());
      fwrite(&size, sizeof(size), 1, stream);
      fwrite(str.c_str(), sizeof(char), size, stream);
    }

    static void ReadStringFromStream(FILE *stream, std::string *str) {
      int size = 0;
      assert(fread(&size, sizeof(size), 1, stream) == 1);

      char cbuff[size];
      assert(fread(cbuff, sizeof(char), size, stream) == static_cast<size_t>(size));

      str->assign(cbuff, size);
    }

};

} // namespace xmlc
} // namespace yj 
