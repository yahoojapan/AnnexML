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

class FileReader {
  public:
    FileReader();
    virtual ~FileReader();
    int Open(const std::string &in_file);
    bool isEOF();
    int Rewind();
    int Seek(long offset, int origin);
    int ReadLine(std::string *line);
    int ReadLines(size_t line_num, std::vector<std::string> *lines);
    size_t GetFileSize();

  private:
    FILE *fp_;
    size_t file_size_;
};


} // namespace xmlc
} // namespace yj 
