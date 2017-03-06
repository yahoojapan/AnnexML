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

#include "FileReader.h"

#include <cstdio>
#include <string>
#include <vector>

namespace yj {
namespace xmlc {

FileReader::FileReader()
  : fp_(NULL), file_size_(0) {
}

FileReader::~FileReader() {
  if (fp_ != NULL) {
    fclose(fp_); fp_ = NULL;
  }
}

int FileReader::Open(const std::string &in_file) {
  if (fp_ != NULL) {
    fclose(fp_); fp_ = NULL;
  }

  fp_ = fopen(in_file.c_str(), "rb");

  if (fp_ == NULL) { return -1; }

  long cur_pos = ftell(fp_);
  fseek(fp_, 0, SEEK_END);
  file_size_ = ftell(fp_);
  fseek(fp_, cur_pos, SEEK_SET);

  return 1;
}

bool FileReader::isEOF() {
  if (fp_ == NULL) { return true; }
  return feof(fp_);
}

int FileReader::Rewind() {
  if (fp_ == NULL) { return -1; }
  return fseek(fp_, 0, SEEK_SET);
}

int FileReader::Seek(long offset, int origin) {
  if (fp_ == NULL) { return -1; }
  return fseek(fp_, offset, origin);
}

int FileReader::ReadLine(std::string *line) {
  if (fp_ == NULL || line == NULL) {
    return -1;
  }
  line->clear();

  int c_buff_len = 1024;
  char c_buff[c_buff_len];

  int ret = 0;
  do {
    if (fgets(c_buff, c_buff_len, fp_) == NULL) {
      break;
    }
    ret = 1;
    (*line) += c_buff;
  } while (line->rfind('\n') == std::string::npos); 

  while (line->size() > 0) {
    std::string::size_type nline_pos = line->rfind('\n');
    if (nline_pos == std::string::npos) {
      break;
    }
    line->erase(nline_pos, line->size() - nline_pos);
  }

  return ret;
}

int FileReader::ReadLines(size_t line_num, std::vector<std::string> *lines) {
  if (fp_ == NULL || lines == NULL) {
    return -1;
  }
  lines->clear();

  std::string line;

  while (!isEOF() && lines->size() < line_num) {
    ReadLine(&line);
    lines->push_back(line);
  }

  return 1;
}

size_t FileReader::GetFileSize() {
  return file_size_;
}

} // namespace xmlc
} // namespace yj 
