// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include "stubs/common.h"

#include "generated_message_util.h"

#include <limits>


namespace google {
namespace protobuf {
namespace internal {

double Infinity() {
  return std::numeric_limits<double>::infinity();
}
double NaN() {
  return std::numeric_limits<double>::quiet_NaN();
}

ExplicitlyConstructed< TProtoStringType> fixed_address_empty_string;
GOOGLE_PROTOBUF_DECLARE_ONCE(empty_string_once_init_);

void DeleteEmptyString() {
  GetEmptyStringAlreadyInited().~string();
}

void InitEmptyString() {
  fixed_address_empty_string.DefaultConstruct();
  OnShutdown(&DeleteEmptyString);
}

int StringSpaceUsedExcludingSelf(const string& str) {
  const void* start = &str;
  const void* end = &str + 1;
  if (start <= str.data() && str.data() < end) {
    // The string's data is stored inside the string object itself.
    return 0;
  } else {
    return str.capacity();
  }
}



void MergeFromFail(const char* file, int line) {
  GOOGLE_CHECK(false) << file << ":" << line;
  // Open-source GOOGLE_CHECK(false) is not NORETURN.
  exit(1);
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google
