#pragma once

namespace flabsdk {

enum class Status {
  kSuccess = 0,
  kInitLogFailed = 1,
  kWrongState = 2,
  kLoadModelFailed = 3,
  kInputInvalid = 4,
  kOutputInvalid = 5,
  kNotFound = 6,
  kRunInferFailed = 7,
  kReadFileFailed = 8,
  kUnknownError = 9
};

}  // namespace flabsdk


