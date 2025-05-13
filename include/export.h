#pragma once

#if defined(_WIN32)
#if defined(FLAB_COMPILE_LIBRARY)
#define FLABINFER_EXPORT __declspec(dllexport)
#else
#define FLABINFER_EXPORT __declspec(dllimport)
#endif  // defined(FLAB_COMPILE_LIBRARY)
#else  // defined(_WIN32)
#if defined(FLAB_COMPILE_LIBRARY)
#define FLABINFER_EXPORT __attribute__((visibility("default")))
#else
#define FLABINFER_EXPORT
#endif
#endif  // defined(_WIN32)


