cmake_minimum_required(VERSION 3.12)
project(flabsdk CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/Applications/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories("D:/Applications/yaml-cpp/include")
include_directories("D:/Applications/onnxruntime-gpu-1.20.0/include")
link_directories("D:/Applications/onnxruntime-gpu-1.20.0/lib")
link_directories("D:/Applications/yaml-cpp/build/Release")

file(GLOB MODULE_SRCS ${PROJECT_SOURCE_DIR}/src/infer_modules/*.cpp)
file(GLOB INFER_MODEL ${PROJECT_SOURCE_DIR}/src/infer_model/*.cpp)
file(GLOB UTILS_SRCS ${PROJECT_SOURCE_DIR}/src/utils/*.cpp)
file(GLOB TASKS_SRCS ${PROJECT_SOURCE_DIR}/src/tasks/*.cpp)
file(GLOB SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(flabsdk SHARED ${SRCS} 
                           ${MODULE_SRCS} 
                           ${UTILS_SRCS} 
                           ${TASKS_SRCS}
                           ${INFER_MODEL}
)
                           
target_link_libraries(flabsdk ${OpenCV_LIBS} onnxruntime yaml-cpp)

target_compile_definitions(flabsdk
  PRIVATE
  FLAB_COMPILE_LIBRARY
)

add_executable(test_sdk ${PROJECT_SOURCE_DIR}/run/test_sdk.cpp)
target_link_libraries(test_sdk flabsdk ${OpenCV_LIBS} yaml-cpp)