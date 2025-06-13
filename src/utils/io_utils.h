#pragma once

#include <string>
#include <vector>
#include "status.h"

namespace flabsdk {

	Status readFileStream(const std::string& filePath, std::vector<char>& cfgs, bool is_model_file = true);

} // namespace flabsdk