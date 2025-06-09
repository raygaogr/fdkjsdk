#include <fstream>
#include <iostream>
#include "utils/io_utils.h"
#include "spdlog/spdlog.h"
#include <codecvt>


namespace flabsdk {
	std::wstring ConvertToWideString(const std::string& utf8_str) {
		std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
		return converter.from_bytes(utf8_str);
	}

	Status readFileStream(const std::string& filePath, std::vector<char>& cfgs) {
		std::wstring wFilePath = ConvertToWideString(filePath);
		std::ifstream file(wFilePath.c_str(), std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size_t size = file.tellg();
			file.seekg(0, file.beg);
			cfgs.resize(size);
			file.read(cfgs.data(), size);
			file.close();
		}
		else {
			spdlog::info("{} is invalid.", filePath);
			return Status::kInputInvalid;
		}
		if (cfgs.empty()) {
			spdlog::info("The content of {} is empty.", filePath);
			return Status::kInputInvalid;
		}
		return Status::kSuccess;	
	}
}  // namespace flabsdk