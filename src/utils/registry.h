#pragma once

#include "infer_modules/base_module.h"

namespace flabsdk {

	modules::BaseModule* get_module(const std::string& name);

//class ModuleRegistry {
//public:
//    //using InferModule = std::function<std::unique_ptr<modules::BaseModule> ()>;
//
//    static ModuleRegistry& instance();
//
//    //void register_module(const std::string& name, InferModule infer_module);
//
//    std::unique_ptr<modules::BaseModule> get_module(const std::string& name);
//
//	void clear_modules();
//
//private:
//    //std::unordered_map<std::string, InferModule> module_map;
//};

//#define REGISTER_MODULE(name, type) \
//    const bool registered_##type = []() { \
//        ModuleRegistry::instance().register_module(name, []() { return std::make_unique<modules::type()>; }); \
//        return true; \
//    }(); 


}  // namespace flabsdk