#include "utils/registry.h"
#include "spdlog/spdlog.h"
#include "infer_modules/det_module.h"
#include <memory>


namespace flabsdk {


	modules::BaseModule* get_module(const std::string& name) {
		bool uninited = false;
		modules::BaseModule* res = nullptr;
		if (name == "DetInferModule") {
			res = new modules::DetInferModule();
		}
		else if (name == "GlassBracketModule") {
			res = new modules::GlassBracketModule();
		}
		else {
			uninited = true;
		}

		if (!uninited) {
			//spdlog::info("The module is registered, name: {}", name);
			return res;
		}
		else {
			spdlog::info("The module is not registered, please check out the input.");
			return new modules::EmptyModule();
		}
	}



//ModuleRegistry& ModuleRegistry::instance() {
//    static ModuleRegistry registry;
//    return registry;
//}

//void ModuleRegistry::register_module(const std::string& name, InferModule infer_module) {
//    module_map[name] = infer_module;
//}

//std::unique_ptr<modules::BaseModule> ModuleRegistry::get_module(const std::string& name) {
//    auto it = module_map.find(name);
//    if (it != module_map.end()) {
//        return std::unique_ptr<modules::BaseModule>(it->second());
//    }
//    return std::make_unique<modules::EmptyModule>();
//}
//
//void ModuleRegistry::clear_modules() {
//    module_map.clear();
//}


} // namespace flabsdk
