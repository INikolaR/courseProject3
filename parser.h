#pragma once
#include "DataLoader.h"

namespace neural_network {
namespace parser {
DataLoader parseMNISTDataset(const std::string& path_to_images_file,
                             const std::string& path_to_labels_file);
TrainTestLoaders parseTitanicDataset(const std::string& filename);
TrainTestLoaders parseBostonDataset(const std::string& filename);
}  // namespace parser
}  // namespace neural_network
