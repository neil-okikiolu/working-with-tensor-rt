#ifndef __JETSONCAM_PLUGINS_H__
#define __JETSONCAM_PLUGINS_H__

#include "Tensor.h"

#include <vector>


namespace trt_pose {
  namespace plugins {
      
  using namespace jetsoncam;

    std::vector<Tensor<int>> find_peaks_torch(
        Tensor<float> input,
        const float threshold,
        const int window_size,
        const int max_count
    );

    Tensor<float> refine_peaks_torch(
        Tensor<int> counts,
        Tensor<int> peaks,
        Tensor<float> cmap,
        int window_size
    );

    Tensor<float> paf_score_graph_torch(
        Tensor<float> paf,
        Tensor<int> topology,
        Tensor<int> counts,
        Tensor<float> peaks,
        const int num_integral_samples
    );

    Tensor<int> assignment_torch(
        Tensor<float> score_graph,
        Tensor<int> topology,
        Tensor<int> counts,
        float score_threshold
    );

    std::vector<Tensor<int>> connect_parts_torch(
        Tensor<int> connections,
        Tensor<int> topology,
        Tensor<int> counts,
        int max_count
    );


    Tensor<int> coco_category_to_topology(std::vector<std::vector<int>> skeleton);

  } // namespace plugins
} // namespace trt_pose

#endif