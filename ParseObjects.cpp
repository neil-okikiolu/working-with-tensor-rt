#include "ParseObjects.hpp"
#include "plugins.hpp"
#include "Tensor.h"

#include <vector>
#include <stdint.h>

namespace trt_pose {

    using namespace plugins;
    using namespace jetsoncam;

    ParseObjects::ParseObjects(){}
    // constructor
    ParseObjects::ParseObjects(
        Tensor<int> top
    )
    {
        topology = top;
    }

    
    ParseResult ParseObjects::Parse(Tensor<float> cmap, Tensor<float> paf) {
        // float l_cmap_threshold = 0.1f;
        // float l_link_threshold = 0.1f;
        float l_cmap_threshold = 0.15f;
        float l_link_threshold = 0.15f;
        int l_cmap_window = 5;
        int l_line_integral_samples = 7;
        int l_max_num_parts = 100;
        int l_max_num_objects = 100;

        std::vector<Tensor<int>> found_peaks = find_peaks_torch(cmap, l_cmap_threshold, l_cmap_window, l_max_num_parts);
        
        Tensor<int> peak_counts = found_peaks[0];
        Tensor<int> peaks = found_peaks[1];

        Tensor<float> normalized_peaks = refine_peaks_torch(peak_counts, peaks, cmap, l_cmap_window);

        Tensor<float> score_graph = paf_score_graph_torch(
            paf,
            topology,
            peak_counts,
            normalized_peaks,
            l_line_integral_samples
        );

        Tensor<int> connections = assignment_torch(
            score_graph,
            topology,
            peak_counts,
            l_link_threshold
        );

        // separated into object_counts, objects
        std::vector<Tensor<int>> connected_parts = connect_parts_torch(connections, topology, peak_counts, l_max_num_objects);

        Tensor<int> object_counts = connected_parts[0];
        Tensor<int> objects = connected_parts[1];
        
        ParseResult p_result;
        p_result.object_counts = object_counts;
        p_result.objects = objects;
        p_result.normalized_peaks = normalized_peaks;
        return p_result;
    }
}
