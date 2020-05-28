#include "plugins.hpp"
#include "Tensor.h"

#include "parse/connect_parts.hpp"
#include "parse/find_peaks.hpp"
#include "parse/munkres.hpp"
#include "parse/paf_score_graph.hpp"
#include "parse/refine_peaks.hpp"
// #include "train/generate_cmap.hpp"
// #include "train/generate_paf.hpp"

#include <vector>

namespace trt_pose {
  namespace plugins {

    using namespace trt_pose::parse;
//     using namespace trt_pose::train;
    using namespace jetsoncam;

    std::vector<Tensor<int>> find_peaks_torch(
        Tensor<float> input,
        const float threshold,
        const int window_size,
        const int max_count
    )
    {
        const int N = input.size(0);
        const int C = input.size(1);
        const int H = input.size(2);
        const int W = input.size(3);
        const int M = max_count;
        
        // printf("find_peaks_torch, {N,C,H,W,M}-> {{%d,%d,%d,%d,%d}}\n", N,C,H,W,M);

        // create output tensors
        Tensor<int> counts = Tensor<int>(
            "counts",
            {N, C}
        );

        Tensor<int> peaks = Tensor<int>(
            "peaks",
            {N, C, M, 2}
        );

        // find peaks        
        // get pointers to tensor data
        int *counts_ptr = (int *)counts.data_ptr();
        int *peaks_ptr = (int *)peaks.data_ptr();
        const float *input_ptr = (const float *)input.data_ptr();

        // find peaks
        find_peaks_out_nchw(
            counts_ptr,
            peaks_ptr,
            input_ptr,
            N, C, H, W, M,
            threshold,
            window_size
        ); 

        return {counts, peaks};
    }


    Tensor<float> refine_peaks_torch(
        Tensor<int> counts,
        Tensor<int> peaks,
        Tensor<float> cmap,
        int window_size
    ) {
        
        Tensor<float> refined_peaks = Tensor<float>(
            "refined_peaks",
            {peaks.size(0), peaks.size(1), peaks.size(2), peaks.size(3)}
        );
        
        // printf("refine_peaks_torch, cmap.dims_size:: %lu, cmap.dims(3):: %d\n", cmap.dims_size, cmap.size(3));
        
        const int N = cmap.size(0);
        const int C = cmap.size(1);
        const int H = cmap.size(2);
        const int W = cmap.size(3);
        const int M = peaks.size(2);
        
        // printf("refine_peaks_torch, {N,C,H,W,M}-> {%d,%d,%d,%d,%d}\n", N,C,H,W,M);
        // printf("refine_peaks_torch: peaks.size(0): %d, peaks.size(1): %d, peaks.size(2): %d, peaks.size(3): %d\n", peaks.size(0), peaks.size(1), peaks.size(2), peaks.size(3));

        refine_peaks_out_nchw(
            (float *)refined_peaks.data_ptr(),
            (const int *)counts.data_ptr(),
            (const int *)peaks.data_ptr(),
            (const float *)cmap.data_ptr(),
            N, C, H, W, M,
            window_size
        );
        return refined_peaks;
    }


    Tensor<float> paf_score_graph_torch(
        Tensor<float> paf,
        Tensor<int> topology,
        Tensor<int> counts,
        Tensor<float> peaks,
        const int num_integral_samples
    ) {

        const int N = peaks.size(0);
        const int K = topology.size(0);
        const int M = peaks.size(2);
        
        // printf("paf_score_graph_torch, {N,K,M}-> {%d,%d,%d}\n", N,K,M);

        Tensor<float> score_graph = Tensor<float>(
            "score_graph",
            {N, K, M, M}
        );

        const int N_2 = paf.size(0);
        const int K_2 = topology.size(0);
        const int C_2 = peaks.size(1);
        const int H_2 = paf.size(2);
        const int W_2 = paf.size(3);
        const int M_2 = score_graph.size(3);
        
        // printf("paf_score_graph_torch, {N_2,K_2,C_2,H_2,W_2,M_2}-> {%d,%d,%d,%d,%d,%d}\n", N_2,K_2,C_2,H_2,W_2,M_2);

        paf_score_graph_out_nkhw(
            (float *)score_graph.data_ptr(),
            (const int *)topology.data_ptr(),
            (const float *)paf.data_ptr(),
            (const int *)counts.data_ptr(),
            (const float *)peaks.data_ptr(),
            N_2, K_2, C_2, H_2, W_2, M_2,
            num_integral_samples
        );

        return score_graph;
    }


    Tensor<int> assignment_torch(
        Tensor<float> score_graph,
        Tensor<int> topology,
        Tensor<int> counts,
        float score_threshold
    ) {
        int N = counts.size(0);
        int K = topology.size(0);
        int M = score_graph.size(2);
        
        // printf("assignment_torch, {N,K,M}-> {%d,%d,%d}\n", N,K,M);

        Tensor<int> connections = Tensor<int>(
            "connections",
            {N, K, 2, M},
            -1
        );

        const int C = counts.size(1);
        void *workspace = (void *)malloc(assignment_out_workspace(M));

        assignment_out_nk(
            (int *)connections.data_ptr(),
            (const float *)score_graph.data_ptr(),
            (const int *)topology.data_ptr(),
            (const int *)counts.data_ptr(),
            N, C, K, M,
            score_threshold,
            workspace
        );

        free(workspace);

        return connections;
    }


    std::vector<Tensor<int>> connect_parts_torch(
        Tensor<int> connections,
        Tensor<int> topology,
        Tensor<int> counts,
        int max_count
    )
    {
        
        int N = counts.size(0);
        int K = topology.size(0);
        int C = counts.size(1);
        int M = connections.size(3);
        
        // printf("connect_parts_torch, {N,K,C,M,max_count}-> {%d,%d,%d,%d,%d}\n", N,K,C,M,max_count);

        Tensor<int> objects = Tensor<int>(
            "objects",
            {N, max_count, C},
            -1
        );

        Tensor<int> object_counts = Tensor<int>(
            "object_counts",
            {N}
        );
        
        const int N_2 = object_counts.size(0);
        const int K_2 = topology.size(0);
        const int C_2 = counts.size(1);
        const int M_2 = connections.size(3);
        const int P_2 = max_count;
        
        // printf("connect_parts_torch, {N_2,K_2,C_2,M_2,P_2}-> {%d,%d,%d,%d,%d}\n", N_2,K_2,C_2,M_2,P_2);
        
        void *workspace = malloc(connect_parts_out_workspace(C_2, M_2));
        connect_parts_out_batch(
            (int *) object_counts.data_ptr(),
            (int *) objects.data_ptr(),
            (const int *) connections.data_ptr(),
            (const int *) topology.data_ptr(),
            (const int *) counts.data_ptr(),
            N_2, K_2, C_2, M_2, P_2,
            workspace
        );
        free(workspace);

        return {object_counts, objects};
    }

    
    Tensor<int> coco_category_to_topology(std::vector<std::vector<int>> skeleton)
    {
   
        const int K = static_cast<int>(skeleton.size());
        
        // create output tensors
        Tensor<int> topology = Tensor<int>(
            "topology",
            {K, 4}
        );
        
        // printf("coco_category_to_topology, {K, 4}-> {%d, 4}\n", K);

        for (int k = 0; k < K; k++) {
            std::vector<int> skel_item = skeleton[k];
            topology.CUDA[(k * 4) + 0] = 2 * k;
            topology.CUDA[(k * 4) + 1] = 2 * k + 1;
            topology.CUDA[(k * 4) + 2] = skel_item[0] - 1;
            topology.CUDA[(k * 4) + 3] = skel_item[1] - 1;
        }

        return topology;
    }
  } // namespace plugins
} // namespace trt_pose
