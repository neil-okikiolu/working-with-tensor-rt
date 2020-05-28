#ifndef __JETSONCAM_TENSOR_H__
#define __JETSONCAM_TENSOR_H__


#include <vector>
#include <cstdint>

namespace jetsoncam {
    
    /**
        Create a tensor class / struct, look at how
        tensors are used in places like "find_peaks_out_torch" in plugins.hpp!
        IT CAN BE DONE!
    */
    
    
    template <class T>
    class Tensor
    {
        public:
            Tensor();
            /**
                Create a tensor an empty tensor and allocate memory for it
            */
            Tensor(const char* name, std::vector<int> dimensions);

            /**
                Create a tensor an empty tensor and allocate memory for it
                and fill it with the fill values
            */
            Tensor(const char* name, std::vector<int> dimensions, T fill_value);

            /**
                Create a tensor with existing values in shared CPU, GPU memory
            */
            Tensor(const char* name, std::vector<int> tensorDims, T* outputCPU, T* outputCUDA);

            /**
             * Destroy
             */
            virtual ~Tensor();

            T operator [] (T i) const {return CUDA[i];}
            T& operator [] (T i) {return CUDA[i];}

            T* data_ptr();

            int size(int d) const;
            
            void printDims();
            
            T retrieve(std::vector<int> dimensions);

            /**
            * Return the stride of a tensor at some dimension.
            */
            // virtual int64_t stride(int64_t d) const;
            const char* tensor_name;
            std::vector<int> dims;
            size_t dims_size;
            uint32_t dataSize;
            T* CPU;
            T* CUDA;
        protected:
    };
    
    
    struct ParseResult
    {
        Tensor<int> object_counts;
        Tensor<int> objects;
        Tensor<float> normalized_peaks;
    };
}

#include "Tensor.tcc"

#endif
