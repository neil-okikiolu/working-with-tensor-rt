#include "Tensor.h"

#include <jetson-utils/cudaMappedMemory.h>
#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>

namespace jetsoncam {

    template <class T>
    Tensor<T>::Tensor(){}
    
    
    template <class T>
    Tensor<T>::Tensor(const char* name, std::vector<int> tensorDims)
    {
        // allocate output memory 
        void* outputCPU  = NULL;
        void* outputCUDA = NULL;

        // size_t outputSize = 1 * DIMS_C(tensorDims) * DIMS_H(tensorDims) * DIMS_W(tensorDims) * sizeof(T);
        size_t outputSize = 1;
        
        for (std::size_t i = 0; i < tensorDims.size(); i++) {
            // access element as v[i]
            int val = tensorDims[i];
            outputSize *= val;
            // any code including continue, break, return
        }

        outputSize *= sizeof(T);


        if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
        {
            printf("failed to alloc CUDA mapped memory for tensor %s output, %zu bytes\n", name, outputSize);
        }

        // create output tensors
        tensor_name = name;
        CPU  = (T*)outputCPU;
        CUDA = (T*)outputCUDA;
        dataSize = outputSize;
        dims = tensorDims;
        dims_size = tensorDims.size();
    }

    template <class T>
    Tensor<T>::Tensor(const char* name, std::vector<int> tensorDims, T fill_value)
    {
        // allocate output memory 
        void* outputCPU  = NULL;
        void* outputCUDA = NULL;

        // size_t outputSize = 1 * DIMS_C(tensorDims) * DIMS_H(tensorDims) * DIMS_W(tensorDims) * sizeof(T);
        size_t outputSize = 1;
        size_t array_Len = 1;
        
        for (std::size_t i = 0; i < tensorDims.size(); i++) {
            // access element as v[i]
            int val = tensorDims[i];
            outputSize *= val;
            array_Len *= val;
            // any code including continue, break, return
        }

        outputSize *= sizeof(T);

        if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
        {
            printf("failed to alloc CUDA mapped memory for tensor %s output, %zu bytes\n", name, outputSize);
        }

        // create output tensors
        tensor_name = name;
        CPU  = (T*)outputCPU;
        CUDA = (T*)outputCUDA;
        // fill our array with the values
        std::fill_n(CUDA, array_Len, fill_value);
        dataSize = outputSize;
        dims = tensorDims;
        dims_size = tensorDims.size();
    }
    
    template <class T>
    Tensor<T>::Tensor(const char* name, std::vector<int> tensorDims, T* outputCPU, T* outputCUDA)
    {
        // size_t outputSize = 1 * DIMS_C(tensorDims) * DIMS_H(tensorDims) * DIMS_W(tensorDims) * sizeof(T);
        size_t outputSize = 1;

        for (std::size_t i = 0; i < tensorDims.size(); i++) {
            // access element as v[i]
            int val = tensorDims[i];
            outputSize *= val;
            // any code including continue, break, return
        }

        outputSize *= sizeof(T);


        // create output tensors
        tensor_name = name;
        CPU  = outputCPU;
        CUDA = outputCUDA;
        dataSize = outputSize;
        dims = tensorDims;
        dims_size = tensorDims.size();
    }

    template <class T>
    T* Tensor<T>::data_ptr() {
        return CUDA;
    }

    template <class T>
    int Tensor<T>::size(int d) const {
        return dims[d];
    }
    
    template <class T>
    void Tensor<T>::printDims() {
        printf("Tensor %s ", tensor_name);
        printf("%lu dimensions ", dims.size());
        printf("{");
        for (std::vector<int>::size_type i = 0; i < dims.size(); i++) {
            int val = dims[i];
            printf(" %d ", val);
        }
        printf("}\n");
    }
    
    template <class T>
    T Tensor<T>::retrieve(std::vector<int> indexes) {
        if (indexes.size() == 1) {
            int index = indexes[0];
            return CUDA[index];
        }
        else if (indexes.size() == 2) {
            int index = indexes[1] + (indexes[0] * dims[1]);
            return CUDA[index];
        }
        else if (indexes.size() == 3) {
            int index = indexes[2] + (indexes[1] * dims[2]) + (indexes[0] * dims[1] * dims[2]);
            return CUDA[index];
        }
        else if (indexes.size() == 4) {
            int index = indexes[3] + (indexes[2] * dims[3]) + (indexes[1] * dims[2] * dims[3]) + (indexes[0] * dims[1] * dims[2] * dims[3]);
            return CUDA[index];
        }
        else {
            return CUDA[0];
        }
        // size_t access_dim = indexes.size();
        // int index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3;
        // return CUDA[index];
    }

    // destructor
    template <class T>
    Tensor<T>::~Tensor()
    {
        // CUDA(cudaFreeHost(imgCPU));
    }
}