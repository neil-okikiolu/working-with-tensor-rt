#ifndef __JETSONCAM_PARSE_OBJECTS_H__
#define __JETSONCAM_PARSE_OBJECTS_H__

#include "Tensor.h"

#include <vector>
#include <stdint.h>


namespace trt_pose {
    
    using namespace jetsoncam;
    
    class ParseObjects
    {
        public:
            ParseObjects();

            ParseObjects(
                Tensor<int> top
            );

            ParseResult Parse(Tensor<float> cmap, Tensor<float> paf);

        protected:
            Tensor<int> topology;
    };
}

#endif