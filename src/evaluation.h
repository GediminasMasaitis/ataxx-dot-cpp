#ifndef EVALUATION_H
#define EVALUATION_H 1

#include "evaluation_hce.h"
#include "evaluation_nn.h"
#include "evaluation_nn_tuple.h"

#if NNUE
    #if TUPLE_NNUE
    using Evaluation = EvaluationNnueTuple;
    #else
    using Evaluation = EvaluationNnue;
    #endif
#else
using Evaluation = EvaluationHce;
#endif

#endif // !EVALUATION_H
