#ifndef EVALUATION_NN_BASE_H
#define EVALUATION_NN_BASE_H 1

#include "types.h"

struct EvaluationNnueBase
{
    using nnue_count_t = int32_t;
    using nnue_param_t = int16_t;

    static constexpr nnue_count_t tuple_count = 36;
    static constexpr nnue_count_t tuple_states = 81;
    static constexpr nnue_count_t feature_size = tuple_count * tuple_states;
    static constexpr nnue_count_t hidden_size = 256;

    static constexpr int32_t QA = 255;
    static constexpr int32_t QB = 64;
    static constexpr int32_t SCALE = 400;

    using hidden_layer_t = std::array<nnue_param_t, hidden_size>;
    using feature_weights_t = std::array<hidden_layer_t, feature_size>;

    inline static feature_weights_t feature_weights;
    inline static hidden_layer_t feature_biases;
    inline static hidden_layer_t output_weights;
    inline static int32_t output_bias;

    static void init();
};

#endif // !EVALUATION_NN_BASE_H
