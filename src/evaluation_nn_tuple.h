#ifndef EVALUATION_NN_TUPLE_H
#define EVALUATION_NN_TUPLE_H 1

#include "types.h"
#include "position.h"

// The Zataxx architecture: 2916 -> 256 -> 1.
//
// Inputs are 36 overlapping 2x2 windows of the board ("tuples"), one per
// top-left square of a 6x6 grid. Each window's 4 squares are empty / ours /
// theirs, so a window has 3^4 = 81 states and contributes exactly one active
// feature - which makes the input always exactly 36-hot.
//
// Unlike the 98 -> 768 network this is *single perspective*: the features are
// already relative to the side to move (us / them), so there is one 256-wide
// accumulator rather than a stm/ntm pair. Zataxx also recomputes it from
// scratch every evaluation instead of updating incrementally, which is what
// this does - 36 row additions is cheap enough that the accumulator stack,
// and all the make/unmake bookkeeping it needs, can be skipped entirely.
struct EvaluationNnueTuple
{
    static constexpr int32_t tuple_side = 6;
    static constexpr int32_t tuple_count = tuple_side * tuple_side;
    static constexpr int32_t per_tuple = 81;
    static constexpr int32_t input_size = tuple_count * per_tuple;
    static constexpr int32_t hidden_size = 256;

    // Quantisation factors, matching Zataxx. The two layers differ, so unlike
    // the 768 network these cannot share one factor.
    static constexpr int32_t qa = 255;
    static constexpr int32_t qb = 64;

    // Zataxx uses 400. 512 keeps this on the same eval scale the rest of the
    // search is tuned against (RFP, futility, aspiration, datagen cutoffs).
    static constexpr int32_t scale = 512;

    using nnue_param_t = int16_t;
    using hidden_layer_t = std::array<nnue_param_t, hidden_size>;

    inline static std::array<hidden_layer_t, input_size> input_weights;
    inline static hidden_layer_t input_biases;
    inline static hidden_layer_t output_weights;
    inline static nnue_param_t output_bias;

    static void init();
    static Score evaluate(const PositionBase& pos);
    static Score evaluate_from_pov(const PositionBase& pos, Color color);
};

#endif // !EVALUATION_NN_TUPLE_H
