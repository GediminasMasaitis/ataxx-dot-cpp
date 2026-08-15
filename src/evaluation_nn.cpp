#include "evaluation_nn.h"

#include "evaluation_nn_base.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

using namespace std;

Score EvaluationNnue::evaluate(const Position& pos)
{
    // Both layers are quantised by this factor when the network is read.
    constexpr int32_t quantisation = 128;

    // Deliberately int32: widening this to int64 costs roughly 3x the nps.
    // The trainer checks at save time that the weights leave headroom here.
    int32_t activations = 0;

    for (auto c = 0; c < 2; c++)
    {
        const auto& accumulator = pos.accumulators_stack[pos.accumulator_index][pos.Turn ^ c];
        const auto& weights = EvaluationNnueBase::hidden_weightses[c];
        for (auto i = 0; i < EvaluationNnueBase::hidden_size; i++)
        {
            if constexpr (do_screlu)
            {
                // Clamp to [0, 1] and square. Quantised that is [0, quantisation],
                // and squaring adds one factor of quantisation to the product.
                const int32_t clipped = std::clamp(static_cast<int32_t>(accumulator[i]), 0, quantisation);
                activations += clipped * clipped * weights[i];
            }
            else
            {
                activations += std::max(static_cast<int32_t>(accumulator[i]), 0) * weights[i];
            }
        }
    }

    int32_t score = EvaluationNnueBase::hidden_bias * quantisation;
    if constexpr (do_screlu)
    {
        // Take back the extra quantisation factor that squaring introduced.
        score += activations / quantisation;
    }
    else
    {
        score += activations;
    }

    score /= 32;

    const auto final_score = static_cast<Score>(score);
    return final_score;
}

Score EvaluationNnue::evaluate_from_pov(const Position& pos, Color color)
{
    Score score = evaluate(pos);
    if (pos.Turn != color)
    {
        score = static_cast<Score>(-score);
    }
    return score;
}
