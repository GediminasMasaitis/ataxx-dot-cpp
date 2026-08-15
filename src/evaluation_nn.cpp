#include "evaluation_nn.h"

#include "evaluation_nn_base.h"

#include <algorithm>
#include <array>

using namespace std;

Score EvaluationNnue::evaluate(const Position& pos)
{
    const Bitboard us = pos.Bitboards[pos.Turn];
    const Bitboard them = pos.Bitboards[!pos.Turn];

    constexpr array<int32_t, 4> pow3 = { 1, 3, 9, 27 };
    auto accumulator = EvaluationNnueBase::feature_biases;

    for (auto i = 0; i < 6; i++)
    {
        for (auto j = 0; j < 6; j++)
        {
            const auto tuple = i * 6 + j;
            const auto offset = i * 8 + j;
            Bitboard us_bits = (us >> offset) & 0x303;
            Bitboard them_bits = (them >> offset) & 0x303;

            auto code = 0;
            while (us_bits)
            {
                auto sq = pop_lsb(us_bits);
                if (sq > 1)
                {
                    sq -= 6;
                }
                code += pow3[sq];
            }
            while (them_bits)
            {
                auto sq = pop_lsb(them_bits);
                if (sq > 1)
                {
                    sq -= 6;
                }
                code += 2 * pow3[sq];
            }

            const auto& weights = EvaluationNnueBase::feature_weights[tuple * EvaluationNnueBase::tuple_states + code];
            for (auto hidden = 0; hidden < EvaluationNnueBase::hidden_size; hidden++)
            {
                accumulator[hidden] += weights[hidden];
            }
        }
    }

    int64_t sum = 0;
    for (auto hidden = 0; hidden < EvaluationNnueBase::hidden_size; hidden++)
    {
        const int32_t clipped = clamp(static_cast<int32_t>(accumulator[hidden]), 0, EvaluationNnueBase::QA);
        sum += static_cast<int64_t>(clipped) * clipped * EvaluationNnueBase::output_weights[hidden];
    }

    int64_t eval = (sum / EvaluationNnueBase::QA + EvaluationNnueBase::output_bias)
        * EvaluationNnueBase::SCALE / (EvaluationNnueBase::QA * EvaluationNnueBase::QB);
    eval = clamp<int64_t>(eval, -(mate - 300), mate - 300);

    const auto final_score = static_cast<Score>(eval);
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
