#include "evaluation_hce.h"

#include "attacks.h"

#include <array>

using namespace std;

void EvaluationHce::init()
{
    // Nothing to do
}

Score evaluate_color(const PositionBase& pos, Color color)
{
    Score score = 0;
    auto pieces = pos.Bitboards[color];

    while (pieces)
    {
        const auto sq = pop_lsb(pieces);

        // MATERIAL
        score += 100;

        // UNATTACKABLE
        const Bitboard empty_near_sq = Attacks.near[sq] & pos.Bitboards[Pieces::Empty];
        if (!empty_near_sq)
        {
            score += 50;
        }
    }

    return score;
}

Score EvaluationHce::evaluate(const PositionBase& pos)
{
    Score score = 0;

    // TEMPO
    score += 400;

    score += evaluate_color(pos, pos.Turn);
    score -= evaluate_color(pos, !pos.Turn);
    return score;
}

Score EvaluationHce::evaluate_from_pov(const PositionBase& pos, Color color)
{
    Score score = evaluate(pos);
    if (pos.Turn != color)
    {
        score = static_cast<Score>(-score);
    }
    return score;
}