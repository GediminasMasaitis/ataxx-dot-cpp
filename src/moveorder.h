#ifndef MOVEORDER_H
#define MOVEORDER_H 1

#include "types.h"
#include "position.h"
#include "searchstate.h"

struct MoveOrder
{
    static void init();
    static void get_all_policy_scores(const PositionBase& position);
    static void calculate_move_scores(const PositionBase& position, const SearchState& state, const Move& tt_move, const MoveArray& moves, MoveScoreArray& move_scores, const MoveCount move_count);
    static void order_next_move(MoveArray& moves, MoveScoreArray& move_scores, const MoveCount move_count, MoveCount current_index);
};
#endif // !MOVEORDER_H
