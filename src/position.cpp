#include "position.h"

#include "attacks.h"
#include "zobrist.h"

#include <cassert>

bool PositionBase::is_terminal() const
{
    if(Bitboards[Turn] == 0)
    {
        return true;
    }

    if((Bitboards[Pieces::White] | Bitboards[Pieces::Black]) == playable)
    {
        return true;
    }

    return false;
}

bool PositionBase::verify_bitboards() const
{
    for(Rank rank = 0; rank < 7; rank++)
    {
        for (File file = 0; file < 7; file++)
        {
            const auto sq = get_square(file, rank);
            const auto square_bb = get_bitboard(sq);
            for (Piece piece = Pieces::White; piece < Pieces::Count; piece++)
            {
                const auto is_set = Squares[sq] == piece;
                if (is_set && !(Bitboards[piece] & square_bb))
                {
                    assert(false);
                    return false;
                }
                if (!is_set && (Bitboards[piece] & square_bb))
                {
                    assert(false);
                    return false;
                }
            }
        }
    }

    return true;
}

PositionBase PositionBase::make_move_copy(const Move& move) const
{
    assert(move != no_move);
    assert(move.Turn == Turn);

    PositionBase new_pos = *this;
    new_pos.Turn = !new_pos.Turn;
    new_pos.Key ^= Zobrist.turn;

    if (move == passes[move.Turn])
    {
        assert(new_pos.verify_bitboards());
        return new_pos;
    }

    // TO
    assert(move.To != no_square);
    assert(new_pos.Squares[move.To] == Pieces::Empty);
    new_pos.Squares[move.To] = move.Turn;
    new_pos.Bitboards[move.Turn] |= get_bitboard(move.To);
    new_pos.Bitboards[Pieces::Empty] &= ~get_bitboard(move.To);
    new_pos.Key ^= Zobrist.squares[move.Turn][move.To];

    // FROM
    if (move.From != no_square)
    {
        assert(move.From != move.To);
        assert(new_pos.Squares[move.From] == move.Turn);
        new_pos.Squares[move.From] = Pieces::Empty;
        new_pos.Bitboards[move.Turn] &= ~get_bitboard(move.From);
        new_pos.Bitboards[Pieces::Empty] |= get_bitboard(move.From);
        new_pos.Key ^= Zobrist.squares[move.Turn][move.From];
    }

    // CAPTURE
    auto attacked = Attacks.near[move.To] & new_pos.Bitboards[!move.Turn];
    new_pos.Bitboards[move.Turn] |= attacked;
    new_pos.Bitboards[!move.Turn] &= ~attacked;
    while (attacked)
    {
        const auto attacked_square = pop_lsb(attacked);
        new_pos.Squares[attacked_square] = move.Turn;
        new_pos.Key ^= Zobrist.squares[move.Turn][attacked_square];
        new_pos.Key ^= Zobrist.squares[!move.Turn][attacked_square];
    }

    assert(new_pos.verify_bitboards());
    return new_pos;
}

PositionBase PositionBase::make_move_copy(const MoveStr& move_str) const
{
    const Move move = Move::from_move_str(Turn, move_str);
    return make_move_copy(move);
}

void Position::accumulators_push()
{
    if constexpr (do_nnue)
    {
        if (!enable_accumulator_stack)
        {
            return;
        }

        accumulator_index++;
        accumulators_stack[accumulator_index] = accumulators_stack[accumulator_index - 1];
    }
}

void Position::accumulators_pop()
{
    if constexpr (!do_nnue)
    {
        return;
    }

    if (!enable_accumulator_stack)
    {
        return;
    }

    accumulator_index--;
}

void Position::accumulators_set(const Square sq, const Piece piece)
{
    if constexpr (!do_nnue)
    {
        return;
    }

    EvaluationNnueBase::apply_piece<true>(accumulators_stack[accumulator_index], sq, piece);
}

void Position::accumulators_unset(const Square sq, const Piece piece)
{
    if constexpr (!do_nnue)
    {
        return;
    }

    EvaluationNnueBase::apply_piece<false>(accumulators_stack[accumulator_index], sq, piece);
}

void Position::reset_accumulators()
{
    if constexpr (!do_nnue)
    {
        return;
    }

    accumulator_index = 0;
    auto& accumulators = accumulators_stack[accumulator_index];
    accumulators[Colors::White] = EvaluationNnueBase::input_biases;
    accumulators[Colors::Black] = EvaluationNnueBase::input_biases;

    for (Rank rank = 0; rank < 7; rank++)
    {
        for (File file = 0; file < 7; file++)
        {
            const auto sq = get_square(file, rank);
            const auto piece = Squares[sq];

            if (piece == Pieces::White)
            {
                accumulators_set(sq, piece);
            }
            else if (piece == Pieces::Black)
            {
                accumulators_set(sq, piece);
            }
        }
    }
}

void Position::make_move_in_place(const Move& move)
{
    assert(move != no_move);
    assert(move.Turn == Turn);

    assert(HistoryCount < max_history_count);
    auto attacked = Attacks.near[move.To] & Bitboards[!move.Turn];
    History[HistoryCount] = { Key, move, attacked };
    HistoryCount++;

    Turn = !Turn;
    Key ^= Zobrist.turn;

    if(move == passes[move.Turn])
    {
        assert(verify_bitboards());
        return;
    }

    accumulators_push();

    // TO
    assert(move.To != no_square);
    assert(Squares[move.To] == Pieces::Empty);
    Squares[move.To] = move.Turn;
    Bitboards[move.Turn] |= get_bitboard(move.To);
    Bitboards[Pieces::Empty] &= ~get_bitboard(move.To);
    accumulators_set(move.To, move.Turn);
    Key ^= Zobrist.squares[move.Turn][move.To];

    // FROM
    if(move.From != no_square)
    {
        assert(move.From != move.To);
        assert(Squares[move.From] == move.Turn);
        Squares[move.From] = Pieces::Empty;
        Bitboards[move.Turn] &= ~get_bitboard(move.From);
        Bitboards[Pieces::Empty] |= get_bitboard(move.From);
        accumulators_unset(move.From, move.Turn);
        Key ^= Zobrist.squares[move.Turn][move.From];
    }

    // CAPTURE
    Bitboards[move.Turn] |= attacked;
    Bitboards[!move.Turn] &= ~attacked;
    while (attacked)
    {
        const auto attacked_square = pop_lsb(attacked);
        Squares[attacked_square] = move.Turn;
        accumulators_unset(attacked_square, !move.Turn);
        accumulators_set(attacked_square, move.Turn);
        Key ^= Zobrist.squares[move.Turn][attacked_square];
        Key ^= Zobrist.squares[!move.Turn][attacked_square];
    }
    assert(verify_bitboards());
}

void Position::make_move_in_place(const MoveStr& move_str)
{
    const Move move = Move::from_move_str(Turn, move_str);
    make_move_in_place(move);
}

void Position::unmake_move()
{
    HistoryCount--;
    UndoData& undo_data = History[HistoryCount];

    const Move& move = undo_data.move;
    assert(move != no_move);

    Key = undo_data.key;
    Turn = !Turn;
    assert(move.Turn == Turn);
    if(move == passes[move.Turn])
    {
        assert(verify_bitboards());
        return;
    }

    // TO
    assert(move.To != no_square);
    assert(Squares[move.To] == move.Turn);
    Squares[move.To] = Pieces::Empty;
    Bitboards[move.Turn] &= ~get_bitboard(move.To);
    Bitboards[Pieces::Empty] |= get_bitboard(move.To);
    //accumulators_unset(move.To, move.Turn);

    // FROM
    if(undo_data.move.From != no_square)
    {
        assert(move.From != move.To);
        assert(Squares[move.From] == Pieces::Empty);
        Squares[move.From] = Turn;
        Bitboards[Turn] |= get_bitboard(move.From);
        Bitboards[Pieces::Empty] &= ~get_bitboard(move.From);
        //accumulators_set(move.From, move.Turn);
    }

    // CAPTURE
    Bitboards[move.Turn] &= ~undo_data.captured;
    Bitboards[!move.Turn] |= undo_data.captured;
    while (undo_data.captured)
    {
        const auto attacked_square = pop_lsb(undo_data.captured);
        Squares[attacked_square] = !move.Turn;
        //accumulators_unset(attacked_square, move.Turn);
        //accumulators_set(attacked_square, !move.Turn);
    }
    accumulators_pop();
    assert(verify_bitboards());
}
