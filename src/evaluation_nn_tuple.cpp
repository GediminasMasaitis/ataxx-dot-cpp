#include "evaluation_nn_tuple.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

#if _WIN32
#else
#define ENABLE_INCBIN_TUPLE 1
#endif

#if ENABLE_INCBIN_TUPLE
#include "external/incbin/incbin.h"
INCBIN(tuplenetwork, "networks/default-tuple.nnue-floats");
#endif

// The board is stored 8 squares to a row, so a 2x2 window at (rank, file)
// covers offsets 0, 1, 8 and 9 from its top-left square. Zataxx packs 7 to a
// row and so uses 0, 1, 7, 8 - same window, different stride.
static constexpr Bitboard tuple_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 8) | (1ULL << 9);
static constexpr std::array<int32_t, 4> powers = { 1, 3, 9, 27 };

// Maps a set bit within the masked window (0, 1, 8, 9) to its digit (0..3).
static constexpr int32_t window_digit(const Square square)
{
    return square > 1 ? square - 6 : square;
}

static EvaluationNnueTuple::nnue_param_t read(std::istream& stream, const int32_t quantisation)
{
    constexpr size_t size = sizeof(float);
    char buffer[size];
    stream.read(buffer, size);
    const float* float_ptr = reinterpret_cast<float*>(buffer);
    const auto result_scaled = *float_ptr * static_cast<float>(quantisation);
    return static_cast<EvaluationNnueTuple::nnue_param_t>(round(result_scaled));
}

void EvaluationNnueTuple::init()
{
#if ENABLE_INCBIN_TUPLE
    auto file = stringstream(ios::in | ios::out | ios::binary);
    file.write(reinterpret_cast<const char*>(gtuplenetworkData), gtuplenetworkSize);
    cout << "Using included tuple NNUE" << endl;
#else
    constexpr auto path = "C:/shared/ataxx/AtaxxDotCpp/src/networks/default-tuple.nnue-floats";
    cout << "Reading tuple NNUE from " << path << endl;
    auto file = ifstream(path, ios::binary);
    if (!file.good())
    {
        cout << "Failed to open " << path << endl;
        return;
    }
#endif

    // Input weights are stored input-major: all 256 hidden weights for input 0,
    // then input 1, and so on. That is bullet's native column-major layout for
    // a 256 x 2916 matrix, so the trainer writes it without transposing.
    for (int32_t input_index = 0; input_index < input_size; input_index++)
    {
        for (int32_t hidden_index = 0; hidden_index < hidden_size; hidden_index++)
        {
            input_weights[input_index][hidden_index] = read(file, qa);
        }
    }

    for (int32_t hidden_index = 0; hidden_index < hidden_size; hidden_index++)
    {
        input_biases[hidden_index] = read(file, qa);
    }

    for (int32_t hidden_index = 0; hidden_index < hidden_size; hidden_index++)
    {
        output_weights[hidden_index] = read(file, qb);
    }

    output_bias = read(file, qa * qb);
}

Score EvaluationNnueTuple::evaluate(const PositionBase& pos)
{
    const Bitboard us = pos.Bitboards[pos.Turn];
    const Bitboard them = pos.Bitboards[!pos.Turn];

    hidden_layer_t hidden = input_biases;

    for (int32_t rank = 0; rank < tuple_side; rank++)
    {
        for (int32_t file = 0; file < tuple_side; file++)
        {
            const int32_t offset = rank * 8 + file;
            int32_t index = per_tuple * (rank * tuple_side + file);

            Bitboard our_window = (us >> offset) & tuple_mask;
            while (our_window)
            {
                index += powers[window_digit(pop_lsb(our_window))];
            }

            Bitboard their_window = (them >> offset) & tuple_mask;
            while (their_window)
            {
                index += 2 * powers[window_digit(pop_lsb(their_window))];
            }

            const auto& weights = input_weights[index];
            for (int32_t hidden_index = 0; hidden_index < hidden_size; hidden_index++)
            {
                hidden[hidden_index] += weights[hidden_index];
            }
        }
    }

    // SCReLU. Worst case here is 255*255*127 per term over 256 terms, which
    // stays inside int32; the trainer checks the real bound at save time.
    int32_t sum = 0;
    for (int32_t hidden_index = 0; hidden_index < hidden_size; hidden_index++)
    {
        const int32_t clipped = std::clamp(static_cast<int32_t>(hidden[hidden_index]), 0, qa);
        sum += clipped * clipped * output_weights[hidden_index];
    }

    const int64_t scaled = (static_cast<int64_t>(sum) / qa + output_bias) * scale / (qa * qb);

    // Keep clear of the mate range so a saturated eval can never look like one.
    constexpr int64_t limit = mate - max_ply - 1;
    return static_cast<Score>(std::clamp(scaled, -limit, limit));
}

Score EvaluationNnueTuple::evaluate_from_pov(const PositionBase& pos, const Color color)
{
    const Score score = evaluate(pos);
    return pos.Turn == color ? score : static_cast<Score>(-score);
}
