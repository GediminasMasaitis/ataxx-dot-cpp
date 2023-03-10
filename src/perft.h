#ifndef PERFT_H
#define PERFT_H 1

#include "types.h"
#include "position.h"

#include <string>
#include <vector>

struct Perft
{
    static uint64_t perft(const Position& pos, const Ply depth);
    static void perft_deepen(const Position& pos, const Ply depth, std::vector<uint64_t>* verfication = nullptr);
    static void perft_deepen(const Fen& fen, const Ply depth, std::vector<uint64_t>* verfication = nullptr);
    static void perft_suite();
};

#endif // !PERFT_H
