#ifndef SEARCHSTATE_H
#define SEARCHSTATE_H 1

#include "types.h"
#include "timer.h"

#include <memory>

#include "options.h"

enum TranspositionTableFlag
{
    Exact,
    Lower,
    Upper
};

struct TranspositionTableEntry
{
    ZobristKey key;
    TranspositionTableFlag flag;
    Score score;
    Ply depth;
    Move move;
};

struct TranspositionTable
{
    std::unique_ptr<TranspositionTableEntry[]> entries;
    size_t entry_count = 0;

    void set_size(const size_t size)
    {
        const auto previous_entry_count = entry_count;
        entry_count = size / sizeof(TranspositionTableEntry);
        if(entry_count != previous_entry_count)
        {
            entries = std::make_unique<TranspositionTableEntry[]>(entry_count);
            clear();
        }
    }

    void set_size_from_options()
    {
        set_size(Options::Hash * 1024 * 1024);
    }

    bool get(const ZobristKey key, TranspositionTableEntry& entry) const
    {
        const size_t index = key % entry_count;
        entry = entries[index];
        return entry.key == key;
    }

    void set(const ZobristKey key, const TranspositionTableFlag flag, const Score score, const Ply depth, const Move& move)
    {
        const size_t index = key % entry_count;
        entries[index] = { key, flag, score, depth, move };
    }

    void clear()
    {
        for (size_t i = 0; i < entry_count; ++i)
        {
            entries[i] = { };
        }
    }
};

struct SearchParameters
{
    bool infinite = false;
    Time white_time = 10000;
    Time white_increment = 100;
    Time black_time = 10000;
    Time black_increment = 100;
    NodeCount nodes_min = 0;
    NodeCount nodes_max = 0;
    bool print_info = true;
    bool print_bestmove = true;
};

struct PrincipalVariationData
{
    Ply length;
    EachPly<Move> moves;
};

struct PlyState
{
    PrincipalVariationData principal_variation;

    void new_search()
    {
        principal_variation.length = 0;
    }

    void clear()
    {
        principal_variation.length = 0;
    }
};

struct ThreadState
{
    SearchParameters parameters;
    Timer timer;
    EachPly<PlyState> plies;
    PrincipalVariationData saved_pv;
    EachSquare<EachSquare<MoveScore>> history;
    uint64_t nodes;

    void new_search(const PositionBase& pos, const SearchParameters& new_parameters)
    {
        parameters = new_parameters;
        const Time time = pos.Turn == Colors::White ? parameters.white_time : parameters.black_time;
        const Time increment = pos.Turn == Colors::White ? parameters.white_increment : parameters.black_increment;
        timer.init(parameters.infinite, parameters.nodes_min, parameters.nodes_max, time, increment);

        for (auto& ply : plies)
        {
            ply.new_search();
        }

        // Age history table
        for (Square from = 0; from < 64; from++)
        {
            for (Square to = 0; to < 64; to++)
            {
                history[from][to] /= 8;
            }
        }

        nodes = 0;
    }

    void clear()
    {
        for (auto& ply : plies)
        {
            ply.clear();
        }
        saved_pv.length = 0;
        
        history = { };
        nodes = 0;
    }
};

struct SearchState
{
    TranspositionTable table;
    std::vector<ThreadState> threads = std::vector<ThreadState>();

    void init_threads_from_options()
    {
        const auto current_threads = threads.size();
        threads.resize(Options::Threads);
        for(auto i = current_threads; i < Options::Threads; i++)
        {
            threads[i].clear();
        }
    }

    void new_search(const PositionBase& pos, const SearchParameters& new_parameters)
    {
        init_threads_from_options();
        for(auto& thread : threads)
        {
            thread.new_search(pos, new_parameters);
        }
    }

    void clear()
    {
        init_threads_from_options();
        table.clear();
        for(auto& thread : threads)
        {
            thread.clear();
        }
    }
};

#endif // !SEARCHSTATE_H
