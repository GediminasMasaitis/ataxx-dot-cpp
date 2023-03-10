#include "datagen.h"

#include "fens.h"
#include "search.h"

#include <iostream>

using namespace std;

void run_iteration(Search& search)
{
    Position pos = Fens::parse(initial_fen);
    SearchParameters parameters;
    search.run(pos, parameters);
}

static void run_thread()
{
    Search search;
    search.state.table.set_size(1024 * 1024 * 32);
    run_iteration(search);
}

static void run_datagen()
{
    run_thread();
}

void Datagen::run()
{
    if constexpr (do_datagen)
    {
        run_datagen();
    }
    else
    {
        cout << "Datagen is not enabled";
    }
}
