#include "uai.h"

#include "evaluation_nn_base.h"
#include "evaluation_nn_tuple.h"

#include <iostream>

using namespace std;

int main()
{
    cout << "ataxx.cpp by Gediminas Masaitis" << endl;

    if constexpr (do_tuple_nnue)
    {
        EvaluationNnueTuple::init();
    }
    else
    {
        EvaluationNnueBase::init();
    }

    Uai uai;
    uai.run();

    return 0;
}