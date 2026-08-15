#include "evaluation_nn_base.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

#if _WIN32
#else
#define ENABLE_INCBIN 1
#endif

#if ENABLE_INCBIN
#include "external/incbin/incbin.h"
INCBIN(network, "networks/default.nnue-floats");
#endif

static float read_float(std::istream& stream)
{
    char buffer[sizeof(float)];
    stream.read(buffer, sizeof(float));
    float value;
    memcpy(&value, buffer, sizeof(float));
    return value;
}

static EvaluationNnueBase::nnue_param_t read_quantized(std::istream& stream, const int32_t scale)
{
    const auto value = read_float(stream);
    const auto rounded = round(value * scale);
    return static_cast<EvaluationNnueBase::nnue_param_t>(rounded);
}

void EvaluationNnueBase::init()
{
    constexpr size_t float_count =
        static_cast<size_t>(feature_size) * hidden_size
        + hidden_size
        + hidden_size
        + 1;
    constexpr size_t expected_size = float_count * sizeof(float);

#if ENABLE_INCBIN
    if (gnetworkSize != expected_size)
    {
        cout << "WARNING: embedded NNUE is " << gnetworkSize << " bytes, expected " << expected_size
             << " - train a tuple network first, evaluation is garbage until then" << endl;
        return;
    }
    auto file = stringstream(ios::in | ios::out | ios::binary);
    file.write(reinterpret_cast<const char*>(gnetworkData), gnetworkSize);
    cout << "Using included NNUE" << endl;
#else
    #ifdef _WIN32
        constexpr auto path = "C:/shared/ataxx/AtaxxDotCpp/src/networks/default.nnue-floats";
    #else
        constexpr auto path = "/mnt/c/shared/ataxx/nets/default.nnue-floats";
    #endif
    cout << "Reading NNUE from " << path << endl;
    auto file = ifstream(path, ios::binary | ios::ate);
    if (!file.good())
    {
        cout << "Failed to open " << path << endl;
        return;
    }
    const auto file_size = static_cast<size_t>(file.tellg());
    if (file_size != expected_size)
    {
        cout << "WARNING: NNUE file is " << file_size << " bytes, expected " << expected_size
             << " - train a tuple network first, evaluation is garbage until then" << endl;
        return;
    }
    file.seekg(0);
#endif

    for (nnue_count_t feature = 0; feature < feature_size; feature++)
    {
        for (nnue_count_t hidden = 0; hidden < hidden_size; hidden++)
        {
            feature_weights[feature][hidden] = read_quantized(file, QA);
        }
    }

    for (nnue_count_t hidden = 0; hidden < hidden_size; hidden++)
    {
        feature_biases[hidden] = read_quantized(file, QA);
    }

    for (nnue_count_t hidden = 0; hidden < hidden_size; hidden++)
    {
        output_weights[hidden] = read_quantized(file, QB);
    }

    output_bias = static_cast<int32_t>(round(read_float(file) * QA * QB));
}
