#include <torch/torch.h>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

using Color = uint8_t;
using Bitboard = uint64_t;
using Wdl = int8_t;
using Score = int16_t;
using Rank = uint8_t;
using File = uint8_t;
using Square = uint8_t;

constexpr auto data_type_val = torch::kFloat32;
using data_type = float_t;
using quantized_type = int16_t;

class TeeBuf : public streambuf
{
    streambuf* first;
    streambuf* second;

public:
    TeeBuf(streambuf* first, streambuf* second) : first(first), second(second)
    {
    }

protected:
    int overflow(int ch) override
    {
        if (ch == char_traits<char>::eof())
        {
            return 0;
        }
        const auto first_result = first->sputc(static_cast<char>(ch));
        const auto second_result = second->sputc(static_cast<char>(ch));
        return (first_result == char_traits<char>::eof() || second_result == char_traits<char>::eof()) ? char_traits<char>::eof() : ch;
    }

    int sync() override
    {
        const auto first_result = first->pubsync();
        const auto second_result = second->pubsync();
        return (first_result == 0 && second_result == 0) ? 0 : -1;
    }
};

struct DataEntry
{
    Bitboard white;
    Bitboard black;
    Color turn;
    Wdl wdl;
    Score score;
};

constexpr int32_t input_size = 49;
constexpr int32_t hidden_size = 256;
constexpr int32_t tuple_count = 36;
constexpr int32_t tuple_states = 81;
constexpr int32_t feature_size = tuple_count * tuple_states;
using InputData = std::array<data_type, input_size>;
using InputDatas = std::array<InputData, 2>;
using OutputData = std::array<data_type, 1>;

struct IDataProvider
{
    virtual ~IDataProvider() {}
    virtual DataEntry get(size_t index) = 0;
    virtual size_t size() const = 0;
};

struct Reader : IDataProvider
{
private:
    constexpr static size_t file_entry_size = sizeof(Bitboard) + sizeof(Bitboard) + sizeof(Color) + sizeof(Wdl) + sizeof(Score);

    ifstream file;
    size_t entry_count;

public:
    Reader(const string& path, const size_t read_limit = 0)
    {
        cout << "Opening " << path << ": ";

        const auto fs_path = filesystem::path(path);
        if (!filesystem::exists(fs_path))
        {
            cerr << path << " doesn't exist";
            throw exception();
        }

        const auto size = filesystem::file_size(fs_path);
        entry_count = size / file_entry_size;

        file = ifstream(path, ios::binary);
        if (!file)
        {
            cerr << "Failed to open file: " << path << endl;
            throw exception();
        }

        cout << size << " bytes, ";
        cout << entry_count << " entries";
        if(read_limit > 0)
        {
            cout << ", limit " << read_limit;
            entry_count = read_limit;
        }
        cout << endl;
    }

    DataEntry get(size_t index) override
    {
        const size_t offset = index * file_entry_size;
        file.seekg(offset);

        DataEntry entry;
        file.read(reinterpret_cast<char*>(&entry.white), sizeof(entry.white));
        file.read(reinterpret_cast<char*>(&entry.black), sizeof(entry.black));
        file.read(reinterpret_cast<char*>(&entry.turn), sizeof(entry.turn));
        file.read(reinterpret_cast<char*>(&entry.wdl), sizeof(entry.wdl));
        file.read(reinterpret_cast<char*>(&entry.score), sizeof(entry.score));

        return entry;
    }

    size_t size() const override
    {
        return entry_count;
    }
};

struct CachingReader : Reader
{
private:
    vector<DataEntry> cache;

public:
    CachingReader(const string& path, const size_t read_limit = 0, size_t cache_limit = 0) : Reader(path, read_limit)
    {
        cache = vector<DataEntry>();
        if(cache_limit == 0)
        {
            cache_limit = read_limit;
        }

        cache_limit = min(read_limit, cache_limit);
        cache.reserve(cache_limit);
        for(size_t index = 0; index < cache_limit; index++)
        {
            const auto entry = Reader::get(index);
            cache.push_back(entry);

            constexpr size_t print_every = 1000000;
            if (index % print_every == print_every - 1 || index == size() - 1)
            {
                cout << "Cached " << index + 1 << " entries" << endl;
            }
        }
    }

    DataEntry get(size_t index) override
    {
        if(index < cache.size())
        {
            return cache[index];
        }

        return Reader::get(index);
    }
};

static constexpr Square get_square(const File file, const Rank rank)
{
    return rank * 8 + file;
}

static constexpr Square get_index(const File file, const Rank rank)
{
    return rank * 7 + file;
}

constexpr Bitboard reverse_bits(Bitboard bitboard)
{
    const Bitboard h1 = 0x5555555555555555;
    const Bitboard h2 = 0x3333333333333333;
    const Bitboard h4 = 0x0F0F0F0F0F0F0F0F;
    const Bitboard v1 = 0x00FF00FF00FF00FF;
    const Bitboard v2 = 0x0000FFFF0000FFFF;
    bitboard = ((bitboard >> 1) & h1) | ((bitboard & h1) << 1);
    bitboard = ((bitboard >> 2) & h2) | ((bitboard & h2) << 2);
    bitboard = ((bitboard >> 4) & h4) | ((bitboard & h4) << 4);
    bitboard = ((bitboard >> 8) & v1) | ((bitboard & v1) << 8);
    bitboard = ((bitboard >> 16) & v2) | ((bitboard & v2) << 16);
    bitboard = (bitboard >> 32) | (bitboard << 32);
    bitboard >>= 9;
    return bitboard;
}

struct TransientDataset : torch::data::Dataset<TransientDataset>
{
private:
    IDataProvider& _provider;

    const torch::TensorOptions tensor_options = torch::TensorOptions().dtype(data_type_val);
    
public:
    TransientDataset(IDataProvider& provider) : _provider(provider)
    {
    }

    torch::data::Example<> get(const size_t index) override
    {
        const auto entry = _provider.get(index);

        InputDatas input_data;
        OutputData output_data;

        const auto is_black = entry.turn == 1;

        const auto us_stm = is_black ? entry.black : entry.white;
        const auto them_stm = is_black ? entry.white : entry.black;

        //const auto us_nstm = is_black ? entry.white : reverse_bits(entry.black);
        //const auto them_nstm = is_black ? entry.black : reverse_bits(entry.white);

        for (Rank rank = 0; rank < 7; rank++)
        {
            for (File file = 0; file < 7; file++)
            {
                const auto square = get_square(file, rank);
                const auto index = get_index(file, rank);

                input_data[0][index] = static_cast<data_type>((us_stm >> square) & 1);
                input_data[1][index] = static_cast<data_type>((them_stm >> square) & 1);
                //input_data[2][index] = static_cast<data_type>((us_nstm >> square) & 1);
                //input_data[3][index] = static_cast<data_type>((them_nstm >> square) & 1);
            }
        }

        const auto wdl = is_black ? static_cast<data_type>(2 - entry.wdl) / 2 : static_cast<data_type>(entry.wdl) / 2;
        output_data[0] = wdl;
        auto input_us_stm = torch::from_blob(input_data[0].data(), { input_size }, tensor_options);
        auto input_them_stm = torch::from_blob(input_data[1].data(), { input_size }, tensor_options);
        //auto input_us_nstm = torch::from_blob(input_data[2].data(), { input_size }, tensor_options);
        //auto input_them_nstm = torch::from_blob(input_data[3].data(), { input_size }, tensor_options);

        auto input = torch::stack({ input_us_stm, input_them_stm }).clone();
        auto target = torch::from_blob(output_data.data(), { 1 }, tensor_options).clone();

        return
        {
            input,
            target
        };
    }

    optional<size_t> size() const override
    {
        return _provider.size();
    }
};

class CachingDataset : public torch::data::Dataset<CachingDataset>
{
public:
    vector<torch::Tensor> inputs;
    vector<torch::Tensor> targets;

    explicit CachingDataset(TransientDataset& dataset)
    {
        assert(dataset.size().has_value());

        const auto size = dataset.size().value();
        inputs.reserve(size);
        targets.reserve(size);

        for (size_t entry_index = 0; entry_index < dataset.size(); entry_index++)
        {
            const auto entry = dataset.get(entry_index);

            inputs.push_back(entry.data);
            targets.push_back(entry.target);

            constexpr size_t print_every = 1000000;
            if (entry_index % print_every == print_every - 1 || entry_index == size - 1)
            {
                cout << "Loaded " << entry_index + 1 << " entries" << endl;
            }
        }
    }

    torch::data::Example<> get(const size_t index) override
    {
        return
        {
            inputs[index],
            targets[index]
        };
    }

    optional<size_t> size() const override
    {
        return inputs.size();
    }
};

struct Net : torch::nn::Module {
    torch::Tensor feature_weights;
    torch::Tensor feature_bias;
    torch::nn::Linear fc2{ nullptr };
    torch::Tensor gather_indices;
    torch::Tensor pow3;
    torch::Tensor tuple_offsets;

    Net() {
        feature_weights = register_parameter("feature_weights", torch::empty({ feature_size, hidden_size }).uniform_(-0.1, 0.1));
        feature_bias = register_parameter("feature_bias", torch::zeros({ hidden_size }));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, 1));

        auto gather = torch::empty({ tuple_count * 4 }, torch::kLong);
        for (auto i = 0; i < 6; i++) {
            for (auto j = 0; j < 6; j++) {
                const auto tuple = i * 6 + j;
                const auto base = i * 7 + j;
                gather[tuple * 4 + 0] = base;
                gather[tuple * 4 + 1] = base + 1;
                gather[tuple * 4 + 2] = base + 7;
                gather[tuple * 4 + 3] = base + 8;
            }
        }
        gather_indices = register_buffer("gather_indices", gather);
        pow3 = register_buffer("pow3", torch::tensor({ 1, 3, 9, 27 }, torch::kLong));
        tuple_offsets = register_buffer("tuple_offsets", torch::arange(tuple_count, torch::kLong) * tuple_states);
    }

    torch::Tensor forward_inner(const torch::Tensor& us, const torch::Tensor& them) {
        const auto batch = us.size(0);
        const auto us_gathered = us.index_select(1, gather_indices).view({ batch, tuple_count, 4 }).to(torch::kLong);
        const auto them_gathered = them.index_select(1, gather_indices).view({ batch, tuple_count, 4 }).to(torch::kLong);
        const auto codes = ((us_gathered + 2 * them_gathered) * pow3).sum(2) + tuple_offsets;
        const auto codes_flat = codes.view({ -1 });
        const auto offsets = torch::arange(0, batch * tuple_count, tuple_count, codes_flat.options());
        const auto hidden = std::get<0>(torch::embedding_bag(feature_weights, codes_flat, offsets)) + feature_bias;
        const auto activated = torch::clamp(hidden, 0.0, 1.0).square();
        return fc2->forward(activated);
    }

    torch::Tensor forward(const torch::Tensor& us, const torch::Tensor& them) {
        return torch::sigmoid(forward_inner(us, them));
    }

    torch::Tensor forward_no_sig(const torch::Tensor& us, const torch::Tensor& them)
    {
        return forward_inner(us, them) * 400;
    }
};

void print_params(const Net& model, const string& file_name, const bool full = true)
{
    const string base_path = "C:/shared/ataxx/nets/current/" + file_name;

    const string float_path = base_path + ".nnue-floats";
    auto file_float = ofstream(float_path, ios::out | ios::binary);

    ofstream file_quantized;
    ofstream file_human;
    if (full)
    {
        file_quantized = ofstream(base_path + ".nnue", ios::out | ios::binary);
        file_human = ofstream(base_path + ".txt", ios::out);
    }

    stringstream ss;
    float max_val = 0;
    for (const auto& pair : model.named_parameters()) {
        auto& name = pair.key();
        const auto param = pair.value().reshape(-1).detach().to(torch::kCPU).contiguous();
        const auto param_accessor = param.accessor<data_type, 1>();
        //cout << name << ": " << pair.value().sizes() << endl;
        //cout << name << ": " << param.sizes() << endl;
        ss << name << endl;
        file_human << name << endl;
        for (auto i = 0; i < param.size(0); i++)
        {
            auto val = param_accessor[i];
            file_float.write(reinterpret_cast<char*>(&val), sizeof(data_type));

            float rounded = round(val * 512);
            if (rounded < -32768)
            {
                cout << "TOO SMALL";
            }
            else if (rounded > 32767)
            {
                cout << "TOO BIG";
            }
            auto rounded_abs = abs(rounded);
            if (rounded_abs > max_val)
            {
                max_val = rounded_abs;
            }

            if (full)
            {
                auto quantized = static_cast<quantized_type>(rounded);
                file_quantized.write(reinterpret_cast<char*>(&quantized), sizeof(quantized_type));

                const auto val_human = to_string(quantized);
                file_human << val_human << " ";
                ss << val_human << " ";
            }
        }
        ss << endl;
        file_human << endl;
    }
    ss << endl;
    cout << "Max val: " << max_val << endl;
    const auto str = ss.str();
    //cout << str;
    file_float.flush();
    if (full)
    {
        file_quantized.flush();
        file_human.flush();
    }
}

void print_time(chrono::time_point<chrono::high_resolution_clock> start)
{
    const auto end = chrono::high_resolution_clock::now();
    const auto milliseconds = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    if (milliseconds < 10000)
    {
        cout << "[" << milliseconds << "ms] ";
        return;
    }
    const auto seconds = chrono::duration_cast<chrono::seconds>(end - start).count();
    cout << "[" << seconds << "s] ";
}

int run_training()
{
    const auto start = chrono::high_resolution_clock::now();

    const auto run_start_time = time(nullptr);
    tm run_start_tm;
    localtime_s(&run_start_tm, &run_start_time);
    char run_stamp[32];
    strftime(run_stamp, sizeof(run_stamp), "%Y%m%d-%H%M%S", &run_start_tm);
    const string run_name = run_stamp;

    static ofstream log_file("C:/shared/ataxx/nets/current/" + run_name + ".log", ios::out);
    static TeeBuf tee_buf(cout.rdbuf(), log_file.rdbuf());
    cout.rdbuf(&tee_buf);

    cout << "Run: " << run_name << endl;
    //auto device = torch::Device(torch::kCPU);
    auto device = torch::Device(torch::kCUDA);
    cout << "CUDA available: " << torch::cuda::is_available() << endl;

    constexpr int32_t batch_size = 1024 * 8;
    //constexpr int32_t batch_size = 1024 * 16;
    //constexpr int32_t batch_size = 128;

    constexpr auto test_path = "C:/shared/ataxx/data/data_test.bin";
    auto test_reader = CachingReader(test_path);
    auto test_transient_set = TransientDataset(test_reader);
    auto test_set = CachingDataset(test_transient_set).map(torch::data::transforms::Stack<>());
    //auto test_set = test_transient_set.map(torch::data::transforms::Stack<>());
    auto test_size = test_set.size().value();
    auto test_loader_options = torch::data::DataLoaderOptions();
    test_loader_options.batch_size(batch_size);
    test_loader_options.workers(2);
    //data_loader_options.enforce_ordering(false);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_set), test_loader_options);
    print_time(start);
    cout << "Loaded test set" << endl;

    constexpr auto train_path = "C:/shared/ataxx/data/Zataxx-550M.bin";
    //constexpr auto train_path = "C:/shared/ataxx/data/data40Mnew.bin";
    //constexpr auto train_path = "C:/shared/ataxx/data/data3M.bin";
    //constexpr auto train_path = "C:/shared/ataxx/data/data_train_small.bin";
    constexpr auto limit = 545'000'000;
    //constexpr auto limit = -1;
    auto train_reader = CachingReader(train_path, limit);
    auto train_transient_set = TransientDataset(train_reader);
    //auto train_set = CachingDataset(train_transient_set).map(torch::data::transforms::Stack<>());
    auto train_set = train_transient_set.map(torch::data::transforms::Stack<>());
    auto train_size = train_set.size().value();
    auto train_loader_options = torch::data::DataLoaderOptions();
    train_loader_options.batch_size(batch_size);
    train_loader_options.workers(12);
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), train_loader_options);
    print_time(start);
    cout << "Loaded training set" << endl;

    auto net = Net();
    net.to(device);

    auto adam_options = torch::optim::AdamOptions(0.001);
    auto optimizer = torch::optim::Adam(net.parameters(), adam_options);
    auto criterion = torch::nn::MSELoss();

    auto static_data = torch::zeros({ batch_size, 2, input_size }, torch::TensorOptions().dtype(data_type_val).device(device));
    auto static_target = torch::zeros({ batch_size, 1 }, torch::TensorOptions().dtype(data_type_val).device(device));
    auto static_loss = torch::zeros({ 1 }, torch::TensorOptions().dtype(data_type_val).device(device));
    auto train_loss_accumulator = torch::zeros({ 1 }, torch::TensorOptions().dtype(data_type_val).device(device));

    const auto train_iteration = [&]()
    {
        optimizer.zero_grad();

        auto us_stm = static_data.select(1, 0);
        auto them_stm = static_data.select(1, 1);

            std::vector<torch::Tensor> us_variants;
            std::vector<torch::Tensor> them_variants;

            us_variants.push_back(us_stm);
            them_variants.push_back(them_stm);
            us_variants.push_back(torch::flip(us_stm, 1));
            them_variants.push_back(torch::flip(them_stm, 1));

            us_stm = us_stm.view({-1, 7, 7}).transpose(1, 2).reshape({-1, 49});
            them_stm = them_stm.view({-1, 7, 7}).transpose(1, 2).reshape({-1, 49});
            us_variants.push_back(us_stm);
            them_variants.push_back(them_stm);
            us_variants.push_back(torch::flip(us_stm, 1));
            them_variants.push_back(torch::flip(them_stm, 1));

            us_stm = us_stm.view({ -1, 7, 7 }).rot90(1, { 1, 2 }).reshape({ -1, 49 });
            them_stm = them_stm.view({ -1, 7, 7 }).rot90(1, { 1, 2 }).reshape({ -1, 49 });
            us_variants.push_back(us_stm);
            them_variants.push_back(them_stm);
            us_variants.push_back(torch::flip(us_stm, 1));
            them_variants.push_back(torch::flip(them_stm, 1));

            us_stm = us_stm.view({ -1, 7, 7 }).transpose(1, 2).reshape({ -1, 49 });
            them_stm = them_stm.view({ -1, 7, 7 }).transpose(1, 2).reshape({ -1, 49 });
            us_variants.push_back(us_stm);
            them_variants.push_back(them_stm);
            us_variants.push_back(torch::flip(us_stm, 1));
            them_variants.push_back(torch::flip(them_stm, 1));

            auto predictions = net.forward(torch::cat(us_variants, 0), torch::cat(them_variants, 0));
            auto targets = static_target.repeat({ 8, 1 });

            auto loss = criterion->forward(predictions, targets);

            loss.backward();
            optimizer.step();

            static_loss.copy_(loss.detach());
            train_loss_accumulator += loss.detach() * static_cast<double>(batch_size);
    };

    at::cuda::CUDAGraph graph;
    const auto capture_stream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
    bool graph_captured = false;

    data_type total_train_loss = 0.0;
    data_type total_test_loss = 0.0;
    for (auto epoch = 0; epoch < 200; epoch++)
    {
        const auto epoch_start = chrono::high_resolution_clock::now();
        print_time(start);
        cout << "Starting epoch " << epoch << endl;
        size_t batch_index = 0;
        for (auto& batch : *train_loader)
        {
            if (batch.data.size(0) != batch_size)
            {
                continue;
            }
            batch_index++;

            static_data.copy_(batch.data);
            static_target.copy_(batch.target);

            if (!graph_captured)
            {
                for (auto warmup = 0; warmup < 3; warmup++)
                {
                    train_iteration();
                }
                torch::cuda::synchronize();
                graph.capture_begin();
                train_iteration();
                graph.capture_end();
                torch::cuda::synchronize();
                graph_captured = true;
            }

            graph.replay();

            if (batch_index % 100 == 0)
            {
                print_time(epoch_start);
                cout << "Batch " << batch_index << " | Loss: " << static_loss.item<data_type>() << endl;
            }
            if (batch_index % 1000 == 0)
            {
                print_params(net, "default-" + run_name + "-" + to_string(epoch) + "-b" + to_string(batch_index), false);
            }
        }
        total_train_loss = train_loss_accumulator.item<data_type>();
        train_loss_accumulator.zero_();

        torch::NoGradGuard no_grad;
        for (auto& batch : *test_loader)
        {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            auto us_stm = data.select(1, 0);
            auto them_stm = data.select(1, 1);
            auto us_nstm = torch::flip(them_stm, 1);
            auto them_nstm = torch::flip(us_stm, 1);

            torch::Tensor prediction = net.forward(us_stm, them_stm);
            auto loss = criterion->forward(prediction, target);

            const auto this_batch_size = batch.data.size(0);
            const auto this_loss = loss.item<data_type>();
            total_test_loss += this_loss * this_batch_size;
        }
        const auto average_train_loss = total_train_loss / train_size;
        const auto average_test_loss = total_test_loss / test_size;
        print_time(epoch_start);
        cout << "Epoch: " << epoch << " | Train loss: " << average_train_loss << " | Test loss: " << average_test_loss << std::endl;
        total_train_loss = 0;
        total_test_loss = 0;
        print_params(net, "default-" + run_name + "-" + to_string(epoch));
    }

    //torch::NoGradGuard no_grad;

    //auto it = test_loader->begin();
    //auto sample_data = it->data.clone();
    //auto sample_target = it->target.reshape(-1);

    //stringstream ss;
    //ss << setprecision(5);

    //ss << "Targets: " << endl;
    //for (auto i = 0; i < sample_data.size(0); i++)
    //{
    //    ss << sample_target[i].item<data_type>() << " ";
    //}
    //ss << endl << endl;

    //const auto predictions = net.forward(sample_data).reshape(-1);
    //const auto predictions_no_sig = net.forward_no_sig(sample_data).reshape(-1);

    //ss << "Predictions: " << endl;
    //for (auto i = 0; i < sample_data.size(0); i++)
    //{
    //    ss << predictions[i].item<data_type>() << " ";
    //}
    //ss << endl << endl;

    //ss << "Predictions no sigmoid: ";
    //for (auto i = 0; i < sample_data.size(0); i++)
    //{
    //    ss << predictions_no_sig[i].item<data_type>() << " ";
    //}
    //ss << endl << endl;
    //auto str = ss.str();
    //cout << str;

    return 0;
}

int main()
{
    try
    {
        return run_training();
    }
    catch (const std::exception& e)
    {
        cout << "FATAL: " << e.what() << endl;
        return 1;
    }
}
