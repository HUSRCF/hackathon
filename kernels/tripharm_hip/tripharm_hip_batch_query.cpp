#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expression)                                                   \
  do {                                                                          \
    const hipError_t status = (expression);                                     \
    if (status != hipSuccess) {                                                 \
      throw std::runtime_error(std::string(#expression) + ": " +               \
                               hipGetErrorString(status));                       \
    }                                                                           \
  } while (false)

namespace {

constexpr char kIndexMagic[12] = {'T', 'P', 'H', 'I', 'P', 'I', 'D', 'X', '1', '\0', '\0', '\0'};
constexpr char kQueryMagic[12] = {'T', 'P', 'H', 'I', 'P', 'B', 'A', 'T', '1', '\0', '\0', '\0'};
constexpr char kOutputMagic[12] = {'T', 'P', 'H', 'I', 'P', 'B', 'O', '1', '\0', '\0', '\0', '\0'};

struct Triangle {
  uint32_t molecule_id;
  uint8_t types[3];
  uint8_t padding;
  float distances[3];
};
static_assert(sizeof(Triangle) == 20, "Python/C++ Triangle ABI must remain 20 bytes");

__host__ __device__ inline float edge(const Triangle &triangle, int left,
                                      int right) {
  if (left > right) {
    const int temporary = left;
    left = right;
    right = temporary;
  }
  if (left == 0 && right == 1) return triangle.distances[0];
  if (left == 0 && right == 2) return triangle.distances[1];
  return triangle.distances[2];
}

__host__ __device__ inline bool matches(const Triangle &query,
                                        const Triangle &candidate,
                                        float tolerance) {
  constexpr int permutations[6][3] = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
      {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
  };
  for (const auto &permutation : permutations) {
    if (query.types[0] != candidate.types[permutation[0]] ||
        query.types[1] != candidate.types[permutation[1]] ||
        query.types[2] != candidate.types[permutation[2]]) continue;
    if (fabsf(edge(query, 0, 1) - edge(candidate, permutation[0], permutation[1])) <= tolerance &&
        fabsf(edge(query, 0, 2) - edge(candidate, permutation[0], permutation[2])) <= tolerance &&
        fabsf(edge(query, 1, 2) - edge(candidate, permutation[1], permutation[2])) <= tolerance) {
      return true;
    }
  }
  return false;
}

__global__ void match_batch(const Triangle *candidates, uint32_t candidate_count,
                            const Triangle *queries, const uint32_t *offsets,
                            uint32_t molecule_count, float tolerance,
                            uint32_t *flags) {
  const uint32_t candidate_index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t batch_index = blockIdx.y;
  if (candidate_index >= candidate_count) return;
  const Triangle candidate = candidates[candidate_index];
  for (uint32_t query = offsets[batch_index]; query < offsets[batch_index + 1]; ++query) {
    if (matches(queries[query], candidate, tolerance)) {
      atomicExch(&flags[static_cast<size_t>(batch_index) * molecule_count +
                          candidate.molecule_id], 1U);
      return;
    }
  }
}

template <typename T> T read_value(std::ifstream &input) {
  T value{};
  input.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!input) throw std::runtime_error("truncated batch input");
  return value;
}
template <typename T> void write_value(std::ofstream &output, const T &value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(value));
  if (!output) throw std::runtime_error("cannot write batch output");
}

struct Options { std::string index; std::string queries; std::string output; };
Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) throw std::invalid_argument("every option requires a value");
    const std::string name = argv[index];
    if (name == "--index") options.index = argv[index + 1];
    else if (name == "--queries") options.queries = argv[index + 1];
    else if (name == "--output") options.output = argv[index + 1];
    else throw std::invalid_argument("unknown option: " + name);
  }
  if (options.index.empty() || options.queries.empty() || options.output.empty())
    throw std::invalid_argument("--index, --queries and --output are required");
  return options;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (std::getenv("HSA_OVERRIDE_GFX_VERSION") != nullptr)
      throw std::runtime_error("HSA_OVERRIDE_GFX_VERSION is forbidden for production queries");
    const Options options = parse_options(argc, argv);
    std::ifstream index_input(options.index, std::ios::binary);
    if (!index_input) throw std::runtime_error("cannot open static index");
    char index_magic[12]{};
    index_input.read(index_magic, sizeof(index_magic));
    if (!index_input || std::string(index_magic, sizeof(index_magic)) !=
                            std::string(kIndexMagic, sizeof(kIndexMagic)))
      throw std::runtime_error("invalid static index magic");
    const uint32_t schema = read_value<uint32_t>(index_input);
    const uint32_t candidate_count = read_value<uint32_t>(index_input);
    const uint32_t molecule_count = read_value<uint32_t>(index_input);
    if (schema != 1 || candidate_count == 0 || molecule_count == 0)
      throw std::runtime_error("invalid static index dimensions");
    std::vector<Triangle> candidates(candidate_count);
    index_input.read(reinterpret_cast<char *>(candidates.data()),
                     candidates.size() * sizeof(Triangle));
    if (!index_input) throw std::runtime_error("truncated static triangle corpus");

    std::ifstream query_input(options.queries, std::ios::binary);
    if (!query_input) throw std::runtime_error("cannot open batch queries");
    char query_magic[12]{};
    query_input.read(query_magic, sizeof(query_magic));
    if (!query_input || std::string(query_magic, sizeof(query_magic)) !=
                            std::string(kQueryMagic, sizeof(kQueryMagic)))
      throw std::runtime_error("invalid batch query magic");
    const uint32_t query_schema = read_value<uint32_t>(query_input);
    const uint32_t batch_count = read_value<uint32_t>(query_input);
    const uint32_t query_count = read_value<uint32_t>(query_input);
    const float tolerance = read_value<float>(query_input);
    if (query_schema != schema || batch_count == 0 || batch_count > 65535 ||
        query_count == 0 || tolerance <= 0.0F)
      throw std::runtime_error("invalid batch query dimensions");
    std::vector<uint32_t> offsets(batch_count + 1);
    query_input.read(reinterpret_cast<char *>(offsets.data()), offsets.size() * sizeof(uint32_t));
    if (!query_input || offsets.front() != 0 || offsets.back() != query_count)
      throw std::runtime_error("invalid batch query offsets");
    for (uint32_t index = 0; index < batch_count; ++index)
      if (offsets[index] >= offsets[index + 1] || offsets[index + 1] - offsets[index] > 64)
        throw std::runtime_error("each batch query must contain 1..64 triangles");
    std::vector<Triangle> queries(query_count);
    query_input.read(reinterpret_cast<char *>(queries.data()), queries.size() * sizeof(Triangle));
    if (!query_input) throw std::runtime_error("truncated batch query triangles");
    for (const Triangle &candidate : candidates)
      if (candidate.molecule_id >= molecule_count)
        throw std::runtime_error("candidate molecule index is out of range");

    int device = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    Triangle *device_candidates = nullptr;
    Triangle *device_queries = nullptr;
    uint32_t *device_offsets = nullptr;
    uint32_t *device_flags = nullptr;
    const size_t flag_count = static_cast<size_t>(batch_count) * molecule_count;
    HIP_CHECK(hipMalloc(&device_candidates, candidates.size() * sizeof(Triangle)));
    HIP_CHECK(hipMalloc(&device_queries, queries.size() * sizeof(Triangle)));
    HIP_CHECK(hipMalloc(&device_offsets, offsets.size() * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&device_flags, flag_count * sizeof(uint32_t)));

    const auto upload_started = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(device_candidates, candidates.data(), candidates.size() * sizeof(Triangle), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_queries, queries.data(), queries.size() * sizeof(Triangle), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_offsets, offsets.data(), offsets.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_flags, 0, flag_count * sizeof(uint32_t)));
    HIP_CHECK(hipDeviceSynchronize());
    const double upload_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - upload_started).count();

    hipEvent_t started{}, stopped{};
    HIP_CHECK(hipEventCreate(&started));
    HIP_CHECK(hipEventCreate(&stopped));
    HIP_CHECK(hipEventRecord(started));
    hipLaunchKernelGGL(match_batch, dim3((candidate_count + 255) / 256, batch_count),
                       dim3(256), 0, 0, device_candidates, candidate_count,
                       device_queries, device_offsets, molecule_count, tolerance, device_flags);
    HIP_CHECK(hipEventRecord(stopped));
    HIP_CHECK(hipEventSynchronize(stopped));
    HIP_CHECK(hipGetLastError());
    float kernel_milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&kernel_milliseconds, started, stopped));

    std::vector<uint32_t> flags(flag_count);
    const auto download_started = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(flags.data(), device_flags, flags.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    const double download_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - download_started).count();

    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create batch output");
    output.write(kOutputMagic, sizeof(kOutputMagic));
    write_value(output, schema);
    write_value(output, molecule_count);
    write_value(output, batch_count);
    output.write(reinterpret_cast<const char *>(flags.data()), flags.size() * sizeof(uint32_t));
    if (!output) throw std::runtime_error("cannot complete batch output");

    const size_t allocated = candidates.size() * sizeof(Triangle) +
        queries.size() * sizeof(Triangle) + offsets.size() * sizeof(uint32_t) +
        flags.size() * sizeof(uint32_t);
    HIP_CHECK(hipEventDestroy(started));
    HIP_CHECK(hipEventDestroy(stopped));
    HIP_CHECK(hipFree(device_candidates));
    HIP_CHECK(hipFree(device_queries));
    HIP_CHECK(hipFree(device_offsets));
    HIP_CHECK(hipFree(device_flags));
    std::cout << std::fixed << std::setprecision(9)
              << "{\"schema_version\":\"1.0\","
              << "\"kind\":\"tripharm-hip-static-resident-batch-prefilter\","
              << "\"architecture\":\"" << properties.gcnArchName << "\","
              << "\"candidate_triangles\":" << candidate_count << ","
              << "\"molecule_count\":" << molecule_count << ","
              << "\"batch_queries\":" << batch_count << ","
              << "\"query_triangles\":" << query_count << ","
              << "\"host_to_device_seconds\":" << upload_seconds << ","
              << "\"kernel_seconds\":" << kernel_milliseconds / 1000.0 << ","
              << "\"device_to_host_seconds\":" << download_seconds << ","
              << "\"allocated_vram_bytes\":" << allocated << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
