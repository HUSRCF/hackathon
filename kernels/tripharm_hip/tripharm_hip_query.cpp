#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expression)                                                  \
  do {                                                                         \
    const hipError_t status = (expression);                                    \
    if (status != hipSuccess) {                                                \
      throw std::runtime_error(std::string(#expression) + ": " +              \
                               hipGetErrorString(status));                      \
    }                                                                          \
  } while (false)

namespace {

constexpr uint32_t kMissingError = 0x7f800000U;
constexpr char kInputMagic[8] = {'T', 'P', 'H', 'I', 'P', 'Q', '1', '\0'};
constexpr char kOutputMagic[8] = {'T', 'P', 'H', 'I', 'P', 'O', '1', '\0'};

struct Triangle {
  uint32_t molecule_id;
  uint8_t types[3];
  uint8_t padding;
  float distances[3];
};

static_assert(sizeof(Triangle) == 20, "Python/C++ Triangle ABI must remain 20 bytes");

__host__ __device__ inline float median3(float a, float b, float c) {
  return fmaxf(fminf(a, b), fminf(fmaxf(a, b), c));
}

__host__ __device__ inline float edge(const Triangle &triangle, int left,
                                      int right) {
  if (left > right) {
    const int temporary = left;
    left = right;
    right = temporary;
  }
  if (left == 0 && right == 1) {
    return triangle.distances[0];
  }
  if (left == 0 && right == 2) {
    return triangle.distances[1];
  }
  return triangle.distances[2];
}

__host__ __device__ inline float best_error(const Triangle &query,
                                            const Triangle &candidate,
                                            float tolerance) {
  constexpr int permutations[6][3] = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
      {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
  };
  float best = 2.0F;
  for (const auto &permutation : permutations) {
    if (query.types[0] != candidate.types[permutation[0]] ||
        query.types[1] != candidate.types[permutation[1]] ||
        query.types[2] != candidate.types[permutation[2]]) {
      continue;
    }
    const float e01 =
        fabsf(edge(query, 0, 1) -
              edge(candidate, permutation[0], permutation[1])) /
        tolerance;
    const float e02 =
        fabsf(edge(query, 0, 2) -
              edge(candidate, permutation[0], permutation[2])) /
        tolerance;
    const float e12 =
        fabsf(edge(query, 1, 2) -
              edge(candidate, permutation[1], permutation[2])) /
        tolerance;
    if (e01 <= 1.0F && e02 <= 1.0F && e12 <= 1.0F) {
      best = fminf(best, median3(e01, e02, e12));
    }
  }
  return best;
}

__global__ void initialize_results(unsigned long long *masks,
                                   uint32_t molecule_count, uint32_t *errors,
                                   size_t error_count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < molecule_count) {
    masks[index] = 0;
  }
  if (index < error_count) {
    errors[index] = kMissingError;
  }
}

__global__ void match_triangles(const Triangle *candidates,
                                uint32_t candidate_count,
                                const Triangle *queries, uint32_t query_count,
                                float tolerance, unsigned long long *masks,
                                uint32_t *errors) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const Triangle candidate = candidates[index];
  for (uint32_t query = 0; query < query_count; ++query) {
    const float error = best_error(queries[query], candidate, tolerance);
    if (error <= 1.0F) {
      atomicOr(&masks[candidate.molecule_id], 1ULL << query);
      atomicMin(&errors[candidate.molecule_id * query_count + query],
                __float_as_uint(error));
    }
  }
}

template <typename T> T read_value(std::ifstream &input) {
  T value{};
  input.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!input) {
    throw std::runtime_error("truncated production query input");
  }
  return value;
}

template <typename T> void write_value(std::ofstream &output, const T &value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(value));
  if (!output) {
    throw std::runtime_error("cannot write production query output");
  }
}

struct Options {
  std::string input;
  std::string output;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires a value");
    }
    const std::string name = argv[index];
    if (name == "--input") {
      options.input = argv[index + 1];
    } else if (name == "--output") {
      options.output = argv[index + 1];
    } else {
      throw std::invalid_argument("unknown option: " + name);
    }
  }
  if (options.input.empty() || options.output.empty()) {
    throw std::invalid_argument("--input and --output are required");
  }
  return options;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (std::getenv("HSA_OVERRIDE_GFX_VERSION") != nullptr) {
      throw std::runtime_error(
          "HSA_OVERRIDE_GFX_VERSION is forbidden for production queries");
    }
    const Options options = parse_options(argc, argv);
    std::ifstream input(options.input, std::ios::binary);
    if (!input) {
      throw std::runtime_error("cannot open production query input");
    }
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (!input || std::string(magic, sizeof(magic)) !=
                      std::string(kInputMagic, sizeof(kInputMagic))) {
      throw std::runtime_error("invalid production query input magic");
    }
    const uint32_t schema = read_value<uint32_t>(input);
    const uint32_t candidate_count = read_value<uint32_t>(input);
    const uint32_t molecule_count = read_value<uint32_t>(input);
    const uint32_t query_count = read_value<uint32_t>(input);
    const float tolerance = read_value<float>(input);
    if (schema != 1 || candidate_count == 0 || molecule_count == 0 ||
        query_count == 0 || query_count > 64 || tolerance <= 0.0F) {
      throw std::runtime_error("invalid production query dimensions");
    }
    std::vector<Triangle> candidates(candidate_count);
    std::vector<Triangle> queries(query_count);
    input.read(reinterpret_cast<char *>(candidates.data()),
               candidates.size() * sizeof(Triangle));
    input.read(reinterpret_cast<char *>(queries.data()),
               queries.size() * sizeof(Triangle));
    if (!input) {
      throw std::runtime_error("truncated production triangle arrays");
    }
    for (const Triangle &candidate : candidates) {
      if (candidate.molecule_id >= molecule_count) {
        throw std::runtime_error("candidate molecule index is out of range");
      }
    }

    int device = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    Triangle *device_candidates = nullptr;
    Triangle *device_queries = nullptr;
    unsigned long long *device_masks = nullptr;
    uint32_t *device_errors = nullptr;
    const size_t error_count =
        static_cast<size_t>(molecule_count) * query_count;
    HIP_CHECK(hipMalloc(&device_candidates,
                        candidates.size() * sizeof(Triangle)));
    HIP_CHECK(hipMalloc(&device_queries, queries.size() * sizeof(Triangle)));
    HIP_CHECK(hipMalloc(&device_masks,
                        static_cast<size_t>(molecule_count) *
                            sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&device_errors, error_count * sizeof(uint32_t)));

    const auto upload_started = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(device_candidates, candidates.data(),
                        candidates.size() * sizeof(Triangle),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_queries, queries.data(),
                        queries.size() * sizeof(Triangle),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    const double upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                     upload_started)
            .count();

    const size_t initialization_count =
        std::max(static_cast<size_t>(molecule_count), error_count);
    hipLaunchKernelGGL(initialize_results,
                       dim3((initialization_count + 255) / 256), dim3(256), 0, 0,
                       device_masks, molecule_count, device_errors, error_count);
    hipEvent_t started{};
    hipEvent_t stopped{};
    HIP_CHECK(hipEventCreate(&started));
    HIP_CHECK(hipEventCreate(&stopped));
    HIP_CHECK(hipEventRecord(started));
    hipLaunchKernelGGL(match_triangles, dim3((candidate_count + 255) / 256),
                       dim3(256), 0, 0, device_candidates, candidate_count,
                       device_queries, query_count, tolerance, device_masks,
                       device_errors);
    HIP_CHECK(hipEventRecord(stopped));
    HIP_CHECK(hipEventSynchronize(stopped));
    HIP_CHECK(hipGetLastError());
    float kernel_milliseconds = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&kernel_milliseconds, started, stopped));

    std::vector<unsigned long long> masks(molecule_count);
    std::vector<uint32_t> errors(error_count);
    const auto download_started = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(masks.data(), device_masks,
                        masks.size() * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(errors.data(), device_errors,
                        errors.size() * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    const double download_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                     download_started)
            .count();

    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot create production query output");
    }
    output.write(kOutputMagic, sizeof(kOutputMagic));
    write_value(output, schema);
    write_value(output, molecule_count);
    write_value(output, query_count);
    output.write(reinterpret_cast<const char *>(masks.data()),
                 masks.size() * sizeof(unsigned long long));
    output.write(reinterpret_cast<const char *>(errors.data()),
                 errors.size() * sizeof(uint32_t));
    if (!output) {
      throw std::runtime_error("cannot complete production query output");
    }

    const size_t allocated =
        candidates.size() * sizeof(Triangle) + queries.size() * sizeof(Triangle) +
        masks.size() * sizeof(unsigned long long) +
        errors.size() * sizeof(uint32_t);
    const size_t matched =
        static_cast<size_t>(std::count_if(masks.begin(), masks.end(),
                                         [](auto value) { return value != 0; }));
    HIP_CHECK(hipEventDestroy(started));
    HIP_CHECK(hipEventDestroy(stopped));
    HIP_CHECK(hipFree(device_candidates));
    HIP_CHECK(hipFree(device_queries));
    HIP_CHECK(hipFree(device_masks));
    HIP_CHECK(hipFree(device_errors));

    std::cout << std::fixed << std::setprecision(9)
              << "{\"schema_version\":\"1.0\","
              << "\"kind\":\"tripharm-hip-production-prefilter\","
              << "\"architecture\":\"" << properties.gcnArchName << "\","
              << "\"candidate_triangles\":" << candidate_count << ","
              << "\"molecule_count\":" << molecule_count << ","
              << "\"query_triangles\":" << query_count << ","
              << "\"matched_molecules\":" << matched << ","
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
