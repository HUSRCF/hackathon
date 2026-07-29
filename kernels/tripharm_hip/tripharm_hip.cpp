#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expression)                                                  \
  do {                                                                         \
    const hipError_t status = (expression);                                    \
    if (status != hipSuccess) {                                                 \
      throw std::runtime_error(std::string(#expression) + ": " +              \
                               hipGetErrorString(status));                      \
    }                                                                          \
  } while (false)

namespace {

constexpr uint32_t kMissingError = 0x7f800000U;

struct Triangle {
  uint32_t molecule_id;
  uint8_t types[3];
  uint8_t padding;
  float distances[3]; // d(0,1), d(0,2), d(1,2)
};

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

__global__ void match_triangles(const Triangle *candidates,
                                uint32_t candidate_count,
                                const Triangle *queries,
                                uint32_t query_count, float tolerance,
                                unsigned long long *molecule_masks,
                                uint32_t *molecule_errors) {
  const uint32_t candidate_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate_index >= candidate_count) {
    return;
  }
  const Triangle candidate = candidates[candidate_index];
  for (uint32_t query_index = 0; query_index < query_count; ++query_index) {
    const float error = best_error(queries[query_index], candidate, tolerance);
    if (error <= 1.0F) {
      atomicOr(&molecule_masks[candidate.molecule_id], 1ULL << query_index);
      atomicMin(&molecule_errors[candidate.molecule_id * query_count + query_index],
                __float_as_uint(error));
    }
  }
}

__global__ void initialize_results(unsigned long long *molecule_masks,
                                   uint32_t molecule_count,
                                   uint32_t *molecule_errors,
                                   size_t error_count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < molecule_count) {
    molecule_masks[index] = 0;
  }
  if (index < error_count) {
    molecule_errors[index] = kMissingError;
  }
}

struct Options {
  uint32_t candidates = 100000;
  uint32_t molecules = 100000;
  uint32_t queries = 64;
  uint32_t repetitions = 7;
  uint32_t seed = 20260721;
  float tolerance = 1.0F;
};

uint32_t argument(const char *value, const char *name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) + " must be a positive uint32");
  }
  return static_cast<uint32_t>(parsed);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires a value");
    }
    const std::string name = argv[index];
    if (name == "--candidates") {
      options.candidates = argument(argv[index + 1], "candidates");
    } else if (name == "--molecules") {
      options.molecules = argument(argv[index + 1], "molecules");
    } else if (name == "--queries") {
      options.queries = argument(argv[index + 1], "queries");
    } else if (name == "--repetitions") {
      options.repetitions = argument(argv[index + 1], "repetitions");
    } else if (name == "--seed") {
      options.seed = argument(argv[index + 1], "seed");
    } else {
      throw std::invalid_argument("unknown option: " + name);
    }
  }
  if (options.queries > 64) {
    throw std::invalid_argument("queries must be <= 64 for the fixed match mask");
  }
  if (options.molecules > options.candidates) {
    throw std::invalid_argument("molecules cannot exceed candidate triangles");
  }
  return options;
}

std::vector<Triangle> make_queries(const Options &options) {
  std::mt19937 generator(options.seed);
  std::uniform_int_distribution<int> type(0, 5);
  std::uniform_real_distribution<float> length(2.0F, 8.0F);
  std::vector<Triangle> result(options.queries);
  for (auto &triangle : result) {
    triangle.molecule_id = 0;
    for (auto &value : triangle.types) {
      value = static_cast<uint8_t>(type(generator));
    }
    triangle.distances[0] = length(generator);
    triangle.distances[1] = length(generator);
    const float low = fabsf(triangle.distances[0] - triangle.distances[1]) + 0.1F;
    const float high = triangle.distances[0] + triangle.distances[1] - 0.1F;
    std::uniform_real_distribution<float> third(low, high);
    triangle.distances[2] = third(generator);
  }
  return result;
}

std::vector<Triangle> make_candidates(const Options &options,
                                      const std::vector<Triangle> &queries) {
  std::mt19937 generator(options.seed ^ 0x9e3779b9U);
  std::uniform_int_distribution<int> type(0, 5);
  std::uniform_real_distribution<float> length(2.0F, 12.0F);
  std::uniform_real_distribution<float> jitter(-0.4F, 0.4F);
  std::vector<Triangle> result(options.candidates);
  for (uint32_t index = 0; index < options.candidates; ++index) {
    Triangle &triangle = result[index];
    triangle.molecule_id = index % options.molecules;
    for (auto &value : triangle.types) {
      value = static_cast<uint8_t>(type(generator));
    }
    for (auto &distance : triangle.distances) {
      distance = length(generator);
    }
    // Deterministic positives exercise all query IDs and atom permutations.
    if (index % 11 == 0) {
      const Triangle &query = queries[(index / 11) % queries.size()];
      triangle.types[0] = query.types[2];
      triangle.types[1] = query.types[0];
      triangle.types[2] = query.types[1];
      triangle.distances[0] = query.distances[1] + jitter(generator); // q(2,0)
      triangle.distances[1] = query.distances[2] + jitter(generator); // q(2,1)
      triangle.distances[2] = query.distances[0] + jitter(generator); // q(0,1)
    }
  }
  return result;
}

double percentile(std::vector<float> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double location = (values.size() - 1) * fraction;
  const auto lower = static_cast<size_t>(location);
  const auto upper = std::min(lower + 1, values.size() - 1);
  const double weight = location - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (std::getenv("HSA_OVERRIDE_GFX_VERSION") != nullptr) {
      throw std::runtime_error(
          "HSA_OVERRIDE_GFX_VERSION is forbidden for competition evidence");
    }
    const Options options = parse_options(argc, argv);
    int device = 0;
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    const auto queries = make_queries(options);
    const auto candidates = make_candidates(options, queries);
    const size_t result_count =
        static_cast<size_t>(options.molecules) * options.queries;
    std::vector<unsigned long long> cpu_masks(options.molecules, 0);
    std::vector<uint32_t> cpu_errors(result_count, kMissingError);
    const auto cpu_started = std::chrono::steady_clock::now();
    for (const Triangle &candidate : candidates) {
      for (uint32_t query_index = 0; query_index < options.queries; ++query_index) {
        const float error =
            best_error(queries[query_index], candidate, options.tolerance);
        if (error <= 1.0F) {
          cpu_masks[candidate.molecule_id] |= 1ULL << query_index;
          const uint32_t bits = [&error] {
            uint32_t value;
            std::memcpy(&value, &error, sizeof(value));
            return value;
          }();
          auto &stored = cpu_errors[candidate.molecule_id * options.queries +
                                    query_index];
          stored = std::min(stored, bits);
        }
      }
    }
    const double cpu_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_started)
            .count();

    Triangle *device_candidates = nullptr;
    Triangle *device_queries = nullptr;
    unsigned long long *device_masks = nullptr;
    uint32_t *device_errors = nullptr;
    HIP_CHECK(hipMalloc(&device_candidates, candidates.size() * sizeof(Triangle)));
    HIP_CHECK(hipMalloc(&device_queries, queries.size() * sizeof(Triangle)));
    HIP_CHECK(hipMalloc(&device_masks, cpu_masks.size() * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&device_errors, cpu_errors.size() * sizeof(uint32_t)));
    const auto transfer_started = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(device_candidates, candidates.data(),
                        candidates.size() * sizeof(Triangle), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_queries, queries.data(), queries.size() * sizeof(Triangle),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    const double host_to_device_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - transfer_started)
            .count();

    hipEvent_t start_event{};
    hipEvent_t stop_event{};
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));
    std::vector<float> kernel_milliseconds;
    for (uint32_t repetition = 0; repetition <= options.repetitions; ++repetition) {
      const size_t initialization_count =
          std::max(cpu_masks.size(), cpu_errors.size());
      hipLaunchKernelGGL(initialize_results,
                         dim3((initialization_count + 255) / 256), dim3(256), 0, 0,
                         device_masks, options.molecules, device_errors,
                         cpu_errors.size());
      HIP_CHECK(hipEventRecord(start_event));
      hipLaunchKernelGGL(match_triangles, dim3((options.candidates + 255) / 256),
                         dim3(256), 0, 0, device_candidates, options.candidates,
                         device_queries, options.queries, options.tolerance,
                         device_masks, device_errors);
      HIP_CHECK(hipEventRecord(stop_event));
      HIP_CHECK(hipEventSynchronize(stop_event));
      HIP_CHECK(hipGetLastError());
      float milliseconds = 0.0F;
      HIP_CHECK(hipEventElapsedTime(&milliseconds, start_event, stop_event));
      if (repetition > 0) {
        kernel_milliseconds.push_back(milliseconds);
      }
    }
    std::vector<unsigned long long> gpu_masks(options.molecules);
    std::vector<uint32_t> gpu_errors(result_count);
    const auto return_started = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpy(gpu_masks.data(), device_masks,
                        gpu_masks.size() * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gpu_errors.data(), device_errors,
                        gpu_errors.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    const double device_to_host_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - return_started)
            .count();

    uint64_t cpu_match_count = 0;
    uint64_t intersection_count = 0;
    uint64_t error_mismatches = 0;
    for (uint32_t molecule = 0; molecule < options.molecules; ++molecule) {
      cpu_match_count += __builtin_popcountll(cpu_masks[molecule]);
      intersection_count +=
          __builtin_popcountll(cpu_masks[molecule] & gpu_masks[molecule]);
      for (uint32_t query = 0; query < options.queries; ++query) {
        const size_t index = static_cast<size_t>(molecule) * options.queries + query;
        if (cpu_errors[index] != gpu_errors[index]) {
          ++error_mismatches;
        }
      }
    }
    const bool exact_masks = cpu_masks == gpu_masks;
    const double recall = cpu_match_count == 0
                              ? 1.0
                              : static_cast<double>(intersection_count) /
                                    static_cast<double>(cpu_match_count);
    const size_t peak_vram_bytes = candidates.size() * sizeof(Triangle) +
                                   queries.size() * sizeof(Triangle) +
                                   gpu_masks.size() * sizeof(unsigned long long) +
                                   gpu_errors.size() * sizeof(uint32_t);

    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));
    HIP_CHECK(hipFree(device_candidates));
    HIP_CHECK(hipFree(device_queries));
    HIP_CHECK(hipFree(device_masks));
    HIP_CHECK(hipFree(device_errors));

    std::cout << std::fixed << std::setprecision(9)
              << "{\n"
              << "  \"schema_version\": \"1.0\",\n"
              << "  \"benchmark_scope\": \"triangle-match-microbenchmark\",\n"
              << "  \"architecture\": \"" << properties.gcnArchName << "\",\n"
              << "  \"candidate_triangles\": " << options.candidates << ",\n"
              << "  \"molecules\": " << options.molecules << ",\n"
              << "  \"query_triangles\": " << options.queries << ",\n"
              << "  \"seed\": " << options.seed << ",\n"
              << "  \"cpu_seconds\": " << cpu_seconds << ",\n"
              << "  \"kernel_p50_seconds\": "
              << percentile(kernel_milliseconds, 0.50) / 1000.0 << ",\n"
              << "  \"kernel_p95_seconds\": "
              << percentile(kernel_milliseconds, 0.95) / 1000.0 << ",\n"
              << "  \"host_to_device_seconds\": " << host_to_device_seconds
              << ",\n"
              << "  \"device_to_host_seconds\": " << device_to_host_seconds
              << ",\n"
              << "  \"allocated_vram_bytes\": " << peak_vram_bytes << ",\n"
              << "  \"match_mask_exact\": " << (exact_masks ? "true" : "false")
              << ",\n"
              << "  \"match_recall\": " << recall << ",\n"
              << "  \"float_bit_mismatches\": " << error_mismatches << "\n"
              << "}\n";
    return exact_masks && recall >= 0.999 && error_mismatches == 0 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
