#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

// splitmix64 for seeding
static inline uint64_t splitmix64_next(uint64_t &x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// xoshiro256**: fast RNG, good statistical quality for this use.
struct Xoshiro256StarStar {
  uint64_t s[4]{};

  explicit Xoshiro256StarStar(uint64_t seed) {
    uint64_t x = seed;
    s[0] = splitmix64_next(x);
    s[1] = splitmix64_next(x);
    s[2] = splitmix64_next(x);
    s[3] = splitmix64_next(x);
  }

  static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

  inline uint64_t next_u64() {
    const uint64_t result = rotl(s[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
  }

  inline double uniform01() {
    // Convert top 53 bits to double in [0,1)
    const uint64_t x = next_u64();
    return (x >> 11) * (1.0 / 9007199254740992.0); // 2^53
  }

  inline int32_t randint(int32_t n) {
    // n must be > 0
    return static_cast<int32_t>(next_u64() % static_cast<uint64_t>(n));
  }

  inline int64_t randint64(int64_t n) {
    // n must be > 0
    return static_cast<int64_t>(next_u64() % static_cast<uint64_t>(n));
  }

  inline double normal01_box_muller() {
    // One-sample Box-Muller transform: N(0, 1)
    // Avoid u=0 to keep log well-defined.
    double u1 = uniform01();
    if (u1 <= 0.0) {
      u1 = 1e-12;
    }
    const double u2 = uniform01();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 6.28318530717958647692 * u2; // 2*pi
    return r * std::cos(theta);
  }
};

static inline double clip(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

static inline int64_t pos_mod_i64(int64_t x, int64_t m) {
  // Python-style modulo for negatives
  int64_t r = x % m;
  return (r < 0) ? (r + m) : r;
}

static py::array_t<int64_t> randomly_construct_video_np_cpp(int64_t length, int64_t n_views, int64_t n_frames,
                                                            double frame_velo_min, double frame_velo_max,
                                                            double view_velo_min, double view_velo_max,
                                                            double frame_acc_min, double frame_acc_max,
                                                            double view_acc_min, double view_acc_max,
                                                            double frame_velo_buffer_min, double frame_velo_buffer_max,
                                                            double view_velo_buffer_min, double view_velo_buffer_max,
                                                            int64_t acc_update_iter, double drag_coefficient,
                                                            int64_t seed) {
  if (length < 0) {
    throw std::runtime_error("length must be >= 0");
  }
  if (n_views <= 0 || n_frames <= 0) {
    throw std::runtime_error("n_views and n_frames must be > 0");
  }
  if (frame_velo_min > frame_velo_max || view_velo_min > view_velo_max ||
      frame_velo_buffer_min > frame_velo_buffer_max || view_velo_buffer_min > view_velo_buffer_max ||
      frame_acc_min > frame_acc_max || view_acc_min > view_acc_max) {
    throw std::runtime_error("min must be <= max for all ranges");
  }
  if (acc_update_iter <= 0) {
    throw std::runtime_error("acc_update_iter must be > 0");
  }

  uint64_t seed_u64 = 0;
  if (seed >= 0) {
    seed_u64 = static_cast<uint64_t>(seed);
  } else {
    // A decent default seed: time + address mix.
    const uint64_t t = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed_u64 = t ^ (reinterpret_cast<uintptr_t>(&seed_u64) * 0x9E3779B97F4A7C15ULL);
  }
  Xoshiro256StarStar rng(seed_u64);

  py::array_t<int64_t> out({static_cast<py::ssize_t>(length), static_cast<py::ssize_t>(2)});
  auto buf = out.mutable_unchecked<2>();

  int64_t view = 0;
  int64_t frame = 0;
  double view_velo = 0.0;
  double frame_velo = 0.0;
  double view_acc = 0.0;
  double frame_acc = 0.0;

  const double view_acc_mean = 0.5 * (view_acc_min + view_acc_max);
  const double view_acc_std = (view_acc_max - view_acc_min) / 6.0;
  const double frame_acc_mean = 0.5 * (frame_acc_min + frame_acc_max);
  const double frame_acc_std = (frame_acc_max - frame_acc_min) / 6.0;

  const int64_t nv = static_cast<int64_t>(n_views);
  const int64_t nf = static_cast<int64_t>(n_frames);

  if (frame_velo_min < 0.0 && frame_velo_max < 0.0) {
    frame = nf - 1;
  } else if (frame_velo_min < 0.0) {
    frame = rng.randint64(nf);
  } else {
    frame = 0;
  }
  view = rng.randint64(nv);

  {
    py::gil_scoped_release release;
    for (int64_t i = 0; i < length; ++i) {
      buf(i, 0) = view;
      buf(i, 1) = frame;

      // Match python:
      //   if i % acc_update_iter == 0:
      //       view_acc = np.random.normal(mean, std)
      //       frame_acc = np.random.normal(mean, std)
      if ((i % acc_update_iter) == 0) {
        view_acc = view_acc_mean + view_acc_std * rng.normal01_box_muller();
        frame_acc = frame_acc_mean + frame_acc_std * rng.normal01_box_muller();
      }

      view_velo += view_acc;
      frame_velo += frame_acc;

      // Consider drag coefficient
      view_velo *= drag_coefficient;
      frame_velo *= drag_coefficient;

      // Match python:
      //   view_velo = np.clip(view_velo, view_velo_buffer_min, view_velo_buffer_max)
      //   frame_velo = np.clip(frame_velo, frame_velo_buffer_min, frame_velo_buffer_max)
      //   view += np.round(np.clip(view_velo, view_velo_min, view_velo_max)).astype(np.int64)
      //   frame += np.round(np.clip(frame_velo, frame_velo_min, frame_velo_max)).astype(np.int64)
      //
      // np.round uses "banker's rounding" (ties-to-even), which matches std::nearbyint() under FE_TONEAREST.
      view_velo = clip(view_velo, view_velo_buffer_min, view_velo_buffer_max);
      frame_velo = clip(frame_velo, frame_velo_buffer_min, frame_velo_buffer_max);
      const double dv = std::nearbyint(clip(view_velo, view_velo_min, view_velo_max));
      const double df = std::nearbyint(clip(frame_velo, frame_velo_min, frame_velo_max));

      view += static_cast<int64_t>(dv);
      frame += static_cast<int64_t>(df);

      view = pos_mod_i64(view, nv);
      frame = pos_mod_i64(frame, nf);
    }
  }

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("randomly_construct_video_np", &randomly_construct_video_np_cpp,
        py::arg("length"), py::arg("n_views"), py::arg("n_frames"),
        py::arg("frame_velo_min") = 1.0, py::arg("frame_velo_max") = 1.0,
        py::arg("view_velo_min") = -4.0, py::arg("view_velo_max") = 4.0,
        py::arg("frame_acc_min") = -1.0, py::arg("frame_acc_max") = 1.0,
        py::arg("view_acc_min") = -1.0, py::arg("view_acc_max") = 1.0,
        py::arg("frame_velo_buffer_min") = -4.0, py::arg("frame_velo_buffer_max") = 4.0,
        py::arg("view_velo_buffer_min") = -4.0, py::arg("view_velo_buffer_max") = 4.0,
        py::arg("acc_update_iter") = 25,
        py::arg("drag_coefficient") = 0.8,
        py::arg("seed") = -1,
        "Fast C++ implementation. Returns np.ndarray int64 of shape (length, 2) with [view_idx, frame_idx].");
}
