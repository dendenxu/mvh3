#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

struct Matrix {
  int rows = 0;
  int cols = 0;
  std::vector<double> data; // row-major
};

using Value = std::variant<std::monostate, double, std::vector<std::string>, Matrix>;

struct ParsedFile {
  std::unordered_map<std::string, Value> kv;
};

static inline bool starts_with(const std::string &s, const std::string &prefix) { return s.rfind(prefix, 0) == 0; }

static inline std::string ltrim(std::string s) {
  size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
  s.erase(0, i);
  return s;
}

static inline std::string rtrim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.pop_back();
  return s;
}

static inline std::string trim(std::string s) { return rtrim(ltrim(std::move(s))); }

static bool try_parse_double(const std::string &s, double *out) {
  std::string t = trim(s);
  if (t.empty())
    return false;
  char *end = nullptr;
  const char *c = t.c_str();
  double v = std::strtod(c, &end);
  if (end == c)
    return false;
  while (*end != '\0') {
    if (!std::isspace(static_cast<unsigned char>(*end)))
      return false;
    ++end;
  }
  *out = v;
  return true;
}

static std::vector<double> parse_number_list(const std::string &s) {
  std::string t = trim(s);
  size_t lb = t.find('[');
  size_t rb = t.rfind(']');
  if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
    t = t.substr(lb + 1, rb - lb - 1);
  }
  std::vector<double> out;
  std::string token;
  std::stringstream ss(t);
  while (std::getline(ss, token, ',')) {
    double v = 0.0;
    if (try_parse_double(token, &v))
      out.push_back(v);
  }
  return out;
}

static ParsedFile parse_opencv_yaml(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  ParsedFile pf;
  std::vector<std::string> lines;
  {
    std::string line;
    while (std::getline(ifs, line)) {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      lines.push_back(line);
    }
  }

  for (size_t i = 0; i < lines.size();) {
    std::string line = trim(lines[i]);
    ++i;
    if (line.empty())
      continue;
    if (starts_with(line, "%YAML") || line == "---")
      continue;

    auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    std::string key = trim(line.substr(0, colon));
    std::string rest = trim(line.substr(colon + 1));
    if (key.empty())
      continue;

    if (rest == "!!opencv-matrix") {
      int rows = 0, cols = 0;
      std::vector<double> data;
      for (int k = 0; k < 4 && i < lines.size(); ++k, ++i) {
        std::string l = trim(lines[i]);
        auto c = l.find(':');
        if (c == std::string::npos)
          continue;
        std::string kname = trim(l.substr(0, c));
        std::string kval = trim(l.substr(c + 1));
        if (kname == "rows") {
          double v = 0.0;
          if (try_parse_double(kval, &v))
            rows = static_cast<int>(v);
        } else if (kname == "cols") {
          double v = 0.0;
          if (try_parse_double(kval, &v))
            cols = static_cast<int>(v);
        } else if (kname == "data") {
          data = parse_number_list(kval);
        }
      }
      Matrix m;
      m.rows = rows;
      m.cols = cols;
      m.data = std::move(data);
      pf.kv.emplace(std::move(key), std::move(m));
      continue;
    }

    if (rest.empty()) {
      std::vector<std::string> elems;
      while (i < lines.size()) {
        std::string raw = lines[i];
        std::string traw = ltrim(raw);
        if (!starts_with(traw, "-") && !starts_with(raw, "  -"))
          break;
        std::string li = trim(raw);
        if (!starts_with(li, "-"))
          break;
        li = trim(li.substr(1));
        if (li.size() >= 2 && ((li.front() == '"' && li.back() == '"') || (li.front() == '\'' && li.back() == '\''))) {
          li = li.substr(1, li.size() - 2);
        }
        if (!li.empty() && li != "none")
          elems.push_back(li);
        ++i;
      }
      pf.kv.emplace(std::move(key), std::move(elems));
      continue;
    }

    double v = 0.0;
    if (try_parse_double(rest, &v)) {
      pf.kv.emplace(std::move(key), v);
    } else {
      pf.kv.emplace(std::move(key), std::monostate{});
    }
  }

  return pf;
}

static bool get_list(const ParsedFile &pf, const std::string &key, std::vector<std::string> *out) {
  auto it = pf.kv.find(key);
  if (it == pf.kv.end())
    return false;
  if (auto p = std::get_if<std::vector<std::string>>(&it->second)) {
    *out = *p;
    return true;
  }
  return false;
}

static bool get_matrix(const ParsedFile &pf, const std::string &key, Matrix *out) {
  auto it = pf.kv.find(key);
  if (it == pf.kv.end())
    return false;
  if (auto p = std::get_if<Matrix>(&it->second)) {
    *out = *p;
    return true;
  }
  return false;
}

static py::array_t<float> to_numpy(const Matrix &m) {
  py::array_t<float> arr({m.rows, m.cols});
  auto buf = arr.mutable_unchecked<2>();
  for (int r = 0; r < m.rows; ++r) {
    for (int c = 0; c < m.cols; ++c) {
      buf(r, c) = m.data[static_cast<size_t>(r * m.cols + c)];
    }
  }
  return arr;
}

static py::dict read_camera_cpp_minimal(const std::string &intri_path, const std::string &extri_path) {
  py::gil_scoped_release release;

  ParsedFile intri = parse_opencv_yaml(intri_path);
  ParsedFile extri = parse_opencv_yaml(extri_path);

  std::vector<std::string> cam_names;
  if (!get_list(intri, "names", &cam_names)) {
    cam_names.clear();
  }

  py::gil_scoped_acquire acquire;
  py::dict cams;

  for (const auto &cam : cam_names) {
    py::dict c;

    Matrix K;
    if (!get_matrix(intri, "K_" + cam, &K)) {
      throw std::runtime_error("Missing K_" + cam);
    }

    Matrix Tvec;
    if (!get_matrix(extri, "T_" + cam, &Tvec)) {
      throw std::runtime_error("Missing T_" + cam);
    }
    Matrix R;
    if (!get_matrix(extri, "Rot_" + cam, &R)) {
      throw std::runtime_error("Missing R_" + cam);
    }

    c["K"] = to_numpy(K);
    c["R"] = to_numpy(R);
    c["T"] = to_numpy(Tvec);
    cams[py::str(cam)] = c;
  }

  return cams;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read_camera_minimal", &read_camera_cpp_minimal, "Read intri/extri OpenCV-YAML cameras (GIL released)");
}
