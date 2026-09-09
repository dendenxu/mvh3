#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
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

static bool get_scalar(const ParsedFile &pf, const std::string &key, double *out) {
  auto it = pf.kv.find(key);
  if (it == pf.kv.end())
    return false;
  if (auto p = std::get_if<double>(&it->second)) {
    *out = *p;
    return true;
  }
  return false;
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

static Matrix mat_identity3() {
  Matrix m;
  m.rows = 3;
  m.cols = 3;
  m.data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

static Matrix matmul(const Matrix &A, const Matrix &B) {
  if (A.cols != B.rows) {
    throw std::runtime_error("matmul shape mismatch");
  }
  Matrix C;
  C.rows = A.rows;
  C.cols = B.cols;
  C.data.assign(static_cast<size_t>(C.rows * C.cols), 0.0);
  for (int i = 0; i < A.rows; ++i) {
    for (int k = 0; k < A.cols; ++k) {
      double a = A.data[static_cast<size_t>(i * A.cols + k)];
      for (int j = 0; j < B.cols; ++j) {
        C.data[static_cast<size_t>(i * C.cols + j)] += a * B.data[static_cast<size_t>(k * B.cols + j)];
      }
    }
  }
  return C;
}

static Matrix hstack_R_T(const Matrix &R, const Matrix &T) {
  if (R.rows != 3 || R.cols != 3)
    throw std::runtime_error("R must be 3x3");
  if (!((T.rows == 3 && T.cols == 1) || (T.rows == 1 && T.cols == 3)))
    throw std::runtime_error("T must be 3x1 or 1x3");
  Matrix TT = T;
  if (TT.rows == 1 && TT.cols == 3) {
    TT.rows = 3;
    TT.cols = 1;
    TT.data = {T.data[0], T.data[1], T.data[2]};
  }
  Matrix RT;
  RT.rows = 3;
  RT.cols = 4;
  RT.data.assign(12, 0.0);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      RT.data[static_cast<size_t>(r * 4 + c)] = R.data[static_cast<size_t>(r * 3 + c)];
    }
    RT.data[static_cast<size_t>(r * 4 + 3)] = TT.data[static_cast<size_t>(r)];
  }
  return RT;
}

static Matrix transpose(const Matrix &A) {
  Matrix AT;
  AT.rows = A.cols;
  AT.cols = A.rows;
  AT.data.assign(static_cast<size_t>(AT.rows * AT.cols), 0.0);
  for (int r = 0; r < A.rows; ++r) {
    for (int c = 0; c < A.cols; ++c) {
      AT.data[static_cast<size_t>(c * AT.cols + r)] = A.data[static_cast<size_t>(r * A.cols + c)];
    }
  }
  return AT;
}

static Matrix inv3x3(const Matrix &K) {
  if (K.rows != 3 || K.cols != 3)
    throw std::runtime_error("inv3x3 expects 3x3");
  const double a = K.data[0], b = K.data[1], c = K.data[2];
  const double d = K.data[3], e = K.data[4], f = K.data[5];
  const double g = K.data[6], h = K.data[7], i = K.data[8];
  const double A = (e * i - f * h);
  const double B = -(d * i - f * g);
  const double C = (d * h - e * g);
  const double D = -(b * i - c * h);
  const double E = (a * i - c * g);
  const double F = -(a * h - b * g);
  const double G = (b * f - c * e);
  const double H = -(a * f - c * d);
  const double I = (a * e - b * d);
  const double det = a * A + b * B + c * C;
  if (std::abs(det) < 1e-18)
    throw std::runtime_error("Singular 3x3 matrix");
  const double invdet = 1.0 / det;
  Matrix inv;
  inv.rows = 3;
  inv.cols = 3;
  inv.data = {A * invdet, D * invdet, G * invdet, B * invdet, E * invdet, H * invdet, C * invdet, F * invdet,
              I * invdet};
  return inv;
}

static Matrix normalize_rvec(const Matrix &rvec) {
  Matrix rv = rvec;
  if (rv.rows == 1 && rv.cols == 3) {
    rv.rows = 3;
    rv.cols = 1;
  }
  if (rv.rows != 3 || rv.cols != 1)
    throw std::runtime_error("rvec must be 3x1 or 1x3");
  return rv;
}

static Matrix rodrigues_rvec_to_R(const Matrix &rvec_in) {
  Matrix rvec = normalize_rvec(rvec_in);
  const double rx = rvec.data[0], ry = rvec.data[1], rz = rvec.data[2];
  const double theta = std::sqrt(rx * rx + ry * ry + rz * rz);

  Matrix R = mat_identity3();
  if (theta < 1e-12) {
    const double kx = rx, ky = ry, kz = rz;
    R.data = {1, -kz, ky, kz, 1, -kx, -ky, kx, 1};
    return R;
  }

  const double ux = rx / theta, uy = ry / theta, uz = rz / theta;
  const double ct = std::cos(theta);
  const double st = std::sin(theta);
  const double vt = 1.0 - ct;

  R.data = {ct + ux * ux * vt, ux * uy * vt - uz * st, ux * uz * vt + uy * st, uy * ux * vt + uz * st,
            ct + uy * uy * vt, uy * uz * vt - ux * st, uz * ux * vt - uy * st, uz * uy * vt + ux * st,
            ct + uz * uz * vt};
  return R;
}

static Matrix rodrigues_R_to_rvec(const Matrix &R) {
  if (R.rows != 3 || R.cols != 3)
    throw std::runtime_error("R must be 3x3");

  const double r00 = R.data[0], r01 = R.data[1], r02 = R.data[2];
  const double r10 = R.data[3], r11 = R.data[4], r12 = R.data[5];
  const double r20 = R.data[6], r21 = R.data[7], r22 = R.data[8];

  const double tr = r00 + r11 + r22;
  double c = (tr - 1.0) * 0.5;
  c = std::max(-1.0, std::min(1.0, c));

  const double rx = r21 - r12;
  const double ry = r02 - r20;
  const double rz = r10 - r01;
  const double s = 0.5 * std::sqrt(std::max(0.0, rx * rx + ry * ry + rz * rz));

  const double theta = std::atan2(s, c);

  Matrix rvec;
  rvec.rows = 3;
  rvec.cols = 1;
  rvec.data.assign(3, 0.0);

  if (s < 1e-8) {
    if (c > 0.0) {
      rvec.data[0] = 0.5 * rx;
      rvec.data[1] = 0.5 * ry;
      rvec.data[2] = 0.5 * rz;
      return rvec;
    }

    const double xx = (r00 + 1.0) * 0.5;
    const double yy = (r11 + 1.0) * 0.5;
    const double zz = (r22 + 1.0) * 0.5;
    const double xy = (r01 + r10) * 0.25;
    const double xz = (r02 + r20) * 0.25;
    const double yz = (r12 + r21) * 0.25;

    double ax = 0.0, ay = 0.0, az = 0.0;
    if (xx > yy && xx > zz) {
      ax = std::sqrt(std::max(0.0, xx));
      ay = (ax > 1e-12) ? (xy / ax) : 0.0;
      az = (ax > 1e-12) ? (xz / ax) : 0.0;
    } else if (yy > zz) {
      ay = std::sqrt(std::max(0.0, yy));
      ax = (ay > 1e-12) ? (xy / ay) : 0.0;
      az = (ay > 1e-12) ? (yz / ay) : 0.0;
    } else {
      az = std::sqrt(std::max(0.0, zz));
      ax = (az > 1e-12) ? (xz / az) : 0.0;
      ay = (az > 1e-12) ? (yz / az) : 0.0;
    }

    if (rx < 0)
      ax = -ax;
    if (ry < 0)
      ay = -ay;
    if (rz < 0)
      az = -az;

    rvec.data[0] = ax * theta;
    rvec.data[1] = ay * theta;
    rvec.data[2] = az * theta;
    return rvec;
  }

  const double k = theta / (2.0 * s);
  rvec.data[0] = k * rx;
  rvec.data[1] = k * ry;
  rvec.data[2] = k * rz;
  return rvec;
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

static std::optional<Matrix> matrix_from_py(const py::handle &obj) {
  if (obj.is_none())
    return std::nullopt;

  if (py::hasattr(obj, "detach") && py::hasattr(obj, "cpu") && py::hasattr(obj, "numpy")) {
    py::object np_arr = obj.attr("detach")().attr("cpu")().attr("numpy")();
    return matrix_from_py(np_arr);
  }

  py::array arr = py::array::ensure(obj);
  if (!arr)
    return std::nullopt;

  py::buffer_info info = arr.request();
  if (info.ndim == 1) {
    Matrix m;
    m.rows = static_cast<int>(info.shape[0]);
    m.cols = 1;
    m.data.resize(static_cast<size_t>(m.rows));
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
      m.data[static_cast<size_t>(i)] = py::cast<double>(arr[py::make_tuple(i)]);
    }
    return m;
  } else if (info.ndim == 2) {
    Matrix m;
    m.rows = static_cast<int>(info.shape[0]);
    m.cols = static_cast<int>(info.shape[1]);
    m.data.resize(static_cast<size_t>(m.rows * m.cols));
    for (py::ssize_t r = 0; r < info.shape[0]; ++r) {
      for (py::ssize_t c = 0; c < info.shape[1]; ++c) {
        m.data[static_cast<size_t>(r * info.shape[1] + c)] = py::cast<double>(arr[py::make_tuple(r, c)]);
      }
    }
    return m;
  }
  return std::nullopt;
}

static std::optional<double> scalar_from_py(const py::handle &obj) {
  if (obj.is_none())
    return std::nullopt;
  try {
    return py::cast<double>(obj);
  } catch (...) {
    return std::nullopt;
  }
}

static bool has_key_or_attr(const py::handle &obj, const char *name) {
  if (py::isinstance<py::dict>(obj)) {
    py::dict d = py::reinterpret_borrow<py::dict>(obj);
    return d.contains(py::str(name));
  }
  if (py::hasattr(obj, name))
    return true;
  try {
    py::object v = obj.attr("__contains__")(py::str(name));
    return py::cast<bool>(v);
  } catch (...) {
    return false;
  }
}

static py::object get_key_or_attr(const py::handle &obj, const char *name) {
  if (py::isinstance<py::dict>(obj)) {
    py::dict d = py::reinterpret_borrow<py::dict>(obj);
    if (d.contains(py::str(name)))
      return d[py::str(name)];
  }
  if (py::hasattr(obj, name))
    return obj.attr(name);
  try {
    return obj.attr("__getitem__")(py::str(name));
  } catch (...) {
    return py::none();
  }
}

static void write_header(std::ofstream &ofs) {
  ofs << "%YAML:1.0\r\n";
  ofs << "---\r\n";
}

static void write_list(std::ofstream &ofs, const std::string &key, const std::vector<std::string> &vals) {
  ofs << key << ":\r\n";
  for (const auto &v : vals) {
    ofs << "  - \"" << v << "\"\r\n";
  }
}

static void write_real(std::ofstream &ofs, const std::string &key, double v) {
  ofs.setf(std::ios::fixed);
  ofs.precision(10);
  ofs << key << ": " << v << "\r\n";
}

static void write_mat(std::ofstream &ofs, const std::string &key, const Matrix &m) {
  ofs << key << ": !!opencv-matrix\r\n";
  ofs << "  rows: " << m.rows << "\r\n";
  ofs << "  cols: " << m.cols << "\r\n";
  ofs << "  dt: d\r\n";
  ofs.setf(std::ios::fixed);
  ofs.precision(10);
  ofs << "  data: [";
  for (int idx = 0; idx < m.rows * m.cols; ++idx) {
    if (idx)
      ofs << ", ";
    ofs << m.data[static_cast<size_t>(idx)];
  }
  ofs << "]\r\n";
}

static Matrix ensure_5x1_D(Matrix D) {
  if (D.rows == 1 && D.cols == 4) {
    Matrix out;
    out.rows = 5;
    out.cols = 1;
    out.data = {D.data[0], D.data[1], D.data[2], D.data[3], 0.0};
    return out;
  }
  if (D.rows == 4 && D.cols == 1) {
    Matrix out;
    out.rows = 5;
    out.cols = 1;
    out.data = {D.data[0], D.data[1], D.data[2], D.data[3], 0.0};
    return out;
  }
  if (D.rows == 1 && D.cols == 5) {
    Matrix out;
    out.rows = 5;
    out.cols = 1;
    out.data = {D.data[0], D.data[1], D.data[2], D.data[3], D.data[4]};
    return out;
  }
  if (D.rows == 5 && D.cols == 1) {
    return D;
  }
  Matrix out;
  out.rows = 5;
  out.cols = 1;
  out.data = {0, 0, 0, 0, 0};
  return out;
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


static py::dict read_camera_cpp(const std::string &intri_path, const std::string &extri_path) {
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
    Matrix invK = inv3x3(K);

    double Hs = 0.0, Ws = 0.0;
    int H = -1, W = -1;
    if (get_scalar(intri, "H_" + cam, &Hs)) {
      H = static_cast<int>(Hs);
      if (H == 0)
        H = -1;
    }
    if (get_scalar(intri, "W_" + cam, &Ws)) {
      W = static_cast<int>(Ws);
      if (W == 0)
        W = -1;
    }

    Matrix Tvec;
    if (!get_matrix(extri, "T_" + cam, &Tvec)) {
      throw std::runtime_error("Missing T_" + cam);
    }
    Matrix Rvec;
    Matrix R;
    bool hasR = get_matrix(extri, "Rot_" + cam, &R);
    bool hasRvec = get_matrix(extri, "R_" + cam, &Rvec);
    if (hasR) {
      Rvec = rodrigues_R_to_rvec(R);
    } else if (hasRvec) {
      R = rodrigues_rvec_to_R(Rvec);
    } else {
      throw std::runtime_error("Either R_" + cam + " or Rot_" + cam + " must be provided");
    }

    Matrix RT = hstack_R_T(R, Tvec);
    Matrix P = matmul(K, RT);

    Matrix Cmat;
    {
      Matrix Rt = transpose(normalize_rvec(Rvec));
      Matrix Tn = Tvec;
      if (Tn.rows == 1 && Tn.cols == 3) {
        Tn.rows = 3;
        Tn.cols = 1;
      }
      Matrix Ct = matmul(Rt, Tn);
      Cmat.rows = 1;
      Cmat.cols = 1;
      Cmat.data = {-Ct.data[0]};
    }

    py::object Dobj = py::none();
    Matrix D;
    if (get_matrix(intri, "D_" + cam, &D)) {
      Dobj = to_numpy(D);
    } else if (get_matrix(intri, "dist_" + cam, &D)) {
      Dobj = to_numpy(D);
    }

    double t = 0.0, v = 0.0;
    if (!get_scalar(extri, "t_" + cam, &t))
      t = 0.0;
    if (!get_scalar(extri, "v_" + cam, &v))
      v = 0.0;

    double n = 0.0001;
    double f = 1e6;
    double nv = 0.0, fv = 0.0;
    if (get_scalar(extri, "n_" + cam, &nv) && std::abs(nv) > 1e-18) {
      n = nv;
    }
    if (get_scalar(extri, "f_" + cam, &fv) && std::abs(fv) > 1e-18) {
      f = fv;
    }

    Matrix bounds;
    if (!get_matrix(extri, "bounds_" + cam, &bounds)) {
      bounds.rows = 2;
      bounds.cols = 3;
      bounds.data = {-1e6, -1e6, -1e6, 1e6, 1e6, 1e6};
    }

    Matrix ccm;
    if (!get_matrix(intri, "ccm_" + cam, &ccm)) {
      ccm = mat_identity3();
    }

    c["K"] = to_numpy(K);
    c["H"] = H;
    c["W"] = W;
    c["invK"] = to_numpy(invK);
    c["R"] = to_numpy(R);
    c["T"] = to_numpy(Tvec);
    c["C"] = to_numpy(Cmat);
    c["RT"] = to_numpy(RT);
    c["Rvec"] = to_numpy(normalize_rvec(Rvec));
    c["P"] = to_numpy(P);
    c["D"] = Dobj;
    c["t"] = t;
    c["v"] = v;
    c["n"] = n;
    c["f"] = f;
    c["bounds"] = to_numpy(bounds);
    c["ccm"] = to_numpy(ccm);

    cams[py::str(cam)] = c;
  }

  return cams;
}

static void write_camera_cpp(py::object cameras_obj, const std::string &path, const std::string &intri_name_in,
                             const std::string &extri_name_in) {
  py::dict cameras = py::dict(cameras_obj);

  std::vector<std::string> cam_names;
  cam_names.reserve(cameras.size());
  struct CamRec {
    std::string name;
    Matrix K;
    std::optional<double> H;
    std::optional<double> W;
    Matrix D;
    Matrix R;
    Matrix Rvec;
    Matrix T;
    std::optional<double> t;
    std::optional<double> n;
    std::optional<double> f;
    std::optional<Matrix> bounds;
    std::optional<Matrix> ccm;
    std::optional<Matrix> rdist;
  };
  std::vector<CamRec> recs;
  recs.reserve(cameras.size());

  for (auto item : cameras) {
    std::string key = py::cast<std::string>(item.first);
    if (key == "basenames")
      continue;
    std::string name = key;
    auto dot = name.find('.');
    if (dot != std::string::npos)
      name = name.substr(0, dot);
    cam_names.push_back(name);

    py::handle val = item.second;
    CamRec rec;
    rec.name = name;

    {
      py::object Kobj = get_key_or_attr(val, "K");
      auto Km = matrix_from_py(Kobj);
      if (!Km)
        throw std::runtime_error("Missing/invalid K for camera: " + name);
      rec.K = *Km;
    }

    if (has_key_or_attr(val, "H")) {
      auto hv = scalar_from_py(get_key_or_attr(val, "H"));
      if (hv)
        rec.H = *hv;
    }
    if (has_key_or_attr(val, "W")) {
      auto wv = scalar_from_py(get_key_or_attr(val, "W"));
      if (wv)
        rec.W = *wv;
    }

    {
      py::object Dobj = py::none();
      if (has_key_or_attr(val, "D"))
        Dobj = get_key_or_attr(val, "D");
      else if (has_key_or_attr(val, "dist"))
        Dobj = get_key_or_attr(val, "dist");

      Matrix Dm;
      auto Dopt = matrix_from_py(Dobj);
      if (Dopt)
        Dm = *Dopt;
      else {
        Dm.rows = 5;
        Dm.cols = 1;
        Dm.data = {0, 0, 0, 0, 0};
      }
      rec.D = ensure_5x1_D(Dm);
    }

    {
      bool hasR = has_key_or_attr(val, "R");
      bool hasRvec = has_key_or_attr(val, "Rvec");
      if (!hasR && !hasRvec)
        throw std::runtime_error("Need R or Rvec for camera: " + name);

      if (hasR) {
        auto Rm = matrix_from_py(get_key_or_attr(val, "R"));
        if (!Rm)
          throw std::runtime_error("Invalid R for camera: " + name);
        rec.R = *Rm;
      }
      if (hasRvec) {
        auto rvm = matrix_from_py(get_key_or_attr(val, "Rvec"));
        if (!rvm)
          throw std::runtime_error("Invalid Rvec for camera: " + name);
        rec.Rvec = normalize_rvec(*rvm);
      }

      if (!hasR) {
        rec.R = rodrigues_rvec_to_R(rec.Rvec);
      }
      if (!hasRvec) {
        rec.Rvec = rodrigues_R_to_rvec(rec.R);
      }
    }

    {
      py::object Tobj = get_key_or_attr(val, "T");
      auto Tm = matrix_from_py(Tobj);
      if (!Tm)
        throw std::runtime_error("Missing/invalid T for camera: " + name);
      rec.T = *Tm;
      if (rec.T.rows == 1 && rec.T.cols == 3) {
        rec.T.rows = 3;
        rec.T.cols = 1;
      }
      if (!(rec.T.rows == 3 && rec.T.cols == 1))
        throw std::runtime_error("T must be 3x1 (or 1x3) for camera: " + name);
    }

    if (has_key_or_attr(val, "t")) {
      auto tv = scalar_from_py(get_key_or_attr(val, "t"));
      if (tv)
        rec.t = *tv;
    }
    if (has_key_or_attr(val, "n")) {
      auto nv = scalar_from_py(get_key_or_attr(val, "n"));
      if (nv)
        rec.n = *nv;
    }
    if (has_key_or_attr(val, "f")) {
      auto fv = scalar_from_py(get_key_or_attr(val, "f"));
      if (fv)
        rec.f = *fv;
    }
    if (has_key_or_attr(val, "bounds")) {
      auto bm = matrix_from_py(get_key_or_attr(val, "bounds"));
      if (bm)
        rec.bounds = *bm;
    }
    if (has_key_or_attr(val, "ccm")) {
      auto cm = matrix_from_py(get_key_or_attr(val, "ccm"));
      if (cm)
        rec.ccm = *cm;
    }
    if (has_key_or_attr(val, "rdist")) {
      auto rm = matrix_from_py(get_key_or_attr(val, "rdist"));
      if (rm)
        rec.rdist = *rm;
    }

    recs.push_back(std::move(rec));
  }

  std::string intri_name = intri_name_in;
  std::string extri_name = extri_name_in;
  if (intri_name.empty() || extri_name.empty()) {
    std::filesystem::path p(path);
    intri_name = (p / "intri.yml").string();
    extri_name = (p / "extri.yml").string();
  }

  py::gil_scoped_release release;

  std::filesystem::create_directories(std::filesystem::path(path));

  std::ofstream intri(intri_name, std::ios::out | std::ios::trunc);
  std::ofstream extri(extri_name, std::ios::out | std::ios::trunc);
  if (!intri.is_open())
    throw std::runtime_error("Failed to open for write: " + intri_name);
  if (!extri.is_open())
    throw std::runtime_error("Failed to open for write: " + extri_name);

  write_header(intri);
  write_header(extri);
  write_list(intri, "names", cam_names);
  write_list(extri, "names", cam_names);

  for (const auto &rec : recs) {
    const std::string &key = rec.name;
    write_mat(intri, "K_" + key, rec.K);
    if (rec.H)
      write_real(intri, "H_" + key, *rec.H);
    if (rec.W)
      write_real(intri, "W_" + key, *rec.W);

    write_mat(intri, "D_" + key, rec.D);
    if (rec.ccm)
      write_mat(intri, "ccm_" + key, *rec.ccm);
    if (rec.rdist)
      write_mat(intri, "rdist_" + key, *rec.rdist);

    write_mat(extri, "R_" + key, rec.Rvec);
    write_mat(extri, "Rot_" + key, rec.R);
    write_mat(extri, "T_" + key, rec.T);

    if (rec.t)
      write_real(extri, "t_" + key, *rec.t);
    if (rec.n)
      write_real(extri, "n_" + key, *rec.n);
    if (rec.f)
      write_real(extri, "f_" + key, *rec.f);
    if (rec.bounds)
      write_mat(extri, "bounds_" + key, *rec.bounds);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read_camera_minimal", &read_camera_cpp_minimal, "Read intri/extri OpenCV-YAML cameras (GIL released)");
  m.def("read_camera", &read_camera_cpp, "Read intri/extri OpenCV-YAML cameras (GIL released)");
  m.def("write_camera", &write_camera_cpp, py::arg("cameras"), py::arg("path"), py::arg("intri_name") = "",
        py::arg("extri_name") = "", "Write intri/extri OpenCV-YAML cameras (GIL released)");
}
