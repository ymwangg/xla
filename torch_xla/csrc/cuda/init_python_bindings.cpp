#include <Python.h>
#include <c10/core/Device.h>
#include <c10/util/Optional.h>

#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl_bind.h"
#include "torch/csrc/jit/python/pybind.h"

namespace torch_xla {
namespace {

static int64_t seed_info_id = -127389;

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

void InitXlaModuleBindings(py::module m) {
  m.def("_get_tensors_handle",
        [](const std::vector<int64_t>& tensors) -> std::vector<int64_t> {
          std::vector<int64_t> handles;
          handles.push_back(seed_info_id);
          return handles;
        });
}
}  // namespace

void InitXlaBindings(py::module m) { InitXlaModuleBindings(m); }

}  // namespace torch_xla

PYBIND11_MODULE(_XLAC_CUDA, m) { torch_xla::InitXlaBindings(m); }
