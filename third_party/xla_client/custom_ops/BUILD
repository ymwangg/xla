licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

cc_library(
    name = "cuda_ops",
    srcs = [
        "cuda_utils.cc",
        "curand_uniform.cc",
    ],
    hdrs = [
        "cuda_utils.h",
        "curand_uniform.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/stream_executor/cuda:curand_lib",
    ],
)
