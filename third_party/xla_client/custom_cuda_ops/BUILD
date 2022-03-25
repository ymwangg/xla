licenses(["notice"])  # Apache 2.0

cc_library(
    name = "curand_ops",
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
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/stream_executor/cuda:curand_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
