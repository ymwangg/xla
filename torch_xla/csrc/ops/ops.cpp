#include "torch_xla/csrc/ops/ops.h"

#include <cmath>

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/logdet.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/helpers.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/generated/LazyIr.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"
#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/log_softmax_backward.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/softmax_backward.h"
#include "torch_xla/csrc/ops/sum.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

#define PTXLA_UNARY_OP(name, sym, xla_fn)                                   \
  torch::lazy::NodePtr name(const torch::lazy::Value& input) {              \
    auto lower_fn = [](const XlaNode& node,                                 \
                       LoweringContext* loctx) -> XlaOpVector {             \
      xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));           \
      return node.ReturnOp(xla_fn(xla_input), loctx);                       \
    };                                                                      \
    return GenericOp(torch::lazy::OpKind(sym), {input}, GetXlaShape(input), \
                     std::move(lower_fn));                                  \
  }

#define PTXLA_BINARY_OP(name, sym, xla_fn)                                     \
  torch::lazy::NodePtr name(const torch::lazy::Value& input0,                  \
                            const torch::lazy::Value& input1) {                \
    auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp { \
      auto promoted = XlaHelpers::Promote(operands[0], operands[1]);           \
      return xla_fn(promoted.first, promoted.second);                          \
    };                                                                         \
    auto lower_fn = [](const XlaNode& node,                                    \
                       LoweringContext* loctx) -> XlaOpVector {                \
      xla::XlaOp xla_input0 = loctx->GetOutputOp(node.operand(0));             \
      xla::XlaOp xla_input1 = loctx->GetOutputOp(node.operand(1));             \
      auto promoted = XlaHelpers::Promote(xla_input0, xla_input1);             \
      return node.ReturnOp(xla_fn(promoted.first, promoted.second), loctx);    \
    };                                                                         \
    return GenericOp(                                                          \
        torch::lazy::OpKind(sym), {input0, input1},                            \
        [&]() {                                                                \
          return InferOutputShape({GetXlaShape(input0), GetXlaShape(input1)},  \
                                  shape_fn);                                   \
        },                                                                     \
        std::move(lower_fn));                                                  \
  }

PTXLA_UNARY_OP(Neg, at::aten::neg, xla::Neg);
PTXLA_UNARY_OP(Exp, at::aten::exp, xla::Exp);
PTXLA_UNARY_OP(Log, at::aten::log, xla::Log);
PTXLA_UNARY_OP(Log1p, at::aten::log1p, xla::Log1p);
PTXLA_UNARY_OP(Sqrt, at::aten::sqrt, xla::Sqrt);

PTXLA_BINARY_OP(Min, at::aten::min, xla::Min);
PTXLA_BINARY_OP(Pow, at::aten::pow, xla::Pow);
PTXLA_BINARY_OP(Fmod, at::aten::fmod, xla::Rem);
PTXLA_BINARY_OP(Atan2, at::aten::atan2, xla::Atan2);

torch::lazy::NodePtr LogBase(const torch::lazy::Value& input,
                             torch::lazy::OpKind op, double base) {
  auto lower_fn = [base](const XlaNode& node,
                         LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp result = xla::Log(xla_input);
    xla::XlaOp ln_base = XlaHelpers::ScalarValue<float>(
        1.0 / std::log(base), node.xla_shape().element_type(),
        xla_input.builder());
    return node.ReturnOp(result * ln_base, loctx);
  };
  return GenericOp(op, {input}, GetXlaShape(input), std::move(lower_fn),
                   /*num_outputs=*/1, torch::lazy::MHash(base));
}

torch::lazy::NodePtr SgnOp(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildSgn(xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::sgn), {input},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr SignOp(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildSign(xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::sign), {input},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr Prelu(const torch::lazy::Value& input,
                           const torch::lazy::Value& weight) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_weight = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildPrelu(xla_input, xla_weight);
    return node.ReturnOp(xla_output, loctx);
  };

  return GenericOp(torch::lazy::OpKind(at::aten::prelu), {input, weight},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr LogSigmoid(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOps(BuildLogSigmoid(xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::log_sigmoid), {input},
                   GetXlaShape(input), std::move(lower_fn), /*num_outputs=*/2);
}

torch::lazy::NodePtr SiLU(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(xla_input * BuildSigmoid(xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::silu), {input},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr SiLUBackward(const torch::lazy::Value& grad_output,
                                  const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_grad_output = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(BuildSiLUBackward(xla_grad_output, xla_input), loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildSiLUBackward(operands[0], operands[1]);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::silu_backward), {grad_output, input},
      [&]() {
        return InferOutputShape({GetXlaShape(grad_output), GetXlaShape(input)},
                                lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr Sigmoid(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildSigmoid(xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::sigmoid), {input},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr SigmoidBackward(const torch::lazy::Value& grad_output,
                                     const torch::lazy::Value& output) {
  return grad_output * (ScalarOp(1, GetXlaShape(output)) - output) * output;
}

torch::lazy::NodePtr LogSoftmaxBackwardOp(const torch::lazy::Value& grad_output,
                                          const torch::lazy::Value& output,
                                          int64_t dim) {
  return torch::lazy::MakeNode<LogSoftmaxBackward>(
      grad_output, output,
      torch::lazy::GetCanonicalDimensionIndex(dim,
                                              GetXlaShape(grad_output).rank()));
}

torch::lazy::NodePtr SoftmaxBackwardOp(const torch::lazy::Value& grad_output,
                                       const torch::lazy::Value& output,
                                       int64_t dim) {
  return torch::lazy::MakeNode<SoftmaxBackward>(
      grad_output, output,
      torch::lazy::GetCanonicalDimensionIndex(dim,
                                              GetXlaShape(grad_output).rank()));
}

torch::lazy::NodePtr Clamp(const torch::lazy::Value& input,
                           const torch::lazy::Value& min,
                           const torch::lazy::Value& max) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_min = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_max = loctx->GetOutputOp(node.operand(2));
    xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
    xla_min = ConvertTo(xla_min, XlaHelpers::TypeOfXlaOp(xla_min), input_type,
                        /*device=*/nullptr);
    xla_max = ConvertTo(xla_max, XlaHelpers::TypeOfXlaOp(xla_max), input_type,
                        /*device=*/nullptr);
    return node.ReturnOp(xla::Clamp(xla_min, xla_input, xla_max), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::clamp), {input, min, max},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr Celu(const torch::lazy::Value& input,
                          const at::Scalar& alpha) {
  auto lower_fn = [=](const XlaNode& node,
                      LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildCelu(xla_input, alpha), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::celu), {input},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr AddMatMulOp(const torch::lazy::Value& input,
                                 const torch::lazy::Value& weight,
                                 const torch::lazy::Value& bias) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    XLA_CHECK_EQ(node.operands().size(), 3) << "Unexpected number of operands";
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_weight = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_bias = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(BuildMatMul(xla_input, xla_weight, xla_bias), loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMatMul(operands[0], operands[1], operands[2]);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::addmm), {input, weight, bias},
      [&]() {
        return InferOutputShape(
            {GetXlaShape(input), GetXlaShape(weight), GetXlaShape(bias)},
            lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr Dot(const torch::lazy::Value& input,
                         const torch::lazy::Value& weight) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_weight = loctx->GetOutputOp(node.operand(1));
    static bool opt = xla::sys_util::GetEnvBool("XLA_OPT", false);
    if (opt) {
      std::cout << "opt barrier" << std::endl;
      return node.ReturnOp(
          xla::OptimizationBarrier(BuildDot(xla_input, xla_weight)), loctx);
    }
    return node.ReturnOp(BuildDot(xla_input, xla_weight), loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildDot(operands[0], operands[1]);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::mm), {input, weight},
      [&]() {
        return InferOutputShape({GetXlaShape(input), GetXlaShape(weight)},
                                lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr MatMul(const torch::lazy::Value& lhs,
                            const torch::lazy::Value& rhs) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_lhs = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_rhs = loctx->GetOutputOp(node.operand(1));
    std::tie(xla_lhs, xla_rhs) = XlaHelpers::PromoteValues(xla_lhs, xla_rhs);
    static bool opt = xla::sys_util::GetEnvBool("XLA_OPT", false);
    if (opt) {
      std::cout << "opt barrier" << std::endl;
      return node.ReturnOp(
          xla::OptimizationBarrier(CreateMatMul(xla_lhs, xla_rhs)), loctx);
    }
    return node.ReturnOp(CreateMatMul(xla_lhs, xla_rhs), loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateMatMul(operands[0], operands[1]);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::matmul), {lhs, rhs},
      [&]() {
        return InferOutputShape({GetXlaShape(lhs), GetXlaShape(rhs)},
                                lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr AdaptiveMaxPool2dBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp grad_output = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp input = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildAdaptiveMaxPoolNdBackward(
        /*out_backprop=*/grad_output, /*input=*/input, /*pool_dim=*/2);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveMaxPoolNdBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1],
                                          /*pool_dim=*/2);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::adaptive_max_pool2d_backward),
      {grad_output, input},
      [&]() {
        return InferOutputShape({GetXlaShape(grad_output), GetXlaShape(input)},
                                lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr ComparisonOp(c10::Symbol kind,
                                  const torch::lazy::Value& input,
                                  const torch::lazy::Value& other) {
  auto lower_fn = [kind](const XlaNode& node,
                         LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_other = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildComparisonOp(kind, xla_input, xla_other);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [kind](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(kind, operands[0], operands[1]);
  };
  return GenericOp(
      torch::lazy::OpKind(kind), {input, other},
      [&]() {
        return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                                lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr Where(const torch::lazy::Value& condition,
                           const torch::lazy::Value& input,
                           const torch::lazy::Value& other) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_condition = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_other = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp pred_condition =
        ConvertTo(xla_condition, XlaHelpers::TypeOfXlaOp(xla_condition),
                  xla::PrimitiveType::PRED, /*device=*/nullptr);
    auto promoted_branches = XlaHelpers::PromoteShapes(xla_input, xla_other);
    return node.ReturnOp(xla::Select(pred_condition, promoted_branches.first,
                                     promoted_branches.second),
                         loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::where),
                   {condition, input, other}, GetXlaShape(input),
                   std::move(lower_fn));
}

torch::lazy::NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
                            const at::Scalar& step,
                            at::ScalarType scalar_type) {
  xla::PrimitiveType type = MakeXlaPrimitiveType(scalar_type,
                                                 /*device=*/nullptr);
  XLA_CHECK_NE(step.toDouble(), 0.0);
  XLA_CHECK(!std::isnan(start.toDouble()) && !std::isnan(end.toDouble()))
      << "unsupported range: " << start.toDouble() << " -> " << end.toDouble();
  XLA_CHECK((start.toDouble() <= end.toDouble() && step.toDouble() > 0.0) ||
            (start.toDouble() >= end.toDouble() && step.toDouble() < 0.0));
  xla::Literal values;
  switch (type) {
    case xla::PrimitiveType::BF16:
      values = XlaHelpers::Range<tensorflow::bfloat16>(
          static_cast<tensorflow::bfloat16>(start.toFloat()),
          static_cast<tensorflow::bfloat16>(end.toFloat()),
          static_cast<tensorflow::bfloat16>(step.toFloat()));
      break;
    case xla::PrimitiveType::F16:
      values =
          XlaHelpers::Range<xla::half>(static_cast<xla::half>(start.toHalf()),
                                       static_cast<xla::half>(end.toHalf()),
                                       static_cast<xla::half>(step.toHalf()));
      break;
    case xla::PrimitiveType::F32:
      values = XlaHelpers::Range<float>(start.toFloat(), end.toFloat(),
                                        step.toFloat());
      break;
    case xla::PrimitiveType::F64:
      values = XlaHelpers::Range<double>(start.toDouble(), end.toDouble(),
                                         step.toDouble());
      break;
    case xla::PrimitiveType::U8:
      values = XlaHelpers::Range<uint8_t>(start.toByte(), end.toByte(),
                                          step.toByte());
      break;
    case xla::PrimitiveType::S8:
      values = XlaHelpers::Range<int8_t>(start.toChar(), end.toChar(),
                                         step.toChar());
      break;
    case xla::PrimitiveType::S16:
      values = XlaHelpers::Range<int16_t>(start.toShort(), end.toShort(),
                                          step.toShort());
      break;
    case xla::PrimitiveType::U16:
      values =
          XlaHelpers::Range<uint16_t>(start.toInt(), end.toInt(), step.toInt());
      break;
    case xla::PrimitiveType::S32:
      values =
          XlaHelpers::Range<int32_t>(start.toInt(), end.toInt(), step.toInt());
      break;
    case xla::PrimitiveType::U32:
      values = XlaHelpers::Range<uint32_t>(start.toLong(), end.toLong(),
                                           step.toLong());
      break;
    case xla::PrimitiveType::S64:
      values = XlaHelpers::Range<int64_t>(start.toLong(), end.toLong(),
                                          step.toLong());
      break;
    case xla::PrimitiveType::U64:
      values = XlaHelpers::Range<uint64_t>(start.toLong(), end.toLong(),
                                           step.toLong());
      break;
    default:
      XLA_ERROR() << "XLA type not supported: " << type;
  }
  return torch::lazy::MakeNode<Constant>(std::move(values));
}

torch::lazy::NodePtr BroadcastTensors(
    c10::ArrayRef<torch::lazy::Value> tensors) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    std::vector<xla::XlaOp> xla_operands;
    for (const torch::lazy::Output& operand : node.operands()) {
      xla_operands.push_back(loctx->GetOutputOp(operand));
    }
    return node.ReturnOps(CreateBroadcastTensors(xla_operands), loctx);
  };
  std::vector<xla::Shape> tensor_shapes;
  for (const torch::lazy::Value& tensor : tensors) {
    tensor_shapes.push_back(GetXlaShape(tensor));
  }
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    auto results = CreateBroadcastTensors(operands);
    return xla::Tuple(results.front().builder(), results);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::broadcast_tensors), tensors,
      [&]() { return InferOutputShape(tensor_shapes, lower_for_shape_fn); },
      std::move(lower_fn), /*num_outputs=*/tensors.size());
}

torch::lazy::NodePtr Norm(const torch::lazy::Value& input,
                          const c10::optional<at::Scalar>& p,
                          c10::optional<at::ScalarType> dtype,
                          absl::Span<const int64_t> dims, bool keepdim) {
  torch::lazy::ScopePusher ir_scope(at::aten::norm.toQualString());
  auto dimensions = torch::lazy::ToVector<int64_t>(dims);
  if (dimensions.empty()) {
    dimensions = torch::lazy::Iota<int64_t>(GetXlaShape(input).rank());
  }
  if (!p.has_value() || p->toDouble() == 2.0) {
    torch::lazy::NodePtr square = input * input;
    torch::lazy::NodePtr result =
        torch::lazy::MakeNode<Sum>(square, dimensions, keepdim, dtype);
    return Sqrt(result);
  }
  double norm_value = p->toDouble();
  if (norm_value == 1.0) {
    // Contrary to documentation, norm(p=1) has nothing to do with traces and
    // standard mathematical definitions of nuclear norms:
    //
    //   >>> import torch
    //   >>> x = torch.randn(4, 4)
    //   >>> print(torch.norm(x, 1))
    //   tensor(11.9437)
    //   >>> print(torch.trace(x.abs()))
    //   tensor(3.1235)
    //   >>> print(x.abs().sum())
    //   tensor(11.9437)
    return torch::lazy::MakeNode<Sum>(
        torch::lazy::MakeNode<Abs>(input, std::vector<torch::lazy::Shape>()),
        dimensions, keepdim, dtype);
  }
  // Generic sum(x^p)^(1/p) norms.
  torch::lazy::NodePtr norm_exp =
      ScalarOp(norm_value, GetXlaShape(input).element_type());
  torch::lazy::NodePtr norm_exp_inv =
      ScalarOp(1.0 / norm_value, GetXlaShape(input).element_type());
  torch::lazy::NodePtr exp =
      Pow(torch::lazy::MakeNode<Abs>(input, std::vector<torch::lazy::Shape>()),
          norm_exp);
  torch::lazy::NodePtr result =
      torch::lazy::MakeNode<Sum>(exp, dimensions, keepdim, dtype);
  return Pow(result, norm_exp_inv);
}

torch::lazy::NodePtr Identity(int64_t lines, int64_t cols,
                              xla::PrimitiveType element_type) {
  auto lower_fn = [=](const XlaNode& node,
                      LoweringContext* loctx) -> XlaOpVector {
    return node.ReturnOp(
        xla::IdentityMatrix(loctx->builder(), element_type, lines, cols),
        loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::eye),
                   xla::ShapeUtil::MakeShape(element_type, {lines, cols}),
                   std::move(lower_fn), /*num_outputs=*/1,
                   torch::lazy::MHash(lines, cols));
}

torch::lazy::NodePtr EluBackward(const torch::lazy::Value& grad_output,
                                 const torch::lazy::Value& output,
                                 const at::Scalar& alpha,
                                 const at::Scalar& scale,
                                 const at::Scalar& input_scale) {
  auto lower_fn = [=](const XlaNode& node,
                      LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_grad_output = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_output = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(BuildEluBackward(xla_grad_output, xla_output, alpha,
                                          scale, input_scale),
                         loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::elu_backward),
                   {grad_output, output}, GetXlaShape(output),
                   std::move(lower_fn));
}

torch::lazy::NodePtr Gelu(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildGelu(xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::gelu), {input},
                   GetXlaShape(input), std::move(lower_fn));
}

torch::lazy::NodePtr GeluBackward(const torch::lazy::Value& grad_output,
                                  const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_grad_output = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(BuildGeluBackward(xla_grad_output, xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::gelu_backward),
                   {grad_output, input}, GetXlaShape(input),
                   std::move(lower_fn));
}

torch::lazy::NodePtr Lshift(const torch::lazy::Value& input,
                            const at::Scalar& other) {
  torch::lazy::ScopePusher ir_scope(at::aten::__lshift__.toQualString());
  return input * ScalarOp(pow(2, other.to<double>()), GetXlaShape(input));
}

torch::lazy::NodePtr Lshift(const torch::lazy::Value& input,
                            const torch::lazy::Value& other) {
  torch::lazy::ScopePusher ir_scope(at::aten::__lshift__.toQualString());
  return input * Pow(ScalarOp(2, GetXlaShape(input)), other);
}

torch::lazy::NodePtr Rshift(const torch::lazy::Value& input,
                            const at::Scalar& other) {
  torch::lazy::ScopePusher ir_scope(at::aten::__rshift__.toQualString());
  return input / ScalarOp(pow(2, other.to<double>()), GetXlaShape(input));
}

torch::lazy::NodePtr Rshift(const torch::lazy::Value& input,
                            const torch::lazy::Value& other) {
  torch::lazy::ScopePusher ir_scope(at::aten::__rshift__.toQualString());
  return input / Pow(ScalarOp(2, GetXlaShape(input)), other);
}

torch::lazy::NodePtr Remainder(const torch::lazy::Value& input,
                               const torch::lazy::Value& divisor) {
  torch::lazy::ScopePusher ir_scope(at::aten::remainder.toQualString());
  torch::lazy::NodePtr f = Fmod(
      input,
      torch::lazy::MakeNode<Abs>(divisor, std::vector<torch::lazy::Shape>()));
  return f + divisor * ComparisonOp(
                           at::aten::lt,
                           torch::lazy::MakeNode<Sign>(
                               f, std::vector<torch::lazy::Shape>()) *
                               torch::lazy::MakeNode<Sign>(
                                   divisor, std::vector<torch::lazy::Shape>()),
                           ScalarOp(0, GetXlaShape(input)));
}

torch::lazy::NodePtr MaxUnary(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
    xla::PrimitiveType element_type = input_shape.element_type();
    XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(element_type);
    xla::XlaOp init_value =
        XlaHelpers::ScalarValue(min_max.min, element_type, loctx->builder());
    xla::XlaOp result = xla::Reduce(
        xla_input, init_value, XlaHelpers::CreateMaxComputation(element_type),
        torch::lazy::Iota<int64_t>(input_shape.rank()));
    return node.ReturnOp(xla::Reshape(result, {}), loctx);
  };
  XLA_CHECK_GT(xla::ShapeUtil::ElementsIn(GetXlaShape(input)), 0);
  return GenericOp(
      torch::lazy::OpKind(at::aten::max), {input},
      xla::ShapeUtil::MakeShape(GetXlaShape(input).element_type(), {}),
      std::move(lower_fn));
}

torch::lazy::NodePtr MinUnary(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
    xla::PrimitiveType element_type = input_shape.element_type();
    XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(element_type);
    xla::XlaOp init_value =
        XlaHelpers::ScalarValue(min_max.max, element_type, loctx->builder());
    xla::XlaOp result = xla::Reduce(
        xla_input, init_value, XlaHelpers::CreateMinComputation(element_type),
        torch::lazy::Iota<int64_t>(input_shape.rank()));
    return node.ReturnOp(xla::Reshape(result, {}), loctx);
  };
  XLA_CHECK_GT(xla::ShapeUtil::ElementsIn(GetXlaShape(input)), 0);
  return GenericOp(
      torch::lazy::OpKind(at::aten::min), {input},
      xla::ShapeUtil::MakeShape(GetXlaShape(input).element_type(), {}),
      std::move(lower_fn));
}

torch::lazy::NodePtr TanhGelu(const torch::lazy::Value& input) {
  // TODO: add proper lowering function
  torch::lazy::ScopePusher ir_scope("aten::tanh_gelu");
  const xla::Shape& shape = GetXlaShape(input);
  // inner = math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(input, 3))
  // input * 0.5 * (1.0 + torch.tanh(inner))
  const static float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  torch::lazy::NodePtr beta = ScalarOp(kBeta, shape);
  torch::lazy::NodePtr kappa = ScalarOp(0.044715, shape);
  torch::lazy::NodePtr three = ScalarOp(3, shape);
  torch::lazy::NodePtr one = ScalarOp(1, shape);
  torch::lazy::NodePtr half = ScalarOp(0.5, shape);
  torch::lazy::NodePtr inner = beta * (input + kappa * Pow(input, three));
  return half * input *
         (one + torch::lazy::MakeNode<Tanh>(inner,
                                            std::vector<torch::lazy::Shape>()));
}

torch::lazy::NodePtr TanhGeluBackward(const torch::lazy::Value& grad,
                                      const torch::lazy::Value& input) {
  // TODO: add proper lowering function
  torch::lazy::ScopePusher ir_scope("aten::tanh_gelu_backward");
  const xla::Shape& shape = GetXlaShape(input);
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  torch::lazy::NodePtr beta = ScalarOp(kBeta, shape);
  torch::lazy::NodePtr kappa = ScalarOp(0.044715, shape);
  torch::lazy::NodePtr one = ScalarOp(1, shape);
  torch::lazy::NodePtr two = ScalarOp(2, shape);
  torch::lazy::NodePtr three = ScalarOp(3, shape);
  torch::lazy::NodePtr half = ScalarOp(0.5, shape);
  torch::lazy::NodePtr inner = beta * (input + kappa * Pow(input, three));
  torch::lazy::NodePtr tanh_inner =
      torch::lazy::MakeNode<Tanh>(inner, std::vector<torch::lazy::Shape>());

  torch::lazy::NodePtr left = half * input;
  torch::lazy::NodePtr right = one + tanh_inner;

  torch::lazy::NodePtr left_derivative = half * right;

  torch::lazy::NodePtr tanh_derivative = one - tanh_inner * tanh_inner;
  torch::lazy::NodePtr inner_derivative =
      beta * (one + three * kappa * Pow(input, two));
  torch::lazy::NodePtr right_derivative =
      left * tanh_derivative * inner_derivative;

  return grad * (left_derivative + right_derivative);
}

torch::lazy::NodePtr BaddBmm(const torch::lazy::Value& lhs,
                             const torch::lazy::Value& rhs,
                             const torch::lazy::Value& bias,
                             const torch::lazy::Value& product_multiplier,
                             const torch::lazy::Value& bias_multiplier) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_lhs = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_rhs = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_bias = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp xla_product_multiplier = loctx->GetOutputOp(node.operand(3));
    xla::XlaOp xla_bias_multiplier = loctx->GetOutputOp(node.operand(4));
    std::tie(xla_lhs, xla_rhs) = XlaHelpers::PromoteValues(xla_lhs, xla_rhs);

    return node.ReturnOp(
        BuildMatMulWithMultiplier(xla_lhs, xla_rhs, xla_bias,
                                  xla_product_multiplier, xla_bias_multiplier),
        loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMatMulWithMultiplier(operands[0], operands[1], operands[2],
                                     operands[3], operands[4]);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::baddbmm),
      {lhs, rhs, bias, product_multiplier, bias_multiplier},
      [&]() {
        return InferOutputShape(
            {GetXlaShape(lhs), GetXlaShape(rhs), GetXlaShape(bias),
             GetXlaShape(product_multiplier), GetXlaShape(bias_multiplier)},
            lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr Lerp(const torch::lazy::Value& start,
                          const torch::lazy::Value& end,
                          const torch::lazy::Value& weight) {
  torch::lazy::ScopePusher ir_scope(at::aten::lerp.toQualString());
  return start + weight * (end - start);
}

torch::lazy::NodePtr XLogY(const torch::lazy::Value& input,
                           const torch::lazy::Value& other) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_other = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildXLogY(xla_input, xla_other);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2) << "Unexpected number of operands";
    return BuildXLogY(operands[0], operands[1]);
  };
  return GenericOp(
      torch::lazy::OpKind(at::aten::xlogy), {input, other},
      [&]() {
        return InferOutputShape({GetXlaShape(input), GetXlaShape(other)},
                                lower_for_shape_fn);
      },
      std::move(lower_fn));
}

torch::lazy::NodePtr NanToNum(const torch::lazy::Value& input,
                              const torch::lazy::Value& nan,
                              const torch::lazy::Value& posinf,
                              const torch::lazy::Value& neginf) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp nan_replacement = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp posinf_replacement = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp neginf_replacement = loctx->GetOutputOp(node.operand(3));
    xla::XlaOp result =
        xla::Select(xla::IsNan(xla_input), nan_replacement,
                    xla::Select(xla::IsPosInf(xla_input), posinf_replacement,
                                xla::Select(xla::IsNegInf(xla_input),
                                            neginf_replacement, xla_input)));
    return node.ReturnOp(result, loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::nan_to_num),
                   {input, nan, posinf, neginf}, GetXlaShape(input),
                   std::move(lower_fn));
}

torch::lazy::NodePtr SLogDet(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::SignAndLogDet result = xla::SLogDet(xla_input);
    return node.ReturnOps({result.sign, result.logdet}, loctx);
  };

  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::SignAndLogDet result = xla::SLogDet(operands[0]);
    return xla::Tuple(operands[0].builder(), {result.sign, result.logdet});
  };

  return GenericOp(
      torch::lazy::OpKind(at::aten::slogdet), {input},
      [&]() {
        return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
      },
      std::move(lower_fn), /*num_outputs=*/2);
}

torch::lazy::NodePtr Softplus(const torch::lazy::Value& input,
                              const torch::lazy::Value& beta,
                              const torch::lazy::Value& threshold) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_beta = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_threshold = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp xla_output = BuildSoftplus(xla_input, xla_beta, xla_threshold);
    return node.ReturnOp(xla_output, loctx);
  };

  return GenericOp(torch::lazy::OpKind(at::aten::softplus),
                   {input, beta, threshold}, GetXlaShape(input),
                   std::move(lower_fn));
}

torch::lazy::NodePtr Selu(const torch::lazy::Value& input) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildSelu(xla_input), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::selu), {input},
                   GetXlaShape(input), std::move(lower_fn));
}

xla::XlaOp BuildCdistForward(xla::XlaOp x1, xla::XlaOp x2, xla::XlaOp p) {
  const xla::Shape& x1_shape = XlaHelpers::ShapeOfXlaOp(x1);
  const xla::Shape& x2_shape = XlaHelpers::ShapeOfXlaOp(x2);
  p = MaybeConvertTo(p, x1_shape.element_type());
  XLA_CHECK(x1_shape.rank() == x2_shape.rank())
      << "x1 and x2 must have same num of dimensions";
  XLA_CHECK(x1_shape.rank() == 2 || x1_shape.rank() == 3)
      << "x1 must have 2 or 3 dimensions";
  XLA_CHECK(x1_shape.dimensions(x1_shape.rank() - 1) ==
            x2_shape.dimensions(x2_shape.rank() - 1))
      << "the last dimension of x1 and x2 must match";
  std::vector<int64_t> bcast_shape;
  std::vector<int64_t> x1_dim_to_bcast;
  std::vector<int64_t> x2_dim_to_bcast;
  int64_t dim = 0;
  if (x1_shape.rank() == 3) {
    bcast_shape = {x1_shape.dimensions(0), x1_shape.dimensions(1),
                   x2_shape.dimensions(1), x1_shape.dimensions(2)};
    x1_dim_to_bcast = {0, 1, 3};
    x2_dim_to_bcast = {0, 2, 3};
    dim = 3;
  } else {
    bcast_shape = {x1_shape.dimensions(0), x2_shape.dimensions(0),
                   x1_shape.dimensions(1)};
    x1_dim_to_bcast = {0, 2};
    x2_dim_to_bcast = {1, 2};
    dim = 2;
  }
  xla::XlaOp x1_bcast = xla::BroadcastInDim(x1, bcast_shape, x1_dim_to_bcast);
  xla::XlaOp x2_bcast = xla::BroadcastInDim(x2, bcast_shape, x2_dim_to_bcast);
  xla::XlaOp abs = xla::Pow(xla::Abs(x1_bcast - x2_bcast), p);
  xla::XlaOp init_value = xla::Zero(x1.builder(), x1_shape.element_type());
  xla::XlaOp reduce = xla::Reduce(
      abs, init_value,
      XlaHelpers::CreateAddComputation(x1_shape.element_type()), {dim});
  xla::XlaOp one = xla::One(x1.builder(), x1_shape.element_type());
  xla::XlaOp p_norm = xla::Pow(reduce, xla::Div(one, p));
  return p_norm;
}

torch::lazy::NodePtr CdistForwardOp(const torch::lazy::Value& x1,
                                    const torch::lazy::Value& x2,
                                    const torch::lazy::Value& p) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp x1 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp x2 = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp p = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp output = BuildCdistForward(x1, x2, p);
    return node.ReturnOp(output, loctx);
  };

  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 3) << "Unexpected number of operands";
    return BuildCdistForward(operands[0], operands[1], operands[2]);
  };

  return GenericOp(
      torch::lazy::OpKind(at::aten::cdist), {x1, x2, p},
      [&]() {
        return InferOutputShape(
            {GetXlaShape(x1), GetXlaShape(x2), GetXlaShape(p)},
            lower_for_shape_fn);
      },
      std::move(lower_fn));
}

std::pair<xla::XlaOp, xla::XlaOp> BuildLinearSumAssignment(xla::XlaOp input) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  XLA_CHECK(input_shape.rank() == 2);
  int dim = input_shape.dimensions(0) < input_shape.dimensions(1)
                ? input_shape.dimensions(0)
                : input_shape.dimensions(1);
  xla::Shape output_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::S64, {dim});
  xla::XlaOp row_index = xla::Iota(input.builder(), output_shape, 0);
  xla::XlaOp col_index = xla::Iota(input.builder(), output_shape, 0);
  return {row_index, col_index};
}

torch::lazy::NodePtr LinearSumAssignmentNode(const torch::lazy::Value& input,
                                             bool maximize) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp input = loctx->GetOutputOp(node.operand(0));
    // auto result = BuildLinearSumAssignment(input);
    xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
    int dim = input_shape.dimensions(0) < input_shape.dimensions(1)
                  ? input_shape.dimensions(0)
                  : input_shape.dimensions(1);
    xla::Shape dim_shape =
        xla::ShapeUtil::MakeShape(xla::PrimitiveType::S64, {dim});
    xla::Shape output_shape = xla::ShapeUtil::MakeTupleShape({
        dim_shape,
        dim_shape,
    });
    std::string shape_proto;
    input_shape.ToProto().SerializeToString(&shape_proto);
    xla::XlaOp outputs = xla::CustomCall(
        loctx->builder(), "LinearSumAssignment", /*operands=*/{input},
        /*shape=*/output_shape, shape_proto, /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/xla::API_VERSION_STATUS_RETURNING);

    std::vector<xla::XlaOp> ops = {xla::GetTupleElement(outputs, 0),
                                   xla::GetTupleElement(outputs, 1)};
    return node.ReturnOps(ops, loctx);
  };

  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    auto result = BuildLinearSumAssignment(operands[0]);
    return xla::Tuple(operands[0].builder(), {result.first, result.second});
  };

  return GenericOp(
      torch::lazy::OpKind(at::aten::cdist), {input},
      [&]() {
        return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
      },
      std::move(lower_fn),
      /*num_outputs=*/2, torch::lazy::MHash(maximize));
}

torch::lazy::NodePtr Dropout(const torch::lazy::Value& input,
                             const torch::lazy::Value& probability) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp input = loctx->GetOutputOp(node.operand(0));
    xla::PrimitiveType type = XlaHelpers::ShapeOfXlaOp(input).element_type();
    input = xla::ConvertElementType(input, xla::PrimitiveType::F16);
    xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
    xla::Shape out_shape = xla::ShapeUtil::MakeTupleShape({
        input_shape,
        xla::ShapeUtil::MakeShape(xla::PrimitiveType::U8,
                                  input_shape.dimensions()),
    });
    std::string shape_proto;
    input_shape.ToProto().SerializeToString(&shape_proto);
    // absl::Span<
    //     const std::pair<xla::ShapeIndex, std::pair<int64_t,
    //     xla::ShapeIndex>>> alias({std::pair<xla::ShapeIndex,
    //     std::pair<int64_t, xla::ShapeIndex>>{
    //       xla::ShapeIndex{0}, std::pair<int64_t, xla::ShapeIndex>(0,
    //       {0})}});
    // std::cout << "build" << std::endl;
    xla::XlaOp outputs = xla::CustomCall(
        loctx->builder(), "XlaCustomDropoutCuda", /*operands=*/{input},
        /*shape=*/out_shape, shape_proto, /*has_side_effect=*/true,
        /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/xla::API_VERSION_STATUS_RETURNING);
    // std::cout << "done" << std::endl;
    std::vector<xla::XlaOp> ops = {
        xla::ConvertElementType(xla::GetTupleElement(outputs, 0), type),
        xla::GetTupleElement(outputs, 1)};
    // std::vector<xla::XlaOp> ops = {xla::GetTupleElement(outputs, 0),
    //                                xla::GetTupleElement(outputs, 1)};
    return node.ReturnOps(ops, loctx);
  };
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp ones =
        xla::ConvertElementType(operands[0], xla::PrimitiveType::U8);
    return xla::Tuple(operands[0].builder(), {operands[0], ones});
  };
  return GenericOp(
      xla_dropout, {input},
      [&]() {
        return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
      },
      std::move(lower_fn), /*num_outputs=*/2);
}

torch::lazy::NodePtr DropoutBackward(const torch::lazy::Value& input,
                                     const torch::lazy::Value& mask,
                                     const torch::lazy::Value& scale) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp mask = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp scale = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp xla_output =
        xla_input *
        xla::ConvertElementType(
            mask, XlaHelpers::ShapeOfXlaOp(xla_input).element_type()) *
        scale;
    return node.ReturnOp(xla_output, loctx);
  };

  return GenericOp(xla_dropout_backward, {input, mask, scale},
                   GetXlaShape(input), std::move(lower_fn));
}

}  // namespace torch_xla
