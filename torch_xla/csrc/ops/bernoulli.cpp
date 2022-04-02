#include "torch_xla/csrc/ops/bernoulli.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Bernoulli::Bernoulli(const Value& probability, const Value& seed,
                     xla::Shape shape)
    : Node(torch::lazy::OpKind(at::aten::bernoulli), {probability, seed},
           std::move(shape)) {}

torch::lazy::NodePtr Bernoulli::Clone(OpList operands) const {
  return ir::MakeNode<Bernoulli>(operands.at(0), operands.at(1), xla_shape());
}

XlaOpVector Bernoulli::Lower(LoweringContext* loctx) const {
  xla::XlaOp probability = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(1));
  const xla::Shape& probability_shape = XlaHelpers::ShapeOfXlaOp(probability);
  xla::Shape bcast_shape(xla_shape());
  bcast_shape.set_element_type(probability_shape.element_type());
  xla::XlaOp bcast_probability = XlaHelpers::ImplicitBroadcast(
      probability, probability_shape, bcast_shape);
  return ReturnOp(
      BuildBernoulli(bcast_probability, rng_seed, xla_shape().element_type()),
      loctx);
}

BernoulliCuda::BernoulliCuda(const Value& probability, xla::Shape shape)
    : Node(torch::lazy::OpKind(at::aten::bernoulli), {probability},
           std::move(shape)) {}

NodePtr BernoulliCuda::Clone(OpList operands) const {
  return MakeNode<BernoulliCuda>(operands.at(0), shape());
}

// XlaOpVector BernoulliCuda::Lower(LoweringContext* loctx) const {
//   xla::XlaOp probability = loctx->GetOutputOp(operand(0));
//   const xla::Shape& probability_shape = XlaHelpers::ShapeOfXlaOp(probability);
//   xla::Shape rng_shape(shape());
//   rng_shape.set_element_type(xla::PrimitiveType::F32);
//   xla::XlaOp bcast_probability =
//       XlaHelpers::ImplicitBroadcast(probability, probability_shape, rng_shape);

//   std::string func_name = "custom_curand_uniform";
//   std::string rng_shape_proto;
//   rng_shape.ToProto().SerializeToString(&rng_shape_proto);
//   // absl::Span<const std::pair<xla::ShapeIndex, std::pair<int64_t, xla::ShapeIndex>>> aliasing({{xla::ShapeIndex({}), {0, xla::ShapeIndex()}}});
//   xla::XlaOp rng =
//       xla::CustomCall(loctx->builder(), func_name, /*operands=*/{},
//                       /*shape=*/rng_shape, rng_shape_proto, false,
//                       /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
//                       /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE,
//                       /*api_version=*/xla::API_VERSION_STATUS_RETURNING);
//   rng = xla::ConvertElementType(rng, shape().element_type());
//   xla::XlaOp res = xla::Lt(rng, probability);
//   return ReturnOp(res, loctx);
// }

XlaOpVector BernoulliCuda::Lower(LoweringContext* loctx) const {
  xla::XlaOp probability = loctx->GetOutputOp(operand(0));
  probability = xla::ConvertElementType(probability, xla::PrimitiveType::F32);
  const xla::Shape& probability_shape = XlaHelpers::ShapeOfXlaOp(probability);
  xla::Shape rng_shape(shape());
  rng_shape.set_element_type(xla::PrimitiveType::F32);
  xla::XlaOp bcast_probability =
      XlaHelpers::ImplicitBroadcast(probability, probability_shape, rng_shape);

  std::string func_name = "custom_cuda_bernoulli";
  std::string rng_shape_proto;
  rng_shape.ToProto().SerializeToString(&rng_shape_proto);
  xla::XlaOp res =
      xla::CustomCall(loctx->builder(), func_name, /*operands=*/{probability},
                      /*shape=*/rng_shape, rng_shape_proto, false,
                      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                      /*schedule=*/xla::CustomCallSchedule::SCHEDULE_LATEST,
                      /*api_version=*/xla::API_VERSION_STATUS_RETURNING);
  return ReturnOp(xla::ConvertElementType(res, shape().element_type()), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
