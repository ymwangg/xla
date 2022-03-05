#include "torch_xla/csrc/ops/bernoulli.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Bernoulli::Bernoulli(const Value& probability, const Value& seed,
                     xla::Shape shape)
    : Node(ir::OpKind(at::aten::bernoulli), {probability, seed},
           std::move(shape)) {}

NodePtr Bernoulli::Clone(OpList operands) const {
  return MakeNode<Bernoulli>(operands.at(0), operands.at(1), shape());
}

XlaOpVector Bernoulli::Lower(LoweringContext* loctx) const {
  xla::XlaOp probability = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(1));
  const xla::Shape& probability_shape = XlaHelpers::ShapeOfXlaOp(probability);
  xla::Shape bcast_shape(shape());
  bcast_shape.set_element_type(probability_shape.element_type());
  xla::XlaOp bcast_probability = XlaHelpers::ImplicitBroadcast(
      probability, probability_shape, bcast_shape);
  return ReturnOp(
      BuildBernoulli(bcast_probability, rng_seed, shape().element_type()),
      loctx);
}

CustomBernoulli::CustomBernoulli(xla::Shape shape, const Value& probability)
    : Node(ir::OpKind(at::aten::bernoulli), {probability}, std::move(shape)),
      has_probability_(false) {}

CustomBernoulli::CustomBernoulli(xla::Shape shape, float probability)
    : Node(ir::OpKind(at::aten::bernoulli), {}, std::move(shape)),
      probability_(probability),
      has_probability_(true) {}

NodePtr CustomBernoulli::Clone(OpList operands) const {
  if (operands.size() > 0) {
    return MakeNode<CustomBernoulli>(shape(), probability_);
  }
  return MakeNode<CustomBernoulli>(shape(), operands.at(0));
}

XlaOpVector CustomBernoulli::Lower(LoweringContext* loctx) const {
  if (has_probability_) {
    xla::Shape new_shape(shape());
    new_shape.set_element_type(xla::PrimitiveType::F32);
    std::string opaque;
    new_shape.ToProto().SerializeToString(&opaque);
    std::string kernel_name = "xla_custom_bernoulli_fast";
    xla::XlaOp mask =
        xla::CustomCall(loctx->builder(), kernel_name, /*operands=*/{},
                        /*shape=*/new_shape, opaque, false,
                        /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE);
    return ReturnOp(xla::ConvertElementType(mask, shape().element_type()),
                    loctx);
  } else {
    xla::XlaOp probability = xla::ConvertElementType(
        loctx->GetOutputOp(operand(0)), xla::PrimitiveType::F32);

    xla::Shape probability_shape = XlaHelpers::ShapeOfXlaOp(probability);
    xla::Shape new_shape(shape());
    new_shape.set_element_type(probability_shape.element_type());
    xla::XlaOp bcast_probability = XlaHelpers::ImplicitBroadcast(
      probability, probability_shape, new_shape);
    std::string opaque;
    new_shape.ToProto().SerializeToString(&opaque);
    std::string kernel_name = "xla_custom_bernoulli";
    xla::XlaOp mask = xla::CustomCall(
        loctx->builder(), kernel_name, /*operands=*/{probability},
        /*shape=*/new_shape, opaque, false,
        /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_NONE);
    return ReturnOp(xla::ConvertElementType(mask, shape().element_type()),
                    loctx);
  }
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
