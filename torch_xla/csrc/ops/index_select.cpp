#include "torch_xla/csrc/ops/index_select.h"

#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& index,
                           int64_t dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::TorchIndexSelect(operands[0], operands[1], dim);
  };
  return InferOutputShape({input.shape(), index.shape()}, lower_for_shape_fn);
}

}  // namespace

IndexSelect::IndexSelect(const Value& input, int64_t dim, const Value& index)
    : Node(ir::OpKind(at::aten::index_select), {input, index},
           [&]() { return NodeOutputShape(input, index, dim); },
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

NodePtr IndexSelect::Clone(OpList operands) const {
  return MakeNode<IndexSelect>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector IndexSelect::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  bool use_custom_select =
      xla::sys_util::GetEnvBool("XLA_CUSTOM_SELECT", false);
  if (use_custom_select) {
    const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
    const xla::Shape& index_shape = XlaHelpers::ShapeOfXlaOp(index);
    std::vector<int64_t> dim;
    for (int i = 0; i < index_shape.rank(); i++) {
      dim.push_back(index_shape.dimensions(i));
    }
    dim.push_back(input_shape.dimensions(1));
    // xla::XlaOp indices = xla::Zeros(
    //     input.builder(),
    //     xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, dim));
    // return ReturnOp(indices, loctx);
    std::string opaque;
    xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, dim);
    shape.ToProto().SerializeToString(&opaque);
    xla::XlaOp custom_call =
        xla::CustomCall(loctx->builder(), "do_custom_call", /*operands=*/{input, index},
                        /*shape=*/shape, opaque, false, /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                        /*schedule=*/xla::CustomCallSchedule::SCHEDULE_EARLIEST);
    return ReturnOp(custom_call, loctx);
  }
  else{
    return ReturnOp(xla::TorchIndexSelect(input, index, dim_), loctx);
  }
}

std::string IndexSelect::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
