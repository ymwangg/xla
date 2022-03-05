#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Bernoulli : public Node {
 public:
  Bernoulli(const Value& probability, const Value& seed, xla::Shape shape);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

class CustomBernoulli : public Node {
 public:
  CustomBernoulli(xla::Shape shape, float probability);
  CustomBernoulli(xla::Shape shape, const Value& probability);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  float probability_;
  bool has_probability_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
