#include "CC_ALG/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "CC_ALG/Ops.cpp.inc"

namespace mlir {
namespace cc_alg {

void CC_Alg_Dialect::initialize() {
  addOperations<
    MatMulOp,
    AddOp,
    ReluOp
  >();
}

mlir::LogicalResult MatMulOp::verify() {

  auto lhsType = mlir::cast<mlir::RankedTensorType>(getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(getRhs().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitOpError("operands must be 2D tensors");

  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return emitOpError("inner dimensions must match: ")
           << lhsType.getDimSize(1) << " vs " << rhsType.getDimSize(0);

  return mlir::success();
}

mlir::LogicalResult AddOp::verify() {
  
  auto lhsType = mlir::cast<mlir::RankedTensorType>(getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(getRhs().getType());

  if (lhsType.getRank() != rhsType.getRank())
    return emitOpError("operands must have the same rank");

  for (int i = 0; i < lhsType.getRank(); ++i) {
    if (lhsType.getDimSize(i) != rhsType.getDimSize(i))
      return emitOpError("operands must have the same shape");
  }

  return mlir::success();
}

CC_Alg_Dialect::CC_Alg_Dialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context,
                    mlir::TypeID::get<CC_Alg_Dialect>()) {
  initialize();
}

} // namespace cc_alg
} // namespace mlir