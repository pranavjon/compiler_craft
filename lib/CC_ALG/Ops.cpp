#include "CC_ALG/Dialect.h"
#include "CC_ALG/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace cc_alg {

void CC_Alg_Dialect::initialize() {
    addOperations<
      MatMulOp
    >();
}

// mlir::logicalResult is MLIR's way of representing success/failure in a more expressive manner than just returning a boolean. It allows for better error handling and reporting in the context of MLIR operations.
mlir::LogicalResult MatMulOp::verify() {
    auto lhsType = getLhs().getType().cast<mlir::RankedTensorType>();
    auto rhsType = getRhs().getType().cast<mlir::RankedTensorType>();

    if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
        return emitOpError("operands must be 2D tensors");

    if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
        return emitOpError("inner dimensions must match: ")
            << lhsType.getDimSize(1) << " vs " << rhsType.getDimSize(0);

  return mlir::success();
}

} // namespace cc_alg
} // namespace mlir