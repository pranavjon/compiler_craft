#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace cc_alg {

class CC_Alg_Dialect : public mlir::Dialect {
public:
  explicit CC_Alg_Dialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "cc_alg"; }
  void initialize();
};

} // namespace cc_alg
} // namespace mlir

// Include tablegen generated op declarations
#define GET_OP_CLASSES
#include "CC_ALG/Ops.h.inc"