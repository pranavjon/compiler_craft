#pragma once 

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace cc_alg {

class CC_Alg_Dialect : public mlir::Dialect {
public:
    explicit CC_Alg_Dialect(MLIRContext *context);
    static StringRef getDialectNamespace() { return "cc_alg"; }
};

} // namespace cc_alg
} // namespace mlir