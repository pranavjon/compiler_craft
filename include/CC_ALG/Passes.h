#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cc_alg {

std::unique_ptr<mlir::Pass> createLowerToAffinePass();

} // namespace cc_alg
} // namespace mlir