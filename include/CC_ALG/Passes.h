#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cc_alg {

std::unique_ptr<mlir::Pass> createLowerToAffinePass();

// Registration
void registerLowerToAffinePass();

} // namespace cc_alg
} // namespace mlir