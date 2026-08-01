#include "CC_ALG/Dialect.h"
#include "CC_ALG/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);
  registry.insert<mlir::cc_alg::CC_Alg_Dialect>();
  mlir::cc_alg::registerLowerToAffinePass();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CompilerCraft optimizer\n", registry));
}