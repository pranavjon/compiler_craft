#include "CC_ALG/Dialect.h"
#include "CC_ALG/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry; // DialectRegistry is a class that holds a collection of dialects and their associated information. It is used to manage the registration and lookup of dialects in MLIR. In this code, we create an instance of DialectRegistry called registry, which will be used to register the dialects needed for the CompilerCraft optimizer tool.
  
  // Register all dialects your tool needs
  registry.insert<mlir::cc_alg::CC_Alg_Dialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();

  // Register your pass
  mlir::cc_alg::registerLowerToAffinePass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CompilerCraft optimizer\n", registry));
}