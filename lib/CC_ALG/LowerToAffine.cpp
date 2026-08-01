#include "CC_ALG/Dialect.h"
#include "CC_ALG/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

struct MatMulLoweringPattern 
    : public mlir::OpRewritePattern<mlir::cc_alg::MatMulOp> {
  
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::cc_alg::MatMulOp op,
      mlir::PatternRewriter &rewriter) const override;
};

/*
Before: 
%out = cc_alg.matmul %lhs, %rhs

After:
%buff = memref.alloc [M X N]
affine.for %i = 0 to M{
  affine.for %j = 0 to N{
    affine.for %k = 0 to K{
        result[i][j] += lhs[i][k] * rhs[k][j]
    }
  }
}
*/

mlir::LogicalResult MatMulLoweringPattern::matchAndRewrite(
    mlir::cc_alg::MatMulOp op,
    mlir::PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto lhsType = mlir::cast<mlir::RankedTensorType>(op.getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(op.getRhs().getType());
  
  int64_t M = lhsType.getDimSize(0);
  int64_t K = lhsType.getDimSize(1);
  int64_t N = rhsType.getDimSize(1);

  // allocate output buffer Memory Bufferization (memref)
  auto refType = mlir::MemRefType::get({M, N}, lhsType.getElementType());
  auto result = rewriter.create<mlir::memref::AllocOp>(loc, refType);
  // allocate input buffer memory bufferization (memref)
  auto lhsMemRef = mlir::MemRefType::get({M, K}, lhsType.getElementType());
  auto rhsMemRef = mlir::MemRefType::get({K, N}, rhsType.getElementType());

  // creates buffers in IR
  auto lhsBuffer = rewriter.create<mlir::memref::AllocOp>(loc, lhsMemRef);
  auto rhsBuffer = rewriter.create<mlir::memref::AllocOp>(loc, rhsMemRef);

  // 3 nested loops to perform matrix multiplication

  // create M x N x K loops 
  auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

  // outer loop i : 0 to M
  auto loopM = rewriter.create<mlir::affine::AffineForOp>(loc, 0, M, 1); // creates affine.for %i = 0 to M step 1
  rewriter.setInsertionPointToStart(loopM.getBody()); // moves the insertion point inside the loop body. Everything you create after this line goes inside that loop.
  auto i = loopM.getInductionVar(); // gets the induction variable of the loop

  // Middle loop j : 0 to N
  auto loopN = rewriter.create<mlir::affine::AffineForOp>(loc, 0, N, 1);
  rewriter.setInsertionPointToStart(loopN.getBody());
  auto j = loopN.getInductionVar();

  // Inner loop k : 0 to K
  auto loopK = rewriter.create<mlir::affine::AffineForOp>(loc, 0, K, 1);
  rewriter.setInsertionPointToStart(loopK.getBody());
  auto k = loopK.getInductionVar();

  // load, multiply and accumulate
  auto lhsVal = rewriter.create<mlir::affine::AffineLoadOp>(loc, lhsBuffer, mlir::ValueRange{i, k});
  auto rhsVal = rewriter.create<mlir::affine::AffineLoadOp>(loc, rhsBuffer, mlir::ValueRange{k, j});
  auto accVal = rewriter.create<mlir::affine::AffineLoadOp>(loc, result, mlir::ValueRange{i, j});

  auto mulVal = rewriter.create<mlir::arith::MulFOp>(loc, lhsVal, rhsVal); // lhs[i][k] * rhs[k][j]
  auto addVal = rewriter.create<mlir::arith::AddFOp>(loc, accVal, mulVal); // acc + (lhs[i][k] * rhs[k][j])

  rewriter.create<mlir::affine::AffineStoreOp>(loc, addVal, result, mlir::ValueRange{i, j}); // store the result back to the output buffer

  // Move the insertion point back to the outer loop for the next iteration
  rewriter.setInsertionPointAfter(loopM);

  // Replace the original cc_alg.matmul operation with result buffer
  rewriter.replaceOp(op, result);

  return mlir::success();
}

struct LowerToAffinePass : public mlir::PassWrapper<LowerToAffinePass, mlir::OperationPass<mlir::func::FuncOp>> {

    void runOnOperation() override {
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<mlir::affine::AffineDialect>();
        target.addLegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<mlir::memref::MemRefDialect>();
        target.addLegalDialect<mlir::func::FuncDialect>();
        target.addIllegalDialect<mlir::cc_alg::CC_Alg_Dialect>();

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<MatMulLoweringPattern>(&getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}

std::unique_ptr<mlir::Pass> mlir::cc_alg::createLowerToAffinePass() {
  return std::make_unique<LowerToAffinePass>();
}

