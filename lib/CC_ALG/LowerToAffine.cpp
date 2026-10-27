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

struct ReluLoweringPattern : public mlir::OpRewritePattern<mlir::cc_alg::ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::cc_alg::ReluOp op, mlir::PatternRewriter &rewriter) const override;
};

struct AddLoweringPattern: public mlir::OpRewritePattern<mlir::cc_alg::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::cc_alg::AddOp op, mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult ReluLoweringPattern::matchAndRewrite(mlir::cc_alg::ReluOp op, mlir::PatternRewriter &rewriter) const  {
  auto loc = op.getLoc();
  auto inputType = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
  int64_t M = inputType.getDimSize(0);
  int64_t N = inputType.getDimSize(1);
  auto inputMemRef = mlir::MemRefType::get({M,N} , inputType.getElementType());
  auto resultMemRef = mlir::MemRefType::get({M,N} , inputType.getElementType());
  auto inputBuffer = rewriter.create<mlir::memref::AllocOp>(loc, inputMemRef);
  auto resultBuffer = rewriter.create<mlir::memref::AllocOp>(loc, resultMemRef);

  auto zero = rewriter.create<mlir::arith::ConstantFloatOp>(
    loc, llvm::APFloat(0.0), rewriter.getF64Type());

  auto loopM = rewriter.create<mlir::affine::AffineForOp>(loc, 0, M, 1);
  rewriter.setInsertionPointToStart(loopM.getBody());
  auto i = loopM.getInductionVar();

  auto loopN = rewriter.create<mlir::affine::AffineForOp>(loc, 0, N, 1);
  rewriter.setInsertionPointToStart(loopN.getBody());
  auto j = loopN.getInductionVar();

  auto inputVal = rewriter.create<mlir::affine::AffineLoadOp>(loc, inputBuffer, mlir::ValueRange{i,j});

  auto reluVal = rewriter.create<mlir::arith::MaximumFOp>(loc, inputVal, zero);

  rewriter.create<mlir::affine::AffineStoreOp>(loc, reluVal, resultBuffer, mlir::ValueRange{i,j});

  rewriter.setInsertionPointAfter(loopM);
  rewriter.replaceOp(op, resultBuffer);
  return mlir::success();
}

mlir::LogicalResult AddLoweringPattern::matchAndRewrite(mlir::cc_alg::AddOp op, mlir::PatternRewriter &rewriter) const  {
  auto loc = op.getLoc();
  auto lhsType = mlir::cast<mlir::RankedTensorType>(op.getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(op.getRhs().getType());
  int64_t M = lhsType.getDimSize(0);
  int64_t N = lhsType.getDimSize(1);

  auto lhsMemRef = mlir::MemRefType::get({M,N}, lhsType.getElementType());
  auto rhsMemRef = mlir::MemRefType::get({M,N}, rhsType.getElementType());
  auto resultMemRef = mlir::MemRefType::get({M,N}, lhsType.getElementType());

  auto lhsBuffer = rewriter.create<mlir::memref::AllocOp>(loc, lhsMemRef);
  auto rhsBuffer = rewriter.create<mlir::memref::AllocOp>(loc, rhsMemRef);
  
  auto resultBuffer = rewriter.create<mlir::memref::AllocOp>(loc, resultMemRef);

  auto loopM = rewriter.create<mlir::affine::AffineForOp>(loc, 0, M, 1);
  rewriter.setInsertionPointToStart(loopM.getBody());
  auto i = loopM.getInductionVar();

  auto loopN = rewriter.create<mlir::affine::AffineForOp>(loc, 0, N, 1);
  rewriter.setInsertionPointToStart(loopN.getBody());
  auto j = loopN.getInductionVar();

  auto lhsVal = rewriter.create<mlir::affine::AffineLoadOp>(loc, lhsBuffer, mlir::ValueRange{i,j});
  auto rhsVal = rewriter.create<mlir::affine::AffineLoadOp>(loc, rhsBuffer, mlir::ValueRange{i,j});
  auto addVal = rewriter.create<mlir::arith::AddFOp>(loc, lhsVal, rhsVal);

  rewriter.create<mlir::affine::AffineStoreOp>(loc, addVal, resultBuffer, mlir::ValueRange{i,j});

  rewriter.setInsertionPointAfter(loopM);
  rewriter.replaceOp(op, resultBuffer);
  return mlir::success();

}

mlir::LogicalResult MatMulLoweringPattern::matchAndRewrite(
    mlir::cc_alg::MatMulOp op,
    mlir::PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto lhsType = mlir::cast<mlir::RankedTensorType>(op.getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(op.getRhs().getType());
  int64_t M = lhsType.getDimSize(0);
  int64_t K = lhsType.getDimSize(1);
  int64_t N = rhsType.getDimSize(1);

  auto refType = mlir::MemRefType::get({M, N}, lhsType.getElementType());
  auto result = rewriter.create<mlir::memref::AllocOp>(loc, refType);
  auto lhsMemRef = mlir::MemRefType::get({M, K}, lhsType.getElementType());
  auto rhsMemRef = mlir::MemRefType::get({K, N}, rhsType.getElementType());
  auto lhsBuffer = rewriter.create<mlir::memref::AllocOp>(loc, lhsMemRef);
  auto rhsBuffer = rewriter.create<mlir::memref::AllocOp>(loc, rhsMemRef);

  auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

  auto loopM = rewriter.create<mlir::affine::AffineForOp>(loc, 0, M, 1);
  rewriter.setInsertionPointToStart(loopM.getBody());
  auto i = loopM.getInductionVar();

  auto loopN = rewriter.create<mlir::affine::AffineForOp>(loc, 0, N, 1);
  rewriter.setInsertionPointToStart(loopN.getBody());
  auto j = loopN.getInductionVar();

  auto loopK = rewriter.create<mlir::affine::AffineForOp>(loc, 0, K, 1);
  rewriter.setInsertionPointToStart(loopK.getBody());
  auto k = loopK.getInductionVar();

  auto lhsVal = rewriter.create<mlir::affine::AffineLoadOp>(
      loc, lhsBuffer, mlir::ValueRange{i, k});
  auto rhsVal = rewriter.create<mlir::affine::AffineLoadOp>(
      loc, rhsBuffer, mlir::ValueRange{k, j});
  auto accVal = rewriter.create<mlir::affine::AffineLoadOp>(
      loc, result, mlir::ValueRange{i, j});
  auto mulVal = rewriter.create<mlir::arith::MulFOp>(loc, lhsVal, rhsVal);
  auto addVal = rewriter.create<mlir::arith::AddFOp>(loc, accVal, mulVal);
  rewriter.create<mlir::affine::AffineStoreOp>(
      loc, addVal, result, mlir::ValueRange{i, j});

  rewriter.setInsertionPointAfter(loopM);
  rewriter.replaceOp(op, result);
  return mlir::success();
}

struct LowerToAffinePass
    : public mlir::PassWrapper<LowerToAffinePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToAffinePass)

  llvm::StringRef getArgument() const override { return "lower-to-affine"; }
  llvm::StringRef getDescription() const override {
    return "Lower cc_alg dialect to affine loops";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mlir::affine::AffineDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addIllegalDialect<mlir::cc_alg::CC_Alg_Dialect>();

    mlir::RewritePatternSet patterns(context);
    patterns.add<MatMulLoweringPattern>(context);
    patterns.add<ReluLoweringPattern>(context);
    patterns.add<AddLoweringPattern>(context);

    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::cc_alg::createLowerToAffinePass() {
  return std::make_unique<LowerToAffinePass>();
}

void mlir::cc_alg::registerLowerToAffinePass() {
  mlir::PassRegistration<LowerToAffinePass>();
}