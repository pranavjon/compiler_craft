add_mlir_library(CCALG
  Ops.cpp
  LowerToAffine.cpp

  DEPENDS
  MLIRCCALGOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransformUtils
  MLIRTransforms
)