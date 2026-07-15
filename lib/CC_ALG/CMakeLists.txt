add_mlir_library(CCALG
  Ops.cpp

  DEPENDS
  MLIRCCALGOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)