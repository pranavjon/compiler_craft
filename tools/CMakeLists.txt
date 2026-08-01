add_llvm_executable(cc-opt cc-opt.cpp)

target_link_libraries(cc-opt PRIVATE
  CCALG
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMlirOptMain
  MLIRTransforms
  MLIRTransformUtils
)