ifndef MATMUL_OPS
#define MATMUL_OPS

#include "mlir/IR/OpBase.td"
#include "mlir/Interfaces/SideEffectInterfaces.td"


// this registers your MATMUL dialect with MLIR dialect registry
def MatMul_Dialect : Dialect {
    let name = "matmul";  // prefix we see in the IR output
    let cppNamespace = "::mlir::matmul"; // this tells tablegen what c++ namespace to generate the classes in
}


// base class
class MatMul_Op<string mnemonic, list<Trait> traits = []> : Op<MatMul_Dialect, mnemonic, traits>; 



#endif MATMUL_OPS