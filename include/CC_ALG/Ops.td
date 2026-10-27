#ifndef CC_ALG_OPS
#define CC_ALG_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// this registers your cc_alg dialect with MLIR dialect registry
def CC_Alg_Dialect : Dialect {
    let name = "cc_alg";  // prefix we see in the IR output
    let cppNamespace = "::mlir::cc_alg"; // this tells tablegen what c++ namespace to generate the classes in
}


// base class
class CC_ALG_Op<string mnemonic, list<Trait> traits = []> : Op<CC_Alg_Dialect, mnemonic, traits>; 

def MatMulOp : CC_ALG_Op<"matmul", [Pure]> {
    let summary = " matrix multiplication operation";
    let description = [{
        Performs Matrix Multiplication of two 2D tensors
        Inner dimensions must match: (M x K) * (K x N) = (M x N)

        Example:
    ```mlir
        %result = cc_alg.matmul %lhs, %rhs :
                (tensor<4x8xf64>, tensor<8x4xf64>) -> tensor<4x4xf64>
    ```}];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let builders = [OpBuilder< (ins "Value":$lhs, "Value":$rhs)>]; // tells MLIR how to construct your Op from C++ code.
    let hasVerifier = 1; // tells MLIR that i will write a C++ verify() for this Op.
}

def AddOp: CC_ALG_Op<"add", [Pure]> {
    let summary = " Element wise Add operation ";
    let description = [{
        Performs Add operation on two tensors
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
    let builders = [OpBuilder< (ins "Value":$lhs, "Value":$rhs)>];
    let hasVerifier = 1;
}

def ReluOp: CC_ALG_Op<"relu", [Pure]> {
    let summary = " Relu activation operation";
    let description = [{
        Performs RELU activation operation on an input tensor
    }];
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);
    let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
    let builders = [OpBuilder< (ins "Value":$input)>];
    let hasVerifier = 0; 
}

#endif // CC_ALG_OPS