func.func @matmul_test(%lhs: tensor<4x8xf64>, %rhs: tensor<8x4xf64>) {
  %result = cc_alg.matmul %lhs, %rhs : (tensor<4x8xf64>, tensor<8x4xf64>) -> tensor<4x4xf64>
  return
}