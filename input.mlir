module  {
  func.func @foo(%arg0: i32) -> i32 {
    %0 = arith.index_cast %arg0: i32 to index
    %c0_i32 = arith.constant 0: i32
    %c1_i32 = arith.constant 1: i32
    %c2_i32 = arith.constant 2: i32
    %1 = scf.index_switch %0 -> i32
    case 0 {
      scf.yield %c0_i32: i32
    }
    default {
      scf.yield %c1_i32: i32
    }
    %2 = arith.shli %1, %c2_i32: i32
    return %2: i32
  }
}

// should expect the following output
// func.func @foo(%arg0: i32) -> i32 {
//   %c0_i32 = arith.constant 0: i32
//   %c4_i32 = arith.constant 4: i32
//   %0 = arith.cmpi eq %arg0, %c0: i32
//   %1 = arith.select %0, %c0_i32, %c4_i32: i32
//   return %1: i32
// }
