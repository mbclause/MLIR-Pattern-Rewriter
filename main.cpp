#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// docs: https://mlir.llvm.org/docs/PatternRewriter/
// arith dialect: https://mlir.llvm.org/docs/Dialects/ArithOps/
// scf dialect: https://mlir.llvm.org/docs/Dialects/SCFDialect/
// examples: https://github.com/llvm/llvm-project/tree/main/mlir/examples

namespace mlir {




class MyPattern : public OpRewritePattern<arith::ShLIOp> 
{
public:
  using OpRewritePattern<arith::ShLIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite
  (arith::ShLIOp shl, PatternRewriter &rewriter) const override 
  {
    // ----MATCH PHASE--- 
    // LHS and RHS of shli
    Value var1 = shl.getLhs();
    Value c2Val = shl.getRhs();

    // LHS must be i32
    if (!var1.getType().isInteger(32))
      return failure();

    // RHS must be a constant equal to 2
    auto const2 = c2Val.getDefiningOp<arith::ConstantOp>();

    if (!const2)
      return failure();

    auto c2Attr = dyn_cast<IntegerAttr>(const2.getValue());

    if (!c2Attr || !c2Attr.getType().isInteger(32) || c2Attr.getInt() != 2)
      return failure();

    // LHS must come from scf.index_switch
    auto iso = var1.getDefiningOp<scf::IndexSwitchOp>();

    if (!iso)
      return failure();

    // Exactly one explicit case (plus default)
    if (iso.getNumCases() != 1)
      return failure();

    // Read the single case and the default regions
    Region &case0 = iso.getCaseRegions().front();

    Region &defCase = iso.getDefaultRegion();

    if (!(case0.hasOneBlock() && defCase.hasOneBlock()))
      return failure();

    // Get yield ops
    auto *y1 = case0.front().getTerminator();

    auto *y2 = defCase.front().getTerminator();

    auto yield1 = dyn_cast<scf::YieldOp>(y1);

    auto yield2 = dyn_cast<scf::YieldOp>(y2);

    if (!yield1 || !yield2 || yield1.getNumOperands() != 1 || yield2.getNumOperands() != 1)
      return failure();

    // Extract yielded constants
    auto cA = yield1.getOperand(0).getDefiningOp<arith::ConstantOp>();

    auto cB = yield2.getOperand(0).getDefiningOp<arith::ConstantOp>();

    if (!cA || !cB)
      return failure();

    auto aAttr = dyn_cast<IntegerAttr>(cA.getValue());

    auto bAttr = dyn_cast<IntegerAttr>(cB.getValue());

    // make sure they're i32
    if (!aAttr || !bAttr || !aAttr.getType().isInteger(32) || !bAttr.getType().isInteger(32))
      return failure();

    int64_t a = aAttr.getInt();
    
    int64_t b = bAttr.getInt();

    // make sure they are equal to 0 and 1 or vice versa
    if (!((a == 0 && b == 1) || (a == 1 && b == 0)))
      return failure();

    // Switch operand must be i32 and the result of an arith.index_cast
    auto iCast = iso.getOperand().getDefiningOp<arith::IndexCastOp>();

    if (!iCast)
      return failure();

    Value cmpLHS = iCast.getOperand();

    if (!cmpLHS.getType().isInteger(32))
      return failure();

    

    // --- REWRITE ---
    // make sure we always insert at the right position
    OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPoint(shl);

    Type i32Type = rewriter.getIntegerType(32);

    // create the new constants c0 and c4 and the new compare and select ops 
    auto newC0 = rewriter.create<arith::ConstantOp>
    (shl.getLoc(), i32Type, rewriter.getIntegerAttr(i32Type, 0));

    auto newC4 = rewriter.create<arith::ConstantOp>
    (shl.getLoc(), i32Type, rewriter.getIntegerAttr(i32Type, 4));

    auto newCmpi = rewriter.create<arith::CmpIOp>
    (shl.getLoc(), arith::CmpIPredicate::eq, cmpLHS, newC0.getResult());

    auto newSelect = rewriter.create<arith::SelectOp>
    (shl.getLoc(), newCmpi.getResult(), newC0.getResult(), newC4.getResult());

    // Replace the ROOT op, then clean up if dead
    rewriter.replaceOp(shl, newSelect.getResult());

    if (iso->use_empty())
      rewriter.eraseOp(iso);

    if (iCast->use_empty())
      rewriter.eraseOp(iCast);

    return success();
  }
};


class InstCombinePass
    : public PassWrapper<InstCombinePass, OperationPass<ModuleOp>> 
{
public:
  StringRef getArgument() const final { return "instcombine"; }

  StringRef getDescription() const final {
    return "A simple pass to optimize some scf and arith operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
  }

  void runOnOperation() override 
  {

    ModuleOp module = getOperation();

    // loop through all the functions in the module
    for (auto func : module.getOps<func::FuncOp>()) 
    {
      RewritePatternSet patterns(&getContext());

      // add the custom pattern, MyPattern, to the set
      patterns.add<MyPattern>(&getContext());

      // apply pattern to input
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
        signalPassFailure();
    }
  }
};
} // namespace mlir


int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::PassRegistration<mlir::InstCombinePass>();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Custom optimizer driver\n", registry));
}
