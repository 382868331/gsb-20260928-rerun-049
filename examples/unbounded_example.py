from optplatform import (
    BranchAndBound,
    Constraint,
    Problem,
    SearchBudget,
    SearchStore,
    Variable,
)


# max x + y with x, y >= 0 and no upper bound: provably unbounded.
unbounded = Problem(
    name="free-ray",
    variables=[Variable("x", 0, None), Variable("y", 0, None)],
    objective={"x": 1, "y": 1},
    constraints=[],
)
result = BranchAndBound(unbounded).solve()
print(result.status.value, result.evidence)


# max 5x + 4y, 2x + y <= 3, x >= 0, 0 <= y <= 2: finite optimum x = y = 1.
capped = Problem(
    name="capped-ray",
    variables=[Variable("x", 0, None), Variable("y", 0, 2)],
    objective={"x": 5, "y": 4},
    constraints=[Constraint("capacity", {"x": 2, "y": 1}, 3)],
)
store = SearchStore("runs/capped-ray")
store.save_problem(capped)
solver = BranchAndBound(capped, checkpoint_callback=store.save_state)
partial = solver.solve(SearchBudget(max_nodes=2))
assert partial.status.value in ("best_known", "unsolved")
final = solver.solve(state=store.load_state())
store.save_result(final)
store.save_proof(capped, store.load_state(), final)
print(final.status.value, final.best_assignment, final.best_objective)

# Re-open the saved problem and proof and re-check the conclusion from disk.
report = store.review()
assert report.valid, report.problems
print("review:", report.valid)
