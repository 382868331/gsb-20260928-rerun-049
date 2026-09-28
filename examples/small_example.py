from optplatform import (
    BranchAndBound,
    Constraint,
    Problem,
    SearchBudget,
    SearchStore,
    Variable,
)


problem = Problem(
    name="small-bounded-ilp",
    variables=[Variable("x", 0, 2), Variable("y", 0, 2)],
    objective={"x": 5, "y": 4},
    constraints=[Constraint("capacity", {"x": 2, "y": 1}, 3)],
)

store = SearchStore("runs/small-bounded-ilp")
store.save_problem(problem)

solver = BranchAndBound(problem, checkpoint_callback=store.save_state)
partial = solver.solve(SearchBudget(max_nodes=3))
store.save_result(partial)

state = store.load_state()
final = solver.solve(state=state)
store.save_result(final)
store.save_proof(problem, state, final)

print(final.to_dict())
