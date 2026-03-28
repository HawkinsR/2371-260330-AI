# Lab: Human-in-the-Loop Approval Workflows

## The Scenario
Your company has developed an autonomous "Refund Bot". However, a bug recently caused it to issue a $10,000 refund for a $10 product. Management has mandated that all refunds over $50 must be manually approved by a human before processing. You will implement a LangGraph Breakpoint to pause the AI's execution right before the irreversible `ProcessRefund` action occurs, manually edit the graph's memory to simulate a human manager rejecting the bad refund, and then resume the graph to gracefully cancel the transaction.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e049-e049-approval_lab.py`.
3. Complete the Graph Compilation:
   - Instantiate the `MemorySaver` as the checkpointer.
   - Call `builder.compile()`. Pass the `checkpointer` argument, and critically, pass the `interrupt_before` argument specifying a list containing the two terminal nodes: `["ProcessRefund", "CancelRefund"]`.
4. Complete the Human Intervention phase:
   - Call `graph.get_state(config)` and print the currently proposed refund amount from the sleeping AI.
   - The AI hallucinates a $10,000 refund. You must overwrite this.
   - Call `graph.update_state()`. Pass the `config` dictionary, and a dictionary containing your state overrides: `{"is_approved": False, "refund_amount": 0}`.
5. Resume Execution:
   - Call `graph.invoke()` to wake the graph up.
   - Pass `None` as the first argument (to tell it to use the current suspended state), and pass your `config` as the second argument.

## Definition of Done
- The script executes successfully.
- The graph correctly pauses execution *before* processing the refund.
- The console outputs the human intervention phase demonstrating the state mutation.
- The graph resumes and successfully routes to the Cancelation node instead of the Processing node, preventing the hallucinated $10,000 refund from executing.
