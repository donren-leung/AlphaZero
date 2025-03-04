from batching.GameWorker import mpQueueGen
from batching.NodeBatch import NodeBatchRequest, NodeBatchResponse
from MCTS_batch import simulate_

class CPUPool(object):
    def __init__(self, inbox: mpQueueGen[NodeBatchRequest],
                 outboxes: list[mpQueueGen[NodeBatchResponse]]):
        self.inbox = inbox
        self.outboxes = outboxes

    def run(self) -> None:
        while True:
            request = self.inbox.get()
            worker_id, thread_id = request.worker_id, request.thread_id
            results = [simulate_(state,
                                 request.curr_player,
                                 action,
                                 target_sims=request.target_sims)
                       for action, state
                       in request.action_and_state]

            response = NodeBatchResponse(worker_id, thread_id, results)
            self.outboxes[worker_id].put(response)
