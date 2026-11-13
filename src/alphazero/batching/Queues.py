from multiprocessing.context import BaseContext
from typing import Generic, TypeVar

T = TypeVar("T")

class mpQueueGen(Generic[T]):
    """
    Generic wrapper around the multiprocessing Queue.
    """
    def __init__(self, ctx: BaseContext, *args, **kwargs):
        self._queue = ctx.Queue(*args, **kwargs)
        # Lock for consumers to ensure only one is waiting on the queue at a time
        self._consumer_lock = ctx.Lock()

    def put(self, obj: T, *args, **kwargs) -> None:
        try:
            self._queue.put(obj, *args, **kwargs)
        except Exception as e:
            raise e

    def get(self, *args, **kwargs) -> T:
        try:
            res = self._queue.get(*args, **kwargs)
        except Exception as e:
            raise e
        return res

    def consumer_lock(self):
        return self._consumer_lock

    def empty(self):
        return self._queue.empty()
