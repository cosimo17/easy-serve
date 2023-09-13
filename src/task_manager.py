from multiprocessing import Queue, Process
import time
import random

__all__ = ['TaskScheduler']


class TaskScheduler(object):
    def __init__(self,
                 process_count,
                 gpus,
                 _handle,
                 max_size=50):
        """
        :param process_count: int. Process per GPU
        :param gpus: list[int]. GPUs ID
        :param _handle: ModelHandle. see handle.py
        :param max_size: int. max size of the task Queue and result Queue
        """
        self.pending_queue = Queue(maxsize=max_size)
        self.task_queue = [Queue(maxsize=max_size) for _ in range(process_count * len(gpus))]
        self.result = [Queue(maxsize=max_size) for _ in range(process_count * len(gpus))]
        self.workers = []
        n = 0
        for i in range(process_count):
            for gpu_id in gpus:
                p = Process(target=self.work_loop, args=(_handle, gpu_id, self.task_queue[n], self.result[n]))
                p.start()
                self.workers.append(p)
                n += 1
        self.results_cache = [{} for _ in range(len(self.workers))]

    def add_task(self, task):
        sizes = [q.qsize() for q in self.task_queue]
        least_loaded_worker = sizes.index(min(sizes))
        task_id = str(int(time.time() * 1e7)) + str(int(random.random() * 1e4))
        _task = {'task_id': task_id, 'task_info': task}
        self.task_queue[least_loaded_worker].put(_task)
        return least_loaded_worker, task_id

    def check_state(self, worker_id, task_id):
        while not self.result[worker_id].empty():
            result = self.result[worker_id].get()
            self.results_cache[worker_id][result['task_id']] = result
        return task_id in self.results_cache[worker_id]

    def get_result(self, worker_id, task_id):
        result = self.results_cache[worker_id][task_id]
        del self.results_cache[worker_id][task_id]
        return result

    def kill_all(self):
        for q in self.task_queue:
            q.put(None)
        for p in self.workers:
            p.join()
        print("Killed")

    @staticmethod
    def work_loop(task__handle, gpu_id, task_queue, result_queue):
        _handle = task__handle(gpu_id)
        while True:
            task = task_queue.get()
            if task is None:
                break
            result = _handle(task)
            result_queue.put(result)


class Fake_handle(object):
    def __init__(self, gpu_id):
        pass

    def __call__(self, task):
        task['state'] = 'finish'
        return task


def test():
    ts = TaskScheduler(4, 1, _handle=Fake_handle, max_size=20)
    for i in range(20):
        task = {i: i}
        worker_id, task_id = ts.add_task(task)
        while True:
            state = ts.check_state(worker_id, task_id)
            if state:
                result = ts.get_result(worker_id, task_id)
                assert result['task_id'] == task_id
                assert result['task_info'] == task
                assert result['state'] == 'finish'
                break
            time.sleep(0.01)
    ts.kill_all()


if __name__ == '__main__':
    test()
