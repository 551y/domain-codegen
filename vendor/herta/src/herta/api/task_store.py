"""任务状态存储（进程内，带容量上限）"""

from collections import OrderedDict

# 容量上限的 FIFO 任务表：超出后淘汰最旧记录，避免长时间运行内存膨胀
_MAX_TASKS = 500


class TaskStore(OrderedDict):
    def __setitem__(self, key, value):
        if key not in self and len(self) >= _MAX_TASKS:
            self.popitem(last=False)
        super().__setitem__(key, value)

    def create(self, task_type: str) -> str:
        """创建任务记录并返回完整 uuid（避免短 id 碰撞覆盖）"""
        import uuid

        task_id = uuid.uuid4().hex
        self[task_id] = {
            "type": task_type,
            "status": "pending",
            "progress": 0,
            "message": "",
            "result": {},
            "error": None,
        }
        return task_id


tasks: TaskStore = TaskStore()
