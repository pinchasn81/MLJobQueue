# ML Training Job Queue - MLOps Home Assignment

## Table of Contents

1. [Product Context](#product-context)
  - [The ML Training Challenge](#the-ml-training-challenge)
  - [Real-World Example](#real-world-example)
2. [Assignment: TaskQueue](#assignment-taskqueue)
  - [What Makes This Different from Standard Queues?](#what-makes-this-different-from-standard-queues)
3. [Task Specification](#task-specification)
  - [Class Interface](#class-interface)
  - [Method Specifications](#method-specifications)
4. [Important Requirements](#important-requirements)
5. [Getting Started](#getting-started)
6. [Submission Guidelines](#submission-guidelines)

---

## Product Context

At our company, we run hundreds of ML model training jobs daily on our compute cluster. These jobs vary from quick hyperparameter sweeps to multi-hour model training sessions. Managing these jobs efficiently is critical for cost optimization and developer productivity.

### The ML Training Challenge

When data scientists submit training jobs to our platform, we need to:

1. **Queue jobs** with different priorities (urgent experiments vs. routine retraining)
2. **Assign jobs** to available workers in the compute pool
3. **Handle failures** gracefully with automatic retry logic
4. **Monitor performance** to optimize resource utilization
5. **Track job status** for observability and debugging

**The Challenge:** Build a task queue system that handles job lifecycle management, retry logic, and worker assignment efficiently.

### Real-World Example

```
Training Jobs Submitted:
- job_001: "train_sentiment_model" (NORMAL priority, max_retries=2)
- job_002: "critical_fraud_model" (HIGH priority, max_retries=3)
- job_003: "routine_rerank_model" (LOW priority, max_retries=1)

Worker Pool:
- worker_1: IDLE
- worker_2: IDLE

System Behavior:
1. job_002 (HIGH) assigned to worker_1 → status: RUNNING
2. job_001 (NORMAL) assigned to worker_2 → status: RUNNING
3. job_003 (LOW) waits in queue → status: PENDING
4. job_002 fails → retry_count=1, backoff=2s → status: RETRYING
5. job_001 completes → status: COMPLETED
6. job_003 assigned to worker_2 → status: RUNNING
```

---

## Assignment: TaskQueue

In this assignment, you will implement a task queue system for managing ML training jobs.

### What Makes This Different from Standard Queues?

Unlike simple FIFO queues, this system must:

- Handle **priority-based scheduling** (HIGH, NORMAL, LOW)
- Implement **intelligent retry logic** with exponential backoff
- Manage **worker states** (IDLE, BUSY, FAILED)
- Track **comprehensive metrics** (completion rate, retry statistics, worker utilization)
- Support **task status queries** for monitoring

---

## Task Specification

Implement the `TaskQueue` class in the provided `task_queue.py` file.

### Class Interface

```python
class TaskQueue:
    def __init__(self, num_workers: int)
    def submit_task(self, task_id: str, priority: Priority, max_retries: int = 3) -> None
    def assign_tasks(self) -> List[Tuple[str, int]]
    def complete_task(self, task_id: str, success: bool) -> None
    def get_task_status(self, task_id: str) -> TaskStatus
    def get_stats(self) -> Dict[str, float]
    def num_pending(self) -> int
    def num_running(self) -> int
```

### Method Specifications

#### 1. `__init__(num_workers)`

Initialize the task queue with a worker pool.

**Arguments:**

- `num_workers` (int): Number of workers in the compute pool (e.g., 4, 8, 16)

**Returns:** None

---

#### 2. `submit_task(task_id, priority, max_retries)`

Submit a new training job to the queue.

**Arguments:**

- `task_id` (str): Unique identifier for the task (e.g., "train_model_v2")
- `priority` (Priority): One of `Priority.HIGH`, `Priority.NORMAL`, `Priority.LOW`
- `max_retries` (int): Maximum number of retry attempts after initial failure (default: 3)

**Returns:** None

---

#### 3. `assign_tasks()`

Assign pending tasks to idle workers, respecting priority order (HIGH > NORMAL > LOW) and FIFO within the same priority. Return the list of assignments made.

**Returns:** `List[Tuple[str, int]]` - List of (task_id, worker_id) assignments made

---

#### 4. `complete_task(task_id, success)`

Mark a running task as completed or failed. On success, the task is done. On failure, retry the task if retries remain (using exponential backoff: `2^retry_count` seconds), otherwise mark it as permanently failed. Free up the worker either way.

**Arguments:**

- `task_id` (str): ID of the task that finished
- `success` (bool): `True` if task succeeded, `False` if it failed

**Returns:** None

---

#### 5. `get_task_status(task_id)`

Query the current status of a task.

**Arguments:**

- `task_id` (str): Task identifier

**Returns:** `TaskStatus` enum value

---

#### 6. `get_stats()`

Get queue performance statistics.

**Returns:** `Dict[str, float]` with the following keys:

- `total_submitted` - Total tasks submitted to queue
- `total_completed` - Tasks that completed successfully
- `total_failed` - Tasks that permanently failed
- `completion_rate` - Success rate among finished tasks (0-1)
- `avg_retries` - Average number of retries across all submitted tasks
- `worker_utilization` - How much of your worker pool is currently in use (0-1)

---

#### 7. `num_pending()` and `num_running()`

Helper methods to check queue state.

- `num_pending()`: Returns `int` - Number of tasks waiting to be assigned (including those queued for retry)
- `num_running()`: Returns `int` - Number of tasks currently executing

---

## Important Requirements

### Performance

Your implementation should scale efficiently to hundreds of tasks. Think about what data structures best support the operations you need.

### Retry Logic

Failed tasks should be retried with **exponential backoff** (`2^retry_count` seconds). You don't need to actually sleep/wait — just track the retry count. When a task retries, it goes to the **back of its priority queue**.

### Priority Handling

- **HIGH** priority tasks must always be assigned before NORMAL/LOW
- Within same priority, use **FIFO** (first-submitted gets assigned first)

### State Management

**Valid State Transitions:**

```
PENDING → RUNNING → COMPLETED
PENDING → RUNNING → RETRYING → ... → COMPLETED
PENDING → RUNNING → RETRYING → ... → FAILED
```

---

## Getting Started

### Files Provided

1. `**task_queue.py`** - Template file with method signatures (implement this!)

---

## Submission Guidelines

Submit your solution with following:

1. **Completed `task_queue.py`** with your implementation
2. **Tests** — a test file (e.g., `test_task_queue.py`) with meaningful test coverage beyond the provided sanity check

**Resources:** You may use any documentation, tutorials, or AI assistants (ChatGPT, Claude, etc.) to help with implementation. We're evaluating your problem-solving approach, not memorization.

---

Remember: **Working code > Perfect code**

Good luck!