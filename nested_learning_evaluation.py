"""
Nested Learning Evaluation - Đánh giá hiệu suất với các chỉ số Continual Learning
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

np.random.seed(42)

# =============================================================================
# DATA & MODELS (Reused from demo)
# =============================================================================

@dataclass
class Task:
    name: str
    X: np.ndarray
    y: np.ndarray

def generate_task_data(task_id: int, n_samples: int = 200) -> Task: # Tăng sample để đánh giá chính xác hơn
    np.random.seed(task_id)
    if task_id == 0:
        X = np.random.randn(n_samples, 2)
        y = (2 * X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
        name = "Linear"
    elif task_id == 1:
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
        name = "XOR"
    else:
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0]**2 + X[:, 1]**2) < 1).astype(float)
        name = "Circular"
    return Task(name=name, X=X, y=y)

class SimpleNetwork:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return 1 / (1 + np.exp(-self.z2))

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, lr: float = 0.1):
        m = X.shape[0]
        dz2 = y_pred - y.reshape(-1, 1)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float) # ReLU derivative
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train_epoch(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1):
        y_pred = self.forward(X)
        self.backward(X, y, y_pred, lr)
        return np.mean((y_pred.flatten() > 0.5) == y)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.forward(X)
        return np.mean((y_pred.flatten() > 0.5) == y)

class MemoryModule:
    def __init__(self, shape: Tuple, decay: float = 0.9):
        self.memory = np.zeros(shape)
        self.decay = decay
    def update(self, gradient: np.ndarray) -> np.ndarray:
        self.memory = self.decay * self.memory + (1 - self.decay) * gradient
        return self.memory

class DeepOptimizer:
    def __init__(self, param_shapes: List[Tuple], n_levels: int = 3):
        self.n_levels = n_levels
        self.memories = []
        decays = [0.5, 0.9, 0.99]
        for level in range(n_levels):
            level_memories = {i: MemoryModule(shape, decays[level]) for i, shape in enumerate(param_shapes)}
            self.memories.append(level_memories)

    def compute_update(self, param_id: int, gradient: np.ndarray) -> np.ndarray:
        updates = []
        for level in range(self.n_levels):
            mem_update = self.memories[level][param_id].update(gradient)
            weight = 1.0 / (level + 1)
            updates.append(weight * mem_update)
        return np.sum(updates, axis=0)

class ContinuumMemoryBlock:
    def __init__(self, dim: int, update_frequency: int, decay: float = 0.95):
        self.dim = dim
        self.update_frequency = update_frequency
        self.decay = decay
        self.memory = np.zeros(dim)
        self.step_count = 0

    def update(self, input_signal: np.ndarray, surprise: float = 1.0):
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            update_strength = min(1.0, surprise)
            self.memory = (self.decay * self.memory + update_strength * (1 - self.decay) * input_signal[:self.dim])
        return self.memory
    
    def read(self) -> np.ndarray:
        return self.memory

class ContinuumMemorySystem:
    def __init__(self, dim: int):
        self.blocks = [
            ContinuumMemoryBlock(dim, update_frequency=1, decay=0.5),
            ContinuumMemoryBlock(dim, update_frequency=5, decay=0.9),
            ContinuumMemoryBlock(dim, update_frequency=20, decay=0.99),
        ]
        self.dim = dim

    def update(self, input_signal: np.ndarray, surprise: float = 1.0) -> np.ndarray:
        combined = np.zeros(self.dim)
        weights = [0.5, 0.3, 0.2]
        for block, weight in zip(self.blocks, weights):
            block.update(input_signal, surprise)
            combined += weight * block.read()
        return combined
    
    def get_memory_state(self) -> Dict[str, np.ndarray]:
        return {'fast': self.blocks[0].read(), 'medium': self.blocks[1].read(), 'slow': self.blocks[2].read()}

class NestedLearningNetwork:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)
        param_shapes = [self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape]
        self.optimizer = DeepOptimizer(param_shapes, n_levels=3)
        self.cms_hidden = ContinuumMemorySystem(hidden_dim)
        self.cms_output = ContinuumMemorySystem(1)
        self.task_memories = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        batch_mean = np.mean(self.a1, axis=0)
        surprise = np.std(self.a1)
        cms_memory = self.cms_hidden.update(batch_mean, surprise)
        self.a1_modulated = self.a1 + 0.1 * cms_memory
        self.z2 = self.a1_modulated @ self.W2 + self.b2
        output = 1 / (1 + np.exp(-self.z2))
        self.cms_output.update(np.mean(output, axis=0, keepdims=True).flatten(), surprise)
        return output

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, lr: float = 0.1):
        m = X.shape[0]
        dz2 = y_pred - y.reshape(-1, 1)
        dW2 = self.a1_modulated.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float) # ReLU derivative
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)

        reg_strength = 15.0 # EWC Regularization
        if self.task_memories:
            last_mem = self.task_memories[-1]
            dW1 += reg_strength * (self.W1 - last_mem['W1']) / m
            dW2 += reg_strength * (self.W2 - last_mem['W2']) / m

        update_W1 = self.optimizer.compute_update(0, dW1)
        update_b1 = self.optimizer.compute_update(1, db1)
        update_W2 = self.optimizer.compute_update(2, dW2)
        update_b2 = self.optimizer.compute_update(3, db2)

        self.W2 -= lr * update_W2
        self.b2 -= lr * update_b2
        self.W1 -= lr * update_W1
        self.b1 -= lr * update_b1

    def consolidate_task(self, task: Task):
        self.task_memories.append({
            'W1': self.W1.copy(),
            'W2': self.W2.copy(),
            'cms_state': self.cms_hidden.get_memory_state()
        })

    def train_epoch(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1):
        y_pred = self.forward(X)
        self.backward(X, y, y_pred, lr)
        return np.mean((y_pred.flatten() > 0.5) == y)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.forward(X)
        return np.mean((y_pred.flatten() > 0.5) == y)

# =============================================================================
# EVALUATION LOGIC
# =============================================================================

def calculate_metrics(R: np.ndarray):
    """
    Tính toán các chỉ số đánh giá Continual Learning
    R[i, j]: Accuracy on task j after training on task i
    """
    n_tasks = R.shape[0]
    
    # 1. Average Accuracy (ACC) - Sau khi học xong tất cả
    # Trung bình accuracy của các task tại thời điểm cuối cùng
    acc = np.mean(R[-1, :])
    
    # 2. Backward Transfer (BWT)
    # Ảnh hưởng của việc học task mới lên task cũ
    # BWT = (1 / (T-1)) * Sum(R[T,i] - R[i,i]) với i < T
    bwt_sum = 0
    for i in range(n_tasks - 1):
        bwt_sum += R[-1, i] - R[i, i]
    bwt = bwt_sum / (n_tasks - 1)
    
    return acc, bwt

def print_table_row(col1, col2, col3, col4):
    print(f"| {col1:<15} | {col2:<15} | {col3:<15} | {col4:<15} |")

def run_evaluation():
    n_tasks = 3
    n_epochs = 100
    tasks = [generate_task_data(i) for i in range(n_tasks)]
    
    models = {
        "Simple Network": SimpleNetwork(),
        "Nested Learning": NestedLearningNetwork()
    }
    
    # R matrix stores accuracy: R[trained_task_idx, eval_task_idx]
    results = {name: np.zeros((n_tasks, n_tasks)) for name in models}
    
    print("\n" + "="*73)
    print("EVALUATION STARTED")
    print("="*73)
    
    for model_name, model in models.items():
        print(f"\nEvaluating: {model_name}...")
        
        for i, task in enumerate(tasks):
            # Train on current task
            for _ in range(n_epochs):
                model.train_epoch(task.X, task.y)
            
            # Consolidate if Nested Learning
            if isinstance(model, NestedLearningNetwork):
                model.consolidate_task(task)
            
            # Evaluate on ALL tasks (up to current)
            # Note: In standard CL evaluation, we usually evaluate on all tasks 
            # to see zero-shot performance on future tasks too, but here we focus on forgetting.
            for j in range(n_tasks):
                acc = model.evaluate(tasks[j].X, tasks[j].y)
                results[model_name][i, j] = acc
                
    # =========================================================================
    # PRINT RESULTS TABLE
    # =========================================================================
    
    print("\n" + "="*73)
    print("FINAL RESULTS TABLE")
    print("="*73)
    
    # Header
    print_table_row("Model", "Metric", "Value", "Description")
    print("-" * 73)
    
    for model_name in models:
        R = results[model_name]
        acc, bwt = calculate_metrics(R)
        
        # Format metrics
        acc_str = f"{acc:.2%}"
        bwt_str = f"{bwt:+.2%}" # Thêm dấu + để biết tăng hay giảm
        
        print_table_row(model_name, "Avg Accuracy", acc_str, "Higher is better")
        print_table_row("", "Backward Trans", bwt_str, ">0: Help, <0: Forget")
        print("-" * 73)

    print("\nDETAILED ACCURACY MATRIX (Rows: Trained Task, Cols: Eval Task)")
    print("-" * 73)
    
    for model_name in models:
        print(f"\nModel: {model_name}")
        print(f"{'':<10} | {'Task 1':<10} | {'Task 2':<10} | {'Task 3':<10}")
        print("-" * 46)
        R = results[model_name]
        for i in range(n_tasks):
            row_str = f"After T{i+1:<2} | "
            for j in range(n_tasks):
                row_str += f"{R[i, j]:.2%}   | "
            print(row_str)

if __name__ == "__main__":
    run_evaluation()
