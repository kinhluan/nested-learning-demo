"""
Nested Learning Demo - Minh họa 4 điểm mạnh của Nested Learning
Paper: "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025)
Authors: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)

4 điểm mạnh được minh họa:
1. Giải quyết Catastrophic Forgetting
2. Thống nhất Architecture và Optimization (Deep Optimizers)
3. Kết quả thực nghiệm - So sánh với baseline
4. Multi-frequency Updates (Continuum Memory System)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

np.random.seed(42)

# =============================================================================
# PHẦN 1: CATASTROPHIC FORGETTING DEMO
# Minh họa vấn đề quên kiến thức khi học task mới
# =============================================================================

@dataclass
class Task:
    """Đại diện cho một task học"""
    name: str
    X: np.ndarray  # Input data
    y: np.ndarray  # Target labels


def generate_task_data(task_id: int, n_samples: int = 100) -> Task:
    """Tạo dữ liệu cho mỗi task - mỗi task là một pattern khác nhau"""
    np.random.seed(task_id)

    if task_id == 0:
        # Task 0: Linear pattern (y = 2x + noise)
        X = np.random.randn(n_samples, 2)
        y = (2 * X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
        name = "Linear Pattern"
    elif task_id == 1:
        # Task 1: XOR pattern
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
        name = "XOR Pattern"
    else:
        # Task 2: Circular pattern
        X = np.random.randn(n_samples, 2)
        y = ((X[:, 0]**2 + X[:, 1]**2) < 1).astype(float)
        name = "Circular Pattern"

    return Task(name=name, X=X, y=y)


class SimpleNetwork:
    """Mạng neural đơn giản - baseline bị catastrophic forgetting"""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        # Sử dụng ReLU thay vì Tanh để tạo các đường gấp khúc sắc nét hơn
        self.a1 = np.maximum(0, self.z1) 
        self.z2 = self.a1 @ self.W2 + self.b2
        return 1 / (1 + np.exp(-self.z2))

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, lr: float = 0.1):
        m = X.shape[0]

        # Output layer gradients
        dz2 = y_pred - y.reshape(-1, 1)
        dW2 = self.a1.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)

        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float) # ReLU derivative
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)

        # Update weights
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


# =============================================================================
# PHẦN 2: DEEP OPTIMIZERS
# Thống nhất Architecture và Optimization thành các cấp độ tối ưu hóa
# =============================================================================

class MemoryModule:
    """
    Memory module cho Deep Optimizer
    Mô phỏng cách optimizer (như Momentum, Adam) hoạt động như memory modules
    """
    def __init__(self, shape: Tuple, decay: float = 0.9):
        self.memory = np.zeros(shape)
        self.decay = decay

    def update(self, gradient: np.ndarray) -> np.ndarray:
        """Nén gradient thông qua exponential moving average"""
        self.memory = self.decay * self.memory + (1 - self.decay) * gradient
        return self.memory


class DeepOptimizer:
    """
    Deep Optimizer - Optimizer với nhiều cấp độ memory
    Mở rộng từ SGD/Adam với deeper memory hierarchy

    Level 0: Immediate gradient (fast)
    Level 1: Short-term memory (medium)
    Level 2: Long-term memory (slow)
    """

    def __init__(self, param_shapes: List[Tuple], n_levels: int = 3):
        self.n_levels = n_levels
        self.memories = []

        # Tạo memory modules cho mỗi level với decay rate khác nhau
        decays = [0.5, 0.9, 0.99]  # Fast -> Slow
        for level in range(n_levels):
            level_memories = {
                i: MemoryModule(shape, decays[level])
                for i, shape in enumerate(param_shapes)
            }
            self.memories.append(level_memories)

    def compute_update(self, param_id: int, gradient: np.ndarray) -> np.ndarray:
        """
        Tính update từ tất cả các level memory
        Kết hợp thông tin từ nhiều time scales
        """
        updates = []
        for level in range(self.n_levels):
            mem_update = self.memories[level][param_id].update(gradient)
            # Weight theo level (deeper = stronger regularization)
            weight = 1.0 / (level + 1)
            updates.append(weight * mem_update)

        return np.sum(updates, axis=0)


# =============================================================================
# PHẦN 3: CONTINUUM MEMORY SYSTEM (CMS)
# Multi-frequency updates - bộ nhớ với tốc độ cập nhật khác nhau
# =============================================================================

class ContinuumMemoryBlock:
    """
    Một block trong Continuum Memory System
    Cập nhật với tần suất khác nhau để capture thông tin ở nhiều time scales
    """

    def __init__(self, dim: int, update_frequency: int, decay: float = 0.95):
        self.dim = dim
        self.update_frequency = update_frequency  # Cập nhật mỗi N steps
        self.decay = decay
        self.memory = np.zeros(dim)
        self.step_count = 0

    def should_update(self) -> bool:
        return self.step_count % self.update_frequency == 0

    def update(self, input_signal: np.ndarray, surprise: float = 1.0):
        """
        Cập nhật memory dựa trên surprise level
        Surprise cao = ghi nhớ mạnh hơn (như trong Titans architecture)
        """
        self.step_count += 1

        if self.should_update():
            # Surprise-modulated update
            update_strength = min(1.0, surprise)
            self.memory = (self.decay * self.memory +
                          update_strength * (1 - self.decay) * input_signal[:self.dim])

        return self.memory

    def read(self) -> np.ndarray:
        return self.memory


class ContinuumMemorySystem:
    """
    Hệ thống bộ nhớ liên tục với nhiều blocks cập nhật ở tần suất khác nhau

    - Fast blocks: Cập nhật mỗi step (short-term memory)
    - Medium blocks: Cập nhật mỗi 5 steps (working memory)
    - Slow blocks: Cập nhật mỗi 20 steps (long-term memory)
    """

    def __init__(self, dim: int):
        self.blocks = [
            ContinuumMemoryBlock(dim, update_frequency=1, decay=0.5),   # Fast
            ContinuumMemoryBlock(dim, update_frequency=5, decay=0.9),   # Medium
            ContinuumMemoryBlock(dim, update_frequency=20, decay=0.99), # Slow
        ]
        self.dim = dim

    def update(self, input_signal: np.ndarray, surprise: float = 1.0) -> np.ndarray:
        """Cập nhật tất cả blocks và trả về combined memory"""
        combined = np.zeros(self.dim)
        weights = [0.5, 0.3, 0.2]  # Weight cho mỗi block

        for block, weight in zip(self.blocks, weights):
            block.update(input_signal, surprise)
            combined += weight * block.read()

        return combined

    def get_memory_state(self) -> Dict[str, np.ndarray]:
        """Lấy trạng thái của tất cả memory blocks"""
        return {
            'fast': self.blocks[0].read(),
            'medium': self.blocks[1].read(),
            'slow': self.blocks[2].read()
        }


# =============================================================================
# PHẦN 4: NESTED LEARNING NETWORK
# Kết hợp tất cả components để giải quyết Catastrophic Forgetting
# =============================================================================

class NestedLearningNetwork:
    """
    Network với Nested Learning paradigm
    Kết hợp:
    - Deep Optimizer (multi-level optimization)
    - Continuum Memory System (multi-frequency updates)
    - Self-modifying weights (inspired by HOPE/Titans)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        # Network weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

        # Deep Optimizer với multi-level memory
        param_shapes = [self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape]
        self.optimizer = DeepOptimizer(param_shapes, n_levels=3)

        # Continuum Memory System cho mỗi layer
        self.cms_hidden = ContinuumMemorySystem(hidden_dim)
        self.cms_output = ContinuumMemorySystem(1)

        # Task context memory (để nhớ patterns từ các tasks trước)
        self.task_memories = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Hidden layer với CMS integration
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU

        # Tích hợp memory từ CMS vào hidden representation
        batch_mean = np.mean(self.a1, axis=0)
        surprise = np.std(self.a1)  # Surprise = variability
        cms_memory = self.cms_hidden.update(batch_mean, surprise)

        # Modulate hidden activations với memory
        self.a1_modulated = self.a1 + 0.1 * cms_memory

        # Output layer
        self.z2 = self.a1_modulated @ self.W2 + self.b2
        output = 1 / (1 + np.exp(-self.z2))

        # Update output CMS
        self.cms_output.update(np.mean(output, axis=0, keepdims=True).flatten(), surprise)

        return output

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, lr: float = 0.1):
        m = X.shape[0]

        # 1. Compute gradients from current task (Task-specific loss)
        dz2 = y_pred - y.reshape(-1, 1)
        dW2 = self.a1_modulated.T @ dz2 / m
        db2 = np.mean(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float) # ReLU derivative
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0)

        # 2. Add Regularization from Memory (Knowledge Preservation)
        # Nếu đã có kiến thức cũ, thêm "lực kéo" để giữ weights không đổi quá nhiều
        reg_strength = 15.0  # Giảm xuống để mô hình linh hoạt hơn khi học task mới
        
        if self.task_memories:
            # Lấy kiến thức gần nhất (hoặc trung bình các task cũ)
            last_mem = self.task_memories[-1]
            
            # Tính gradient phạt: reg_strength * (current_weight - old_weight)
            # Điều này tương đương với L2 regularization quanh điểm tối ưu cũ
            dW1 += reg_strength * (self.W1 - last_mem['W1']) / m
            dW2 += reg_strength * (self.W2 - last_mem['W2']) / m
            # Bias thường ít quan trọng hơn nên có thể bỏ qua hoặc phạt nhẹ

        # 3. Deep Optimizer Update
        # Sử dụng Deep Optimizer để làm mượt gradient
        update_W1 = self.optimizer.compute_update(0, dW1)
        update_b1 = self.optimizer.compute_update(1, db1)
        update_W2 = self.optimizer.compute_update(2, dW2)
        update_b2 = self.optimizer.compute_update(3, db2)

        # 4. Apply updates
        self.W2 -= lr * update_W2
        self.b2 -= lr * update_b2
        self.W1 -= lr * update_W1
        self.b1 -= lr * update_b1

    def consolidate_task(self, task: Task):
        """
        Lưu lại knowledge từ task hiện tại vào long-term memory
        Đây là bước quan trọng để tạo 'checkpoint' cho kiến thức
        """
        print(f"   [Consolidating knowledge from task: {task.name}]")
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
# PHẦN 5: EXPERIMENTS & VISUALIZATION
# =============================================================================

def run_continual_learning_experiment(n_epochs_per_task: int = 50):
    """
    Thí nghiệm so sánh Simple Network vs Nested Learning Network
    trên 3 tasks liên tiếp
    """
    # Tạo 3 tasks
    tasks = [generate_task_data(i) for i in range(3)]

    # Khởi tạo 2 networks
    simple_net = SimpleNetwork()
    nested_net = NestedLearningNetwork()

    # Lưu kết quả
    results = {
        'simple': {task.name: [] for task in tasks},
        'nested': {task.name: [] for task in tasks}
    }

    print("=" * 70)
    print("NESTED LEARNING DEMO - Continual Learning Experiment")
    print("=" * 70)

    # Training loop - học từng task
    for task_idx, task in enumerate(tasks):
        print(f"\n>>> Training on Task {task_idx + 1}: {task.name}")
        print("-" * 50)

        for epoch in range(n_epochs_per_task):
            # Train
            simple_net.train_epoch(task.X, task.y)
            nested_net.train_epoch(task.X, task.y)

            # Evaluate on ALL tasks (để track forgetting)
            if epoch % 10 == 0 or epoch == n_epochs_per_task - 1:
                for eval_task in tasks:
                    simple_acc = simple_net.evaluate(eval_task.X, eval_task.y)
                    nested_acc = nested_net.evaluate(eval_task.X, eval_task.y)

                    results['simple'][eval_task.name].append(simple_acc)
                    results['nested'][eval_task.name].append(nested_acc)

        # Consolidate knowledge sau mỗi task (Nested Learning only)
        nested_net.consolidate_task(task)

        # Print current performance
        print(f"\nAfter Task {task_idx + 1}:")
        for eval_task in tasks:
            simple_acc = simple_net.evaluate(eval_task.X, eval_task.y)
            nested_acc = nested_net.evaluate(eval_task.X, eval_task.y)
            print(f"  {eval_task.name:20s} | Simple: {simple_acc:.2%} | Nested: {nested_acc:.2%}")

    return tasks, results


def visualize_results(tasks: List[Task], results: Dict):
    """Tạo visualization so sánh kết quả"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Nested Learning Demo - 4 Key Advantages', fontsize=14, fontweight='bold')

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # Plot 1: Catastrophic Forgetting Comparison
    ax1 = axes[0, 0]
    task_names = [t.name for t in tasks]
    x = np.arange(len(task_names))
    width = 0.35

    # Final accuracy sau khi học tất cả tasks
    simple_final = [results['simple'][name][-1] for name in task_names]
    nested_final = [results['nested'][name][-1] for name in task_names]

    bars1 = ax1.bar(x - width/2, simple_final, width, label='Simple Network', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, nested_final, width, label='Nested Learning', color='#2ecc71', alpha=0.8)

    ax1.set_ylabel('Accuracy')
    ax1.set_title('1. Catastrophic Forgetting\n(Final accuracy after all tasks)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names, rotation=15)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

    # Add value labels
    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.0%}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.annotate(f'{bar.get_height():.0%}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Deep Optimizer - Multi-level Memory Visualization
    ax2 = axes[0, 1]

    # Simulate gradient flow through deep optimizer levels
    steps = 50
    gradients = np.random.randn(steps) * 0.5

    # Memory at different levels
    level0 = np.zeros(steps)
    level1 = np.zeros(steps)
    level2 = np.zeros(steps)

    m0, m1, m2 = 0, 0, 0
    for i in range(steps):
        m0 = 0.5 * m0 + 0.5 * gradients[i]   # Fast decay
        m1 = 0.9 * m1 + 0.1 * gradients[i]   # Medium decay
        m2 = 0.99 * m2 + 0.01 * gradients[i] # Slow decay
        level0[i], level1[i], level2[i] = m0, m1, m2

    ax2.plot(level0, label='Level 0 (Fast)', color='#e74c3c', linewidth=2)
    ax2.plot(level1, label='Level 1 (Medium)', color='#f39c12', linewidth=2)
    ax2.plot(level2, label='Level 2 (Slow)', color='#2ecc71', linewidth=2)
    ax2.fill_between(range(steps), gradients, alpha=0.2, color='gray', label='Raw Gradients')

    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Memory Value')
    ax2.set_title('2. Deep Optimizer\n(Multi-level memory compression)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Performance Comparison
    ax3 = axes[1, 0]

    # Average accuracy across all tasks over time
    n_evals = len(results['simple'][tasks[0].name])
    simple_avg = np.mean([[results['simple'][t.name][i] for t in tasks] for i in range(n_evals)], axis=1)
    nested_avg = np.mean([[results['nested'][t.name][i] for t in tasks] for i in range(n_evals)], axis=1)

    ax3.plot(simple_avg, label='Simple Network', color='#e74c3c', linewidth=2, marker='o', markersize=4)
    ax3.plot(nested_avg, label='Nested Learning', color='#2ecc71', linewidth=2, marker='s', markersize=4)

    # Mark task boundaries
    task_boundaries = [6, 12]  # Approximate boundaries
    for i, boundary in enumerate(task_boundaries):
        if boundary < n_evals:
            ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            ax3.text(boundary + 0.2, 0.95, f'Task {i+2}', fontsize=9, alpha=0.7)

    ax3.set_xlabel('Evaluation Points')
    ax3.set_ylabel('Average Accuracy (all tasks)')
    ax3.set_title('3. Experimental Results\n(Average accuracy over training)')
    ax3.legend()
    ax3.set_ylim(0.4, 1.0)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Continuum Memory System - Multi-frequency Updates
    ax4 = axes[1, 1]

    # Simulate CMS behavior
    steps = 100
    input_signal = np.sin(np.linspace(0, 4*np.pi, steps)) + 0.3 * np.random.randn(steps)

    fast_mem = np.zeros(steps)
    medium_mem = np.zeros(steps)
    slow_mem = np.zeros(steps)

    f, m, s = 0, 0, 0
    for i in range(steps):
        # Fast: update every step
        f = 0.5 * f + 0.5 * input_signal[i]
        fast_mem[i] = f

        # Medium: update every 5 steps
        if i % 5 == 0:
            m = 0.9 * m + 0.1 * input_signal[i]
        medium_mem[i] = m

        # Slow: update every 20 steps
        if i % 20 == 0:
            s = 0.99 * s + 0.01 * input_signal[i]
        slow_mem[i] = s

    ax4.plot(input_signal, alpha=0.3, color='gray', label='Input Signal')
    ax4.plot(fast_mem, label='Fast Memory (every step)', color='#e74c3c', linewidth=2)
    ax4.plot(medium_mem, label='Medium Memory (every 5)', color='#f39c12', linewidth=2)
    ax4.plot(slow_mem, label='Slow Memory (every 20)', color='#2ecc71', linewidth=2)

    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Memory Value')
    ax4.set_title('4. Continuum Memory System\n(Multi-frequency updates)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nested_learning_demo.png', dpi=150, bbox_inches='tight')
    # plt.show()  # Comment out for non-GUI environment

    print("\n" + "=" * 70)
    print("Visualization saved to: nested_learning_demo.png")
    print("=" * 70)


def plot_decision_boundary(ax, model, X, y, title):
    """Hàm helper để vẽ decision boundary"""
    # Tạo lưới điểm
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Dự đoán trên toàn bộ lưới
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid_data)
    Z = Z.reshape(xx.shape)

    # Vẽ contour
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    
    # Vẽ data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='white', s=30)
    
    ax.set_title(title)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


def run_demo_with_boundary_visualization(n_epochs: int = 100):
    """
    Chạy demo và vẽ Decision Boundary sau mỗi task
    để thấy rõ Catastrophic Forgetting vs Nested Learning
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION DEMO - Decision Boundaries")
    print("=" * 70)

    tasks = [generate_task_data(i) for i in range(3)]
    simple_net = SimpleNetwork()
    nested_net = NestedLearningNetwork()

    # Setup plot: 2 rows (Simple vs Nested), 3 columns (Task 1, 2, 3)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Decision Boundary Evolution: Simple vs Nested Learning', fontsize=16, fontweight='bold')

    # Row labels
    pad = 5
    axes[0, 0].annotate("Simple Network", xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[0, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90, fontweight='bold')
    axes[1, 0].annotate("Nested Learning", xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center', rotation=90, fontweight='bold')

    for task_idx, task in enumerate(tasks):
        print(f"Training Task {task_idx + 1}: {task.name}...")

        # Train both models
        for _ in range(n_epochs):
            simple_net.train_epoch(task.X, task.y)
            nested_net.train_epoch(task.X, task.y)

        # Consolidate for Nested Net
        nested_net.consolidate_task(task)

        # Plot Simple Network
        plot_decision_boundary(axes[0, task_idx], simple_net, task.X, task.y,
                             f"After Task {task_idx+1}\n({task.name})")

        # Plot Nested Network
        plot_decision_boundary(axes[1, task_idx], nested_net, task.X, task.y,
                             f"After Task {task_idx+1}\n({task.name})")

    plt.tight_layout()
    plt.subplots_adjust(left=0.05) # Make room for row labels
    plt.savefig('nested_learning_boundaries.png', dpi=150)
    print("Visualization saved to: nested_learning_boundaries.png")


def demonstrate_cms_detail():
    """Demo chi tiết hoạt động của Continuum Memory System"""
    print("\n" + "=" * 70)
    print("CONTINUUM MEMORY SYSTEM - Detailed Demo")
    print("=" * 70)

    cms = ContinuumMemorySystem(dim=4)

    print("\nSimulating 30 time steps with varying input signals...\n")
    print(f"{'Step':>5} | {'Input':>30} | {'Fast':>12} | {'Medium':>12} | {'Slow':>12}")
    print("-" * 80)

    for step in range(30):
        # Generate input with some pattern
        input_signal = np.array([
            np.sin(step * 0.3),
            np.cos(step * 0.2),
            0.5 * np.sin(step * 0.5),
            0.3 * np.cos(step * 0.1)
        ])

        # Surprise varies - higher when input is very different
        surprise = abs(np.sin(step * 0.7))

        cms.update(input_signal, surprise)
        state = cms.get_memory_state()

        if step % 5 == 0:
            input_str = f"[{input_signal[0]:.2f}, {input_signal[1]:.2f}, ...]"
            fast_str = f"{np.mean(state['fast']):.3f}"
            med_str = f"{np.mean(state['medium']):.3f}"
            slow_str = f"{np.mean(state['slow']):.3f}"
            print(f"{step:>5} | {input_str:>30} | {fast_str:>12} | {med_str:>12} | {slow_str:>12}")

    print("\n" + "-" * 80)
    print("Observation: Fast memory changes rapidly, Slow memory changes gradually")
    print("This mimics how the brain handles short-term vs long-term memory!")


def main():
    """Main function - chạy tất cả demos"""
    print("\n" + "=" * 70)
    print("   NESTED LEARNING - A New ML Paradigm for Continual Learning")
    print("   Paper: NeurIPS 2025 (Google Research)")
    print("=" * 70)

    # Demo 1: Continual Learning Experiment
    print("\n[1/3] Running Continual Learning Experiment...")
    tasks, results = run_continual_learning_experiment(n_epochs_per_task=50)

    # Demo 2: CMS Detail
    print("\n[2/3] Demonstrating Continuum Memory System...")
    demonstrate_cms_detail()

    # Demo 3: Visualization
    print("\n[3/3] Creating Visualization...")
    visualize_results(tasks, results)

    # Demo 4: Boundary Visualization (New)
    print("\n[4/4] Creating Decision Boundary Visualization...")
    run_demo_with_boundary_visualization()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - 4 Key Advantages of Nested Learning")
    print("=" * 70)
    print("""
    1. CATASTROPHIC FORGETTING
       - Simple networks forget old tasks when learning new ones
       - Nested Learning maintains knowledge through multi-level memory

    2. UNIFIED ARCHITECTURE & OPTIMIZATION
       - Traditional: Architecture and Optimizer are separate
       - Nested Learning: Both are "levels of optimization"
       - Deep Optimizer compresses gradients at multiple time scales

    3. SUPERIOR EXPERIMENTAL RESULTS
       - Nested Learning maintains higher average accuracy
       - Less performance degradation on earlier tasks

    4. MULTI-FREQUENCY UPDATES (CMS)
       - Fast memory: Captures immediate patterns
       - Medium memory: Working memory for current context
       - Slow memory: Long-term knowledge preservation
       - Mimics human brain's neuroplasticity
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
