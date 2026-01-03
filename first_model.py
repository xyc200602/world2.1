import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class WorldModel:
    def __init__(self, num_individuals=10000, initial_value=100, num_iterations=200):
        """
        初始化世界模型
        
        参数:
        num_individuals: 个体数量
        initial_value: 初始势力值
        num_iterations: 模拟迭代次数
        """
        self.num_individuals = num_individuals
        self.initial_value = initial_value
        self.num_iterations = num_iterations
        
        # 初始化个体：智力值（正态分布，平均100，标准差约16.67，范围50-150）
        # 使用截断正态分布确保智力值在50-150之间
        mean_intelligence = 100
        std_intelligence = 25  # 标准差
        
        # 生成正态分布的智力值，然后截断到50-150
        self.intelligence = np.random.normal(mean_intelligence, std_intelligence, num_individuals)
        self.intelligence = np.clip(self.intelligence, 50, 150)
        
        # 初始化势力值
        self.values = np.full(num_individuals, initial_value, dtype=float)
        
        # 记录每次迭代的平均值
        self.avg_values_over_time = []
        
    def simulate_iteration(self):
        """执行一次迭代计算"""
        # 运气部分：50%概率×1.1，50%概率×0.9
        luck_factor = np.random.choice([1.1, 0.9], size=self.num_individuals, p=[0.5, 0.5])
        luck_values = self.values * luck_factor
        
        # 实力部分：根据智力值计算因子
        # 线性映射：智力50→0.9，智力100→1.0，智力150→1.1
        intelligence_factor = 0.9 + (self.intelligence - 50) * (0.2 / 100)
        strength_values = self.values * intelligence_factor
        
        # 综合计算：30%运气 + 70%实力
        self.values = 0.3 * luck_values + 0.7 * strength_values
        
        # 记录当前平均值
        self.avg_values_over_time.append(np.mean(self.values))
        
    def run_simulation(self):
        """运行完整模拟"""
        for i in range(self.num_iterations):
            self.simulate_iteration()
            
            # 每10次迭代打印一次进度
            if (i + 1) % 10 == 0:
                print(f"迭代 {i+1}/{self.num_iterations} 完成，平均势力值: {self.avg_values_over_time[-1]:.2f}")
    
    def analyze_results(self):
        """分析模拟结果"""
        print("\n=== 模拟结果分析 ===")
        print(f"初始个体数量: {self.num_individuals}")
        print(f"初始势力值: {self.initial_value}")
        print(f"模拟迭代次数: {self.num_iterations}")
        print(f"最终平均势力值: {np.mean(self.values):.2f}")
        print(f"最终势力值标准差: {np.std(self.values):.2f}")
        print(f"最终势力值范围: {np.min(self.values):.2f} - {np.max(self.values):.2f}")
        print(f"智力值范围: {np.min(self.intelligence):.2f} - {np.max(self.intelligence):.2f}")
        
        # 计算相关系数
        correlation = np.corrcoef(self.intelligence, self.values)[0, 1]
        print(f"智力与最终势力值的相关系数: {correlation:.4f}")
        
        # 计算基尼系数（衡量不平等程度）
        sorted_values = np.sort(self.values)
        n = len(sorted_values)
        cumulative_values = np.cumsum(sorted_values)
        total_value = cumulative_values[-1]
        
        # 计算洛伦兹曲线面积
        lorenz_curve = cumulative_values / total_value
        perfect_equality = np.linspace(0, 1, n)
        
        # 基尼系数 = (完美平等线与洛伦兹曲线之间面积) / (完美平等线下面积)
        gini_coefficient = np.sum(perfect_equality - lorenz_curve) / np.sum(perfect_equality)
        print(f"最终势力值的基尼系数: {gini_coefficient:.4f}")
        
    def plot_results(self):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 折线图：势力值随时间变化
        ax1 = axes[0, 0]
        ax1.plot(range(1, self.num_iterations + 1), self.avg_values_over_time, 
                linewidth=2, color='royalblue')
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('平均势力值', fontsize=12)
        ax1.set_title('势力值随时间变化趋势', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.initial_value, color='red', linestyle='--', alpha=0.7, 
                   label=f'初始值: {self.initial_value}')
        ax1.legend()
        
        # 2. 散点图：智力与最终势力值的关系
        ax2 = axes[0, 1]
        scatter = ax2.scatter(self.intelligence, self.values, 
                             c=self.values, cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel('智力值', fontsize=12)
        ax2.set_ylabel('最终势力值', fontsize=12)
        ax2.set_title('智力与最终势力值的关系', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('势力值', fontsize=12)
        
        # 添加回归线
        z = np.polyfit(self.intelligence, self.values, 1)
        p = np.poly1d(z)
        ax2.plot(self.intelligence, p(self.intelligence), "r--", alpha=0.8, 
                label=f'回归线: y={z[0]:.3f}x+{z[1]:.2f}')
        ax2.legend()
        
        # 3. 最终势力值分布直方图
        ax3 = axes[1, 0]
        ax3.hist(self.values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('最终势力值', fontsize=12)
        ax3.set_ylabel('个体数量', fontsize=12)
        ax3.set_title('最终势力值分布', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'平均值: {np.mean(self.values):.2f}\n标准差: {np.std(self.values):.2f}\n中位数: {np.median(self.values):.2f}'
        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 4. 智力值分布直方图
        ax4 = axes[1, 1]
        ax4.hist(self.intelligence, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('智力值', fontsize=12)
        ax4.set_ylabel('个体数量', fontsize=12)
        ax4.set_title('智力值分布', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加正态分布曲线
        intelligence_mean = np.mean(self.intelligence)
        intelligence_std = np.std(self.intelligence)
        x = np.linspace(np.min(self.intelligence), np.max(self.intelligence), 100)
        y = stats.norm.pdf(x, intelligence_mean, intelligence_std) * len(self.intelligence) * (np.max(self.intelligence) - np.min(self.intelligence)) / 30
        ax4.plot(x, y, 'r-', linewidth=2, label='正态分布拟合')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_lorenz_curve(self):
        """绘制洛伦兹曲线，展示不平等程度"""
        # 对势力值排序
        sorted_values = np.sort(self.values)
        n = len(sorted_values)
        cumulative_values = np.cumsum(sorted_values)
        total_value = cumulative_values[-1]
        
        # 计算洛伦兹曲线
        lorenz_curve = cumulative_values / total_value
        
        # 完美平等线
        perfect_equality = np.linspace(0, 1, n)
        
        # 计算基尼系数
        gini_coefficient = np.sum(perfect_equality - lorenz_curve) / np.sum(perfect_equality)
        
        # 绘制洛伦兹曲线
        plt.figure(figsize=(10, 8))
        plt.plot(np.linspace(0, 1, n), perfect_equality, 'b--', alpha=0.7, label='完美平等线')
        plt.plot(np.linspace(0, 1, n), lorenz_curve, 'r-', linewidth=3, label='洛伦兹曲线')
        plt.fill_between(np.linspace(0, 1, n), lorenz_curve, perfect_equality, 
                        color='red', alpha=0.2, label='不平等区域')
        
        plt.xlabel('人口累积比例', fontsize=12)
        plt.ylabel('财富累积比例', fontsize=12)
        plt.title(f'洛伦兹曲线 (基尼系数: {gini_coefficient:.4f})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# 运行模拟
def main():
    # 设置参数
    NUM_INDIVIDUALS = 10000  # 个体数量
    INITIAL_VALUE = 100     # 初始势力值
    NUM_ITERATIONS = 200    # 迭代次数
    
    print("=" * 60)
    print("世界模型模拟开始")
    print(f"个体数量: {NUM_INDIVIDUALS}")
    print(f"初始势力值: {INITIAL_VALUE}")
    print(f"迭代次数: {NUM_ITERATIONS}")
    print("=" * 60)
    
    # 创建世界模型实例
    world = WorldModel(NUM_INDIVIDUALS, INITIAL_VALUE, NUM_ITERATIONS)
    
    # 运行模拟
    print("\n正在运行模拟...")
    world.run_simulation()
    
    # 分析结果
    world.analyze_results()
    
    # 绘制图表
    print("\n正在生成图表...")
    world.plot_results()
    
    # 可选：绘制洛伦兹曲线
    print("\n是否绘制洛伦兹曲线？(y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        world.plot_lorenz_curve()
    
    print("\n模拟完成！")


if __name__ == "__main__":
    # 设置随机种子以确保结果可重复
    np.random.seed(42)
    
    # 设置中文字体（如果系统中有中文字体）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()