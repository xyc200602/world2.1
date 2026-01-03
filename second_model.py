import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

class WorldModelEnhanced:
    def __init__(self, num_individuals=10000, initial_value=100, num_iterations=200):
        self.num_individuals = num_individuals
        self.initial_value = initial_value
        self.num_iterations = num_iterations
        
        # 生成智力值（使用截断正态分布，确保在50-150之间）
        self.intelligence = self._generate_intelligence(num_individuals)
        
        # 初始化势力值
        self.values = np.full(num_individuals, initial_value, dtype=np.float64)
        
        # 历史记录
        self.history = {
            'avg_values': [],      # 平均势力值
            'median_values': [],   # 中位数势力值 - 新增
            'min_values': [],      # 最小势力值
            'max_values': [],      # 最大势力值
            'std_values': [],      # 标准差
            'gini_coefficients': [],
            'correlations': []
        }
        
        # 最终统计
        self.final_stats = {}
    
    def _generate_intelligence(self, n):
        """生成50-150之间的智力值，改进的正态分布生成方法"""
        # 方法：生成正态分布，然后截断到指定范围
        mean, std = 100, 25  # 目标均值和标准差
        
        # 生成标准正态分布
        samples = np.random.randn(n) * std + mean
        
        # 反复生成直到所有样本都在范围内
        out_of_range = (samples < 50) | (samples > 150)
        while np.any(out_of_range):
            # 只替换超出范围的样本
            new_samples = np.random.randn(np.sum(out_of_range)) * std + mean
            samples[out_of_range] = new_samples
            out_of_range = (samples < 50) | (samples > 150)
        
        return samples
    
    def _calculate_gini(self, values):
        """计算基尼系数"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumulative_values = np.cumsum(sorted_values)
        total_value = cumulative_values[-1]
        
        if total_value == 0:
            return 0.0
        
        lorenz_curve = cumulative_values / total_value
        perfect_equality = np.linspace(0, 1, n)
        gini = np.sum(perfect_equality - lorenz_curve) / np.sum(perfect_equality)
        return gini
    
    def simulate_iteration(self, iteration):
        """执行一次迭代计算"""
        # 运气因素
        luck_multiplier = np.where(
            np.random.random(self.num_individuals) > 0.5,
            1.1,  # 好运
            0.9   # 坏运
        )
        
        # 智力因素
        intelligence_factor = 0.9 + (self.intelligence - 50) * 0.2 / 100
        
        # 综合计算
        luck_effect = self.values * luck_multiplier
        intelligence_effect = self.values * intelligence_factor
        
        self.values = 0.7 * luck_effect + 0.3 * intelligence_effect
        
        # 记录统计信息
        self._record_stats(iteration)
    
    def _record_stats(self, iteration):
        """记录当前统计信息"""
        current_values = self.values
        
        # 基本统计
        self.history['avg_values'].append(np.mean(current_values))
        self.history['median_values'].append(np.median(current_values))  # 新增中位数
        self.history['min_values'].append(np.min(current_values))
        self.history['max_values'].append(np.max(current_values))
        self.history['std_values'].append(np.std(current_values))
        
        # 基尼系数
        gini = self._calculate_gini(current_values)
        self.history['gini_coefficients'].append(gini)
        
        # 相关系数
        correlation = np.corrcoef(self.intelligence, current_values)[0, 1]
        self.history['correlations'].append(correlation if not np.isnan(correlation) else 0.0)
    
    def run_simulation(self, verbose=True):
        """运行完整模拟"""
        if verbose:
            print(f"开始模拟: {self.num_individuals}个个体, {self.num_iterations}次迭代")
            print("-" * 60)
        
        for i in range(self.num_iterations):
            self.simulate_iteration(i)
            
            if verbose and (i + 1) % 20 == 0:
                avg_val = self.history['avg_values'][-1]
                median_val = self.history['median_values'][-1]
                print(f"迭代 {i+1:3d}/{self.num_iterations} | "
                      f"平均势力: {avg_val:8.2f} | "
                      f"中位数: {median_val:8.2f}")
        
        # 计算最终统计
        self._calculate_final_stats()
        
        if verbose:
            self.print_summary()
    
    def _calculate_final_stats(self):
        """计算最终统计信息"""
        final_values = self.values
        
        self.final_stats = {
            'mean_value': np.mean(final_values),
            'median_value': np.median(final_values),
            'std_value': np.std(final_values),
            'min_value': np.min(final_values),
            'max_value': np.max(final_values),
            'gini': self._calculate_gini(final_values),
            'correlation': np.corrcoef(self.intelligence, final_values)[0, 1],
            'deciles': np.percentile(final_values, range(0, 101, 10))
        }
    
    def print_summary(self):
        """打印模拟总结"""
        print("\n" + "="*70)
        print("模拟结果总结")
        print("="*70)
        
        stats = self.final_stats
        
        print(f"\n整体统计:")
        print(f"  平均势力值: {stats['mean_value']:.2f}")
        print(f"  中位数势力值: {stats['median_value']:.2f}")
        print(f"  标准差: {stats['std_value']:.2f}")
        print(f"  范围: {stats['min_value']:.2f} - {stats['max_value']:.2f}")
        print(f"  基尼系数: {stats['gini']:.4f}")
        print(f"  智力与势力相关系数: {stats['correlation']:.4f}")
        
        # 平均与中位数差距分析
        mean_median_gap = abs(stats['mean_value'] - stats['median_value'])
        print(f"\n平均值与中位数差距分析:")
        print(f"  平均值 - 中位数 = {mean_median_gap:.2f}")
        print(f"  差距百分比: {(mean_median_gap / stats['mean_value'] * 100):.2f}%")
        
        if stats['mean_value'] > stats['median_value']:
            print("  说明: 分布右偏，存在少数极高值拉高平均值")
        else:
            print("  说明: 分布左偏，存在少数极低值拉低平均值")
    
    def visualize_results(self, save_path=None):
        """可视化结果"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 势力值随时间变化（添加中位数）
        ax1 = plt.subplot(2, 3, 1)
        iterations = range(1, self.num_iterations + 1)
        
        # 绘制平均势力值
        avg_line, = ax1.plot(iterations, self.history['avg_values'], 
                            'b-', linewidth=2.5, label='平均值', alpha=0.8)
        
        # 绘制中位数势力值
        median_line, = ax1.plot(iterations, self.history['median_values'],
                               'r-', linewidth=2.5, label='中位数', alpha=0.8)
        
        # 填充范围区域
        ax1.fill_between(iterations, 
                         self.history['min_values'],
                         self.history['max_values'],
                         alpha=0.15, color='blue', label='范围')
        
        # 添加初始值参考线
        ax1.axhline(y=self.initial_value, color='gray', linestyle='--', 
                   alpha=0.5, label='初始值')
        
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('势力值', fontsize=12)
        ax1.set_title('势力值随时间变化（平均值 vs 中位数）', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # 2. 智力与最终势力值散点图
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(self.intelligence, self.values, 
                             c=self.values, cmap='viridis', 
                             alpha=0.4, s=8, edgecolors='none')
        
        # 添加回归线
        if len(self.intelligence) > 1:
            z = np.polyfit(self.intelligence, self.values, 1)
            p = np.poly1d(z)
            x_range = np.array([np.min(self.intelligence), np.max(self.intelligence)])
            ax2.plot(x_range, p(x_range), 'r-', linewidth=2, 
                    label=f'y = {z[0]:.4f}x + {z[1]:.2f}')
        
        ax2.set_xlabel('智力值', fontsize=12)
        ax2.set_ylabel('最终势力值', fontsize=12)
        ax2.set_title(f'智力与最终势力值 (r={self.final_stats["correlation"]:.3f})', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='势力值')
        
        # 3. 最终势力值分布（改进的正态分布拟合）
        ax3 = plt.subplot(2, 3, 3)
        
        # 绘制直方图
        hist_values, bins, patches = ax3.hist(self.values, bins=80, 
                                             color='steelblue', edgecolor='black', 
                                             alpha=0.7, density=True)
        
        # 改进的正态分布拟合
        mean_val = self.final_stats['mean_value']
        std_val = self.final_stats['std_value']
        
        # 确定合适的x范围
        data_min = self.final_stats['min_value']
        data_max = self.final_stats['max_value']
        data_range = data_max - data_min
        
        # 确保x范围足够宽，但不过分
        x_min = max(data_min - 0.1 * data_range, 0)  # 不能小于0
        x_max = data_max + 0.1 * data_range
        x = np.linspace(x_min, x_max, 500)
        
        # 只有当标准差大于0时才绘制正态分布
        if std_val > 0:
            # 使用scipy.stats的正态分布
            y = stats.norm.pdf(x, mean_val, std_val)
            ax3.plot(x, y, 'r-', linewidth=2.5, label=f'正态分布\nμ={mean_val:.1f}, σ={std_val:.1f}')
        
        # 添加平均值和中位数线
        ax3.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label=f'均值: {mean_val:.1f}')
        ax3.axvline(self.final_stats['median_value'], color='orange', 
                   linestyle='--', linewidth=1.5, alpha=0.7, 
                   label=f'中位数: {self.final_stats["median_value"]:.1f}')
        
        # 添加KS检验结果
        if std_val > 0:
            # 执行Kolmogorov-Smirnov检验
            ks_stat, p_value = stats.kstest((self.values - mean_val) / std_val, 'norm')
            ax3.text(0.95, 0.95, f'KS检验: p={p_value:.4f}', 
                    transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax3.set_xlabel('势力值', fontsize=12)
        ax3.set_ylabel('概率密度', fontsize=12)
        ax3.set_title('最终势力值分布（正态分布拟合）', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=9)
        
        # 4. 平均值与中位数差距
        ax4 = plt.subplot(2, 3, 4)
        
        # 计算平均值与中位数的差距
        gaps = np.array(self.history['avg_values']) - np.array(self.history['median_values'])
        gap_line = ax4.plot(iterations, gaps, 'purple', linewidth=2)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 填充正负区域
        ax4.fill_between(iterations, 0, gaps, where=(gaps > 0), 
                        color='red', alpha=0.2, label='平均值 > 中位数')
        ax4.fill_between(iterations, 0, gaps, where=(gaps < 0), 
                        color='blue', alpha=0.2, label='平均值 < 中位数')
        
        ax4.set_xlabel('迭代次数', fontsize=12)
        ax4.set_ylabel('平均值 - 中位数', fontsize=12)
        ax4.set_title('平均值与中位数差距变化', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. 智力值分布（改进的正态分布拟合）
        ax5 = plt.subplot(2, 3, 5)
        
        # 绘制智力值直方图
        hist_intel, bins_intel, patches_intel = ax5.hist(self.intelligence, bins=40, 
                                                        color='darkorange', 
                                                        edgecolor='black', 
                                                        alpha=0.7, density=True)
        
        # 智力值的正态分布拟合
        intel_mean = np.mean(self.intelligence)
        intel_std = np.std(self.intelligence)
        
        # 生成x值
        x_intel = np.linspace(np.min(self.intelligence), np.max(self.intelligence), 200)
        
        if intel_std > 0:
            y_intel = stats.norm.pdf(x_intel, intel_mean, intel_std)
            ax5.plot(x_intel, y_intel, 'b-', linewidth=2.5, 
                    label=f'正态分布\nμ={intel_mean:.1f}, σ={intel_std:.1f}')
        
        # 添加目标线（100）
        ax5.axvline(x=100, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='目标均值: 100')
        
        ax5.set_xlabel('智力值', fontsize=12)
        ax5.set_ylabel('概率密度', fontsize=12)
        ax5.set_title('智力值分布', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. 基尼系数变化
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(iterations, self.history['gini_coefficients'], 
                'g-', linewidth=2)
        ax6.set_xlabel('迭代次数', fontsize=12)
        ax6.set_ylabel('基尼系数', fontsize=12)
        ax6.set_title('不平等程度变化（基尼系数）', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 添加基尼系数参考线
        ax6.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='警戒线(0.3)')
        ax6.axhline(y=0.4, color='darkred', linestyle='--', alpha=0.5, label='危险线(0.4)')
        ax6.legend()
        
        plt.suptitle(f'世界模型模拟结果 (N={self.num_individuals}, 迭代={self.num_iterations})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_comparison_chart(self):
        """专门绘制平均值与中位数的对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        iterations = range(1, self.num_iterations + 1)
        
        # 子图1：平均值与中位数对比
        ax1.plot(iterations, self.history['avg_values'], 'b-', 
                linewidth=2.5, label='平均值', alpha=0.8)
        ax1.plot(iterations, self.history['median_values'], 'r-', 
                linewidth=2.5, label='中位数', alpha=0.8)
        
        # 填充两者之间的区域
        ax1.fill_between(iterations, 
                         self.history['avg_values'], 
                         self.history['median_values'],
                         alpha=0.2, color='purple', label='差距区域')
        
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('势力值', fontsize=12)
        ax1.set_title('平均值与中位数对比', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 添加统计信息框
        final_avg = self.history['avg_values'][-1]
        final_median = self.history['median_values'][-1]
        gap = final_avg - final_median
        gap_percent = (gap / final_avg) * 100
        
        stats_text = f'最终平均值: {final_avg:.2f}\n最终中位数: {final_median:.2f}\n差距: {gap:.2f}\n差距百分比: {gap_percent:.2f}%'
        
        if gap > 0:
            stats_text += f'\n(分布右偏)'
        else:
            stats_text += f'\n(分布左偏)'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # 子图2：差距变化趋势
        gaps = np.array(self.history['avg_values']) - np.array(self.history['median_values'])
        ax2.plot(iterations, gaps, 'purple', linewidth=2.5)
        ax2.fill_between(iterations, 0, gaps, 
                        where=(gaps > 0), 
                        color='red', alpha=0.3, label='平均值 > 中位数')
        ax2.fill_between(iterations, 0, gaps, 
                        where=(gaps < 0), 
                        color='blue', alpha=0.3, label='平均值 < 中位数')
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 添加趋势线
        if len(iterations) > 1:
            z_gap = np.polyfit(iterations, gaps, 1)
            p_gap = np.poly1d(z_gap)
            ax2.plot(iterations, p_gap(iterations), 'k--', alpha=0.7, 
                    linewidth=1.5, label=f'趋势线: y={z_gap[0]:.6f}x+{z_gap[1]:.2f}')
        
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('平均值 - 中位数', fontsize=12)
        ax2.set_title('平均值与中位数差距变化趋势', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(f'平均值与中位数对比分析 (N={self.num_individuals})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    print("世界模型模拟器 - 增强版")
    print("="*60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 创建模型
    model = WorldModelEnhanced(
        num_individuals=10000,  # 10000个个体
        initial_value=100,
        num_iterations=200      # 200次迭代
    )
    
    # 运行模拟
    print("开始模拟...")
    model.run_simulation(verbose=True)
    
    # 可视化结果
    print("\n生成可视化图表...")
    model.visualize_results()
    
    # 额外生成对比图
    response = input("\n是否生成平均值与中位数对比图? (y/n): ").strip().lower()
    if response == 'y':
        model.plot_comparison_chart()
    
    # 统计总结
    print("\n" + "="*60)
    print("统计总结:")
    print("="*60)
    
    # 计算最终差距
    final_avg = model.history['avg_values'][-1]
    final_median = model.history['median_values'][-1]
    gap = final_avg - final_median
    
    print(f"最终平均值: {final_avg:.2f}")
    print(f"最终中位数: {final_median:.2f}")
    print(f"平均值 - 中位数 = {gap:.2f}")
    print(f"差距百分比: {(gap / final_avg * 100):.2f}%")
    
    if gap > 0:
        print("结论: 分布右偏，存在少数高势力值个体拉高了平均值")
    else:
        print("结论: 分布左偏，存在少数低势力值个体拉低了平均值")
    
    print(f"基尼系数: {model.final_stats['gini']:.4f}")
    print(f"不平等程度: ", end="")
    if model.final_stats['gini'] < 0.3:
        print("相对平等")
    elif model.final_stats['gini'] < 0.4:
        print("中等不平等")
    else:
        print("高度不平等")


if __name__ == "__main__":
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 运行主程序
    main()