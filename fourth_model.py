import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import matplotlib

warnings.filterwarnings('ignore')

# ==================== 字体设置 ====================
def setup_fonts():
    """设置字体，解决中文显示和重叠问题"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小防止重叠
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14

setup_fonts()

# ==================== 世界模型类 ====================
class WorldModel:
    def __init__(self, num_individuals=10000, initial_value=100, num_iterations=200, random_seed=42):
        self.num_individuals = num_individuals
        self.initial_value = initial_value
        self.num_iterations = num_iterations
        
        # 设置随机种子以确保可重复性
        np.random.seed(random_seed)
        
        # 生成智力值
        self.intelligence = self._generate_intelligence(num_individuals)
        
        # 初始化势力值
        self.values = np.full(num_individuals, initial_value, dtype=np.float64)
        
        # 历史记录
        self.history = {
            'avg_values': [],
            'median_values': [],
            'min_values': [],
            'max_values': [],
            'std_values': [],
            'gini_coefficients': [],
            'pearson_correlations': [],
            'spearman_correlations': [],
            'kendall_correlations': [],
            'percentile_10': [],  # 10%分位数
            'percentile_90': []   # 90%分位数
        }
        
        self.final_stats = {}
    
    def _generate_intelligence(self, n):
        """生成智力值 - 使用截断正态分布"""
        mean, std = 100, 25
        min_val, max_val = 50, 150
        
        # 使用截断正态分布生成样本
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
        
        return samples
    
    def _calculate_gini(self, values):
        """计算基尼系数 - 使用梯形积分法"""
        # 过滤掉非正值，因为基尼系数通常对正数定义
        positive_values = values[values > 0]
        n = len(positive_values)
        
        if n <= 1:
            return 0.0
        
        # 排序
        sorted_values = np.sort(positive_values)
        
        # 计算累积和
        cumulative_values = np.cumsum(sorted_values)
        total_value = cumulative_values[-1]
        
        if total_value == 0:
            return 0.0
        
        # 洛伦兹曲线值
        lorenz_curve = cumulative_values / total_value
        
        # 完美平等线
        perfect_equality = np.linspace(0, 1, n)
        
        # 使用梯形法计算面积（更准确）
        # 洛伦兹曲线与完美平等线之间的面积
        area_between = np.trapz(perfect_equality - lorenz_curve, dx=1/(n-1)) if n > 1 else 0
        
        # 完美平等线下的面积
        area_under_perfect = np.trapz(perfect_equality, dx=1/(n-1)) if n > 1 else 0.5
        
        # 基尼系数
        gini = area_between / area_under_perfect
        
        return gini
    
    def calculate_spearman_correlation(self, intelligence, values):
        """计算斯皮尔曼等级相关系数 - 使用scipy函数"""
        try:
            # 使用scipy的spearmanr函数，它处理了相同排名的情况
            correlation, p_value = stats.spearmanr(intelligence, values, nan_policy='omit')
            return correlation if not np.isnan(correlation) else 0.0
        except:
            # 备用方法：使用排名后的皮尔逊相关系数
            intel_ranks = stats.rankdata(intelligence, method='average')
            value_ranks = stats.rankdata(values, method='average')
            correlation = np.corrcoef(intel_ranks, value_ranks)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_kendall_correlation(self, intelligence, values):
        """计算肯德尔等级相关系数 - 使用scipy函数"""
        try:
            correlation, p_value = stats.kendalltau(intelligence, values, nan_policy='omit')
            return correlation if not np.isnan(correlation) else 0.0
        except:
            # 手动计算（简略版）
            n = len(intelligence)
            if n <= 1:
                return 0.0
            
            concordant = 0
            discordant = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    intel_compare = intelligence[i] - intelligence[j]
                    value_compare = values[i] - values[j]
                    
                    if intel_compare * value_compare > 0:
                        concordant += 1
                    elif intel_compare * value_compare < 0:
                        discordant += 1
            
            total = concordant + discordant
            if total == 0:
                return 0
            
            correlation = (concordant - discordant) / total
            return correlation
    
    def calculate_all_correlations(self, intelligence, values):
        """计算所有类型的相关系数"""
        correlations = {}
        
        # 皮尔逊相关系数
        if len(intelligence) > 1:
            corr_matrix = np.corrcoef(intelligence, values)
            correlations['pearson'] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            correlations['pearson'] = 0.0
        
        # 斯皮尔曼等级相关系数
        correlations['spearman'] = self.calculate_spearman_correlation(intelligence, values)
        
        # 肯德尔等级相关系数
        correlations['kendall'] = self.calculate_kendall_correlation(intelligence, values)
        
        return correlations
    
    def simulate_iteration(self, iteration):
        """执行一次迭代计算 - 修正的更新公式"""
        # 运气因素 - 保持不变
        luck_multiplier = np.where(
            np.random.random(self.num_individuals) > 0.5,
            1.1,  # 好运
            0.9   # 坏运
        )
        
        # 智力因素 - 修正：确保在合理范围内
        # 智力值50-150映射到0.9-1.1
        intelligence_factor = 0.9 + (self.intelligence - 50) * 0.2 / 100
        
        # 综合计算 - 修正：使用乘性组合而不是加性组合
        # 原来：self.values = 0.7 * luck_effect + 0.3 * intelligence_effect
        # 现在：使用几何加权平均，更符合复利效应
        combined_factor = np.power(luck_multiplier, 0.7) * np.power(intelligence_factor, 0.3)
        
        # 更新势力值
        self.values = self.values * combined_factor
        
        # 记录统计信息
        self._record_stats(iteration)
    
    def _record_stats(self, iteration):
        """记录当前统计信息"""
        current_values = self.values
        
        # 基本统计
        self.history['avg_values'].append(np.mean(current_values))
        self.history['median_values'].append(np.median(current_values))
        self.history['min_values'].append(np.min(current_values))
        self.history['max_values'].append(np.max(current_values))
        self.history['std_values'].append(np.std(current_values))
        
        # 计算分位数
        self.history['percentile_10'].append(np.percentile(current_values, 10))
        self.history['percentile_90'].append(np.percentile(current_values, 90))
        
        # 基尼系数
        gini = self._calculate_gini(current_values)
        self.history['gini_coefficients'].append(gini)
        
        # 计算所有相关系数
        correlations = self.calculate_all_correlations(self.intelligence, current_values)
        
        self.history['pearson_correlations'].append(correlations['pearson'])
        self.history['spearman_correlations'].append(correlations['spearman'])
        self.history['kendall_correlations'].append(correlations['kendall'])
        
        # 每20次迭代汇报一次相关系数
        if (iteration + 1) % 20 == 0:
            self._report_correlations(iteration + 1, correlations)
    
    def _report_correlations(self, iteration, correlations):
        """汇报相关系数"""
        print(f"迭代 {iteration:3d} - 相关系数: "
              f"皮尔逊={correlations['pearson']:.4f}, "
              f"斯皮尔曼={correlations['spearman']:.4f}, "
              f"肯德尔={correlations['kendall']:.4f}")
    
    def run_simulation(self, verbose=True):
        """运行完整模拟"""
        if verbose:
            print(f"开始模拟: {self.num_individuals}个个体, {self.num_iterations}次迭代")
            print("相关系数说明:")
            print("  皮尔逊: 线性相关 [-1,1]")
            print("  斯皮尔曼: 排序相关 [-1,1]")
            print("  肯德尔: 排序和谐性 [-1,1]")
            print("-" * 80)
        
        for i in range(self.num_iterations):
            self.simulate_iteration(i)
            
            if verbose and (i + 1) % 20 == 0:
                avg_val = self.history['avg_values'][-1]
                median_val = self.history['median_values'][-1]
                gini_val = self.history['gini_coefficients'][-1]
                print(f"迭代 {i+1:3d}/{self.num_iterations} | "
                      f"平均势力: {avg_val:8.2f} | "
                      f"中位数: {median_val:8.2f} | "
                      f"基尼系数: {gini_val:.4f}")
        
        # 计算最终统计
        self._calculate_final_stats()
        
        if verbose:
            self.print_summary()
    
    def _calculate_final_stats(self):
        """计算最终统计信息"""
        final_values = self.values
        
        # 计算最终相关系数
        final_correlations = self.calculate_all_correlations(self.intelligence, final_values)
        
        # 计算十分位数
        deciles = np.percentile(final_values, range(0, 101, 10))
        
        # 计算排名数据
        intel_ranks = stats.rankdata(self.intelligence, method='average')
        value_ranks = stats.rankdata(final_values, method='average')
        rank_diff = intel_ranks - value_ranks
        
        self.final_stats = {
            'mean_value': np.mean(final_values),
            'median_value': np.median(final_values),
            'std_value': np.std(final_values),
            'min_value': np.min(final_values),
            'max_value': np.max(final_values),
            'gini': self._calculate_gini(final_values),
            
            # 相关系数
            'pearson_correlation': final_correlations['pearson'],
            'spearman_correlation': final_correlations['spearman'],
            'kendall_correlation': final_correlations['kendall'],
            
            # 分位数
            'percentile_10': np.percentile(final_values, 10),
            'percentile_90': np.percentile(final_values, 90),
            
            # 十分位数
            'deciles': deciles,
            
            # 排名差异统计
            'mean_rank_diff': np.mean(rank_diff),
            'median_rank_diff': np.median(rank_diff),
            'std_rank_diff': np.std(rank_diff)
        }
    
    def print_summary(self):
        """打印模拟总结"""
        print("\n" + "="*80)
        print("模拟结果总结")
        print("="*80)
        
        stats = self.final_stats
        
        print(f"\n整体统计:")
        print(f"  平均势力值: {stats['mean_value']:.2f}")
        print(f"  中位数势力值: {stats['median_value']:.2f}")
        print(f"  标准差: {stats['std_value']:.2f}")
        print(f"  范围: {stats['min_value']:.2f} - {stats['max_value']:.2f}")
        print(f"  基尼系数: {stats['gini']:.4f}")
        
        print(f"\n分位数:")
        print(f"  10%分位数: {stats['percentile_10']:.2f}")
        print(f"  90%分位数: {stats['percentile_90']:.2f}")
        print(f"  平均值/中位数比率: {stats['mean_value']/stats['median_value']:.2f}")
        
        print(f"\n最终相关系数:")
        print(f"  皮尔逊相关系数 (线性相关): {stats['pearson_correlation']:.4f}")
        print(f"  斯皮尔曼等级相关系数 (排序相关): {stats['spearman_correlation']:.4f}")
        print(f"  肯德尔等级相关系数 (排序和谐性): {stats['kendall_correlation']:.4f}")
        
        # 解释相关系数
        print(f"\n相关系数解释:")
        
        for corr_type, corr_value in [('皮尔逊', stats['pearson_correlation']),
                                     ('斯皮尔曼', stats['spearman_correlation']),
                                     ('肯德尔', stats['kendall_correlation'])]:
            
            if abs(corr_value) > 0.7:
                strength = "强"
            elif abs(corr_value) > 0.4:
                strength = "中等"
            elif abs(corr_value) > 0.2:
                strength = "弱"
            else:
                strength = "极弱或无"
            
            direction = "正" if corr_value > 0 else "负"
            
            print(f"  {corr_type}: {strength}{direction}相关 ({corr_value:.4f})")
    
    def plot_chart_set_1(self):
        """绘制第一组图表（4个子图）"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('世界模型模拟结果 - 图表组1', fontsize=16, fontweight='bold')
        
        iterations = range(1, self.num_iterations + 1)
        
        # 图表1：势力值随时间变化 - 修正：使用实际分位数
        ax1 = plt.subplot(2, 2, 1)
        
        avg_line, = ax1.plot(iterations, self.history['avg_values'], 
                            'b-', linewidth=2, label='平均值', alpha=0.8)
        median_line, = ax1.plot(iterations, self.history['median_values'],
                               'r-', linewidth=2, label='中位数', alpha=0.8)
        
        # 使用实际分位数而不是标准差近似
        ax1.fill_between(iterations, 
                        self.history['percentile_10'], 
                        self.history['percentile_90'],
                        alpha=0.15, color='blue', label='10%-90%范围')
        
        ax1.set_xlabel('迭代次数', fontsize=10)
        ax1.set_ylabel('势力值', fontsize=10)
        ax1.set_title('1. 势力值随时间变化', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
        
        # 自动调整x轴标签避免重叠
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        # 图表2：三种相关系数对比
        ax2 = plt.subplot(2, 2, 2)
        
        ax2.plot(iterations, self.history['pearson_correlations'], 
                'b-', linewidth=1.5, label='皮尔逊', alpha=0.8)
        ax2.plot(iterations, self.history['spearman_correlations'], 
                'g-', linewidth=1.5, label='斯皮尔曼', alpha=0.8)
        ax2.plot(iterations, self.history['kendall_correlations'], 
                'orange', linewidth=1.5, label='肯德尔', alpha=0.8)
        
        ax2.set_xlabel('迭代次数', fontsize=10)
        ax2.set_ylabel('相关系数', fontsize=10)
        ax2.set_title('2. 三种相关系数对比', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # 自动调整x轴标签
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        # 图表3：智力与最终势力值散点图 - 修正：使用正确的排名
        ax3 = plt.subplot(2, 2, 3)
        
        # 计算正确的排名
        intel_ranks = stats.rankdata(self.intelligence, method='average')
        value_ranks = stats.rankdata(self.values, method='average')
        
        # 排名差异（绝对值越小，相关性越高）
        rank_diff = np.abs(intel_ranks - value_ranks)
        
        # 用排名差异着色
        scatter = ax3.scatter(self.intelligence, self.values, 
                             c=rank_diff, cmap='RdBu_r', 
                             alpha=0.5, s=5, edgecolors='none')
        
        ax3.set_xlabel('智力值', fontsize=10)
        ax3.set_ylabel('最终势力值', fontsize=10)
        ax3.set_title(f'3. 智力 vs 势力 (斯皮尔曼={self.final_stats["spearman_correlation"]:.3f})', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax3, label='排名差异', shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        
        # 图表4：基尼系数变化
        ax4 = plt.subplot(2, 2, 4)
        
        ax4.plot(iterations, self.history['gini_coefficients'], 
                'r-', linewidth=2, alpha=0.8)
        ax4.set_xlabel('迭代次数', fontsize=10)
        ax4.set_ylabel('基尼系数', fontsize=10)
        ax4.set_title('4. 不平等程度变化(基尼系数)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, 
                   label='警戒线(0.3)')
        ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, 
                   label='危险线(0.4)')
        ax4.legend(loc='lower right', fontsize=9)
        
        # 自动调整x轴标签
        ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        plt.tight_layout()
        plt.show()
    
    def plot_chart_set_2(self):
        """绘制第二组图表（4个子图）"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('世界模型模拟结果 - 图表组2', fontsize=16, fontweight='bold')
        
        # 图表5：最终势力值分布
        ax1 = plt.subplot(2, 2, 1)
        
        # 对势力值取对数，以便更好地显示分布
        log_values = np.log10(self.values + 1)  # +1避免log(0)
        
        ax1.hist(log_values, bins=50, color='steelblue', 
                edgecolor='black', alpha=0.7, density=True)
        
        # 计算对数平均值和中位数
        mean_log = np.mean(log_values)
        median_log = np.median(log_values)
        
        ax1.axvline(mean_log, color='red', linestyle='--', 
                   label=f'均值: {10**mean_log:.1f}(取对数后{mean_log:.2f})')
        ax1.axvline(median_log, color='orange', linestyle='--',
                   label=f'中位数: {10**median_log:.1f}(取对数后{median_log:.2f})')
        
        ax1.set_xlabel('势力值(对数尺度)', fontsize=10)
        ax1.set_ylabel('概率密度', fontsize=10)
        ax1.set_title('5. 最终势力值分布(对数尺度)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper right', fontsize=9)
        
        # 图表6：智力值分布
        ax2 = plt.subplot(2, 2, 2)
        
        ax2.hist(self.intelligence, bins=30, color='darkorange', 
                edgecolor='black', alpha=0.7, density=True)
        
        # 添加正态分布拟合
        intel_mean = np.mean(self.intelligence)
        intel_std = np.std(self.intelligence)
        x = np.linspace(np.min(self.intelligence), np.max(self.intelligence), 100)
        y = stats.norm.pdf(x, intel_mean, intel_std)
        ax2.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, label='正态分布拟合')
        
        ax2.axvline(x=100, color='green', linestyle='--', 
                   label=f'目标均值: 100', alpha=0.7)
        ax2.axvline(x=intel_mean, color='red', linestyle='--',
                   label=f'实际均值: {intel_mean:.1f}', alpha=0.7)
        
        ax2.set_xlabel('智力值', fontsize=10)
        ax2.set_ylabel('概率密度', fontsize=10)
        ax2.set_title('6. 智力值分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='upper left', fontsize=9)
        
        # 图表7：平均值与中位数差距
        ax3 = plt.subplot(2, 2, 3)
        
        iterations = range(1, self.num_iterations + 1)
        gaps = np.array(self.history['avg_values']) - np.array(self.history['median_values'])
        
        ax3.plot(iterations, gaps, 'purple', linewidth=2, alpha=0.8)
        ax3.fill_between(iterations, 0, gaps, 
                        where=(gaps > 0), 
                        color='red', alpha=0.3, label='平均值 > 中位数')
        ax3.fill_between(iterations, 0, gaps, 
                        where=(gaps < 0), 
                        color='blue', alpha=0.3, label='平均值 < 中位数')
        
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax3.set_xlabel('迭代次数', fontsize=10)
        ax3.set_ylabel('平均值 - 中位数', fontsize=10)
        ax3.set_title('7. 平均值与中位数差距', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left', fontsize=9)
        
        # 自动调整x轴标签
        ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        # 图表8：排名差异分布 - 修正：使用正确的排名
        ax4 = plt.subplot(2, 2, 4)
        
        # 计算正确的排名
        intel_ranks = stats.rankdata(self.intelligence, method='average')
        value_ranks = stats.rankdata(self.values, method='average')
        
        # 排名差异
        rank_diff = intel_ranks - value_ranks
        
        ax4.hist(rank_diff, bins=50, color='purple', alpha=0.7, edgecolor='black', density=True)
        ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        ax4.axvline(x=np.mean(rank_diff), color='red', linestyle='--', 
                   label=f'平均差异: {np.mean(rank_diff):.1f}')
        ax4.axvline(x=np.median(rank_diff), color='orange', linestyle='--',
                   label=f'中位差异: {np.median(rank_diff):.1f}')
        
        ax4.set_xlabel('智力排名 - 势力排名', fontsize=10)
        ax4.set_ylabel('概率密度', fontsize=10)
        ax4.set_title('8. 排名差异分布', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def print_chart_explanations(self):
        """打印图表解释"""
        print("\n" + "="*80)
        print("图表详细解释")
        print("="*80)
        
        print("\n图表组1:")
        print("1. 势力值随时间变化:")
        print("   - 蓝线: 平均势力值(被少数极高值拉高)")
        print("   - 红线: 中位数势力值(大多数个体的状况)")
        print("   - 蓝区: 10%-90%分位数范围(实际分布)")
        
        print("\n2. 三种相关系数对比:")
        print("   - 蓝线: 皮尔逊相关系数(线性相关)")
        print("   - 绿线: 斯皮尔曼相关系数(排序相关)")
        print("   - 橙线: 肯德尔相关系数(排序和谐性)")
        
        print("\n3. 智力 vs 势力:")
        print("   - 散点图: 每个点代表一个个体")
        print("   - 颜色: 红(排名接近) → 蓝(排名差异大)")
        print("   - 对角线趋势: 智力越高，势力越高")
        
        print("\n4. 不平等程度变化(基尼系数):")
        print("   - 红线: 基尼系数随时间变化")
        print("   - 橙线: 警戒线(0.3)")
        print("   - 红线: 危险线(0.4)")
        
        print("\n图表组2:")
        print("5. 最终势力值分布(对数尺度):")
        print("   - 使用对数尺度显示极度右偏的分布")
        print("   - 大多数个体势力值很低(左侧)")
        print("   - 少数个体势力值极高(右侧长尾)")
        
        print("\n6. 智力值分布:")
        print("   - 近似正态分布")
        print("   - 蓝线: 正态分布拟合")
        print("   - 绿线: 目标均值(100)")
        print("   - 红线: 实际均值")
        
        print("\n7. 平均值与中位数差距:")
        print("   - 紫线: 平均值与中位数的差值")
        print("   - 红区: 平均值大于中位数(右偏分布)")
        print("   - 蓝区: 平均值小于中位数(左偏分布)")
        
        print("\n8. 排名差异分布:")
        print("   - 直方图: 智力排名与势力排名的差异")
        print("   - 正值: 智力排名高于势力排名(怀才不遇)")
        print("   - 负值: 智力排名低于势力排名(幸运儿)")
        
        print("\n关键洞察:")
        print("1. 乘性更新产生更强的复利效应，导致更极端的不平等")
        print("2. 使用实际分位数比标准差近似更准确地反映分布")
        print("3. 正确的排名计算使相关性分析更可靠")


def main():
    """主函数"""
    print("世界模型模拟器 - 相关系数改进版(修正后)")
    print("="*80)
    
    # 创建模型（包含随机种子）
    model = WorldModel(
        num_individuals=10000,
        initial_value=100,
        num_iterations=200,
        random_seed=42  # 添加随机种子参数
    )
    
    # 运行模拟
    print("开始模拟...")
    model.run_simulation(verbose=True)
    
    # 生成第一组图表
    print("\n生成第一组图表...")
    model.plot_chart_set_1()
    
    # 生成第二组图表
    print("\n生成第二组图表...")
    model.plot_chart_set_2()
    
    # 打印图表解释
    model.print_chart_explanations()
    
    # 总结
    print("\n" + "="*80)
    print("关键发现总结")
    print("="*80)
    
    stats = model.final_stats
    
    # 计算一些关键指标
    mean_median_ratio = stats['mean_value'] / stats['median_value'] if stats['median_value'] > 0 else float('inf')
    top10_share = np.sum(model.values >= stats['percentile_90']) / model.num_individuals * 100
    
    print(f"1. 极端不平等: 基尼系数={stats['gini']:.4f} (极度不平等)")
    print(f"2. 平均值失真: 平均值/中位数比率={mean_median_ratio:.1f}倍")
    print(f"3. 智力与势力强相关: 斯皮尔曼系数={stats['spearman_correlation']:.4f}")
    print(f"4. 顶部集中: 前{top10_share:.1f}%个体拥有超过90%分位数的势力值")
    print(f"5. 乘性效应: 70%运气×30%智力产生指数级分化")


if __name__ == "__main__":
    main()