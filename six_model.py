import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import matplotlib
from collections import defaultdict

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
    def __init__(self, num_individuals=10000, initial_value=100, num_iterations=200, 
                 random_seed=42, average_death_age=40, inheritance_type="direct"):
        """
        初始化世界模型
        
        参数:
        num_individuals: 个体数量
        initial_value: 初始势力值
        num_iterations: 迭代次数
        random_seed: 随机种子
        average_death_age: 平均死亡年龄（迭代次数）
        inheritance_type: 继承类型 ("direct": 直接继承, "redistribute": 部分重新分配)
        """
        self.num_individuals = num_individuals
        self.initial_value = initial_value
        self.num_iterations = num_iterations
        self.average_death_age = average_death_age
        self.inheritance_type = inheritance_type
        
        # 设置随机种子以确保可重复性
        np.random.seed(random_seed)
        
        # 生成初始智力值
        self.intelligence = self._generate_intelligence(num_individuals)
        
        # 初始化势力值
        self.values = np.full(num_individuals, initial_value, dtype=np.float64)
        
        # 初始化年龄和死亡时间
        self.ages = np.zeros(num_individuals, dtype=int)  # 当前年龄
        self.death_times = self._generate_death_times(num_individuals, average_death_age)  # 死亡时间
        
        # 跟踪血统信息
        self.generations = np.zeros(num_individuals, dtype=int)  # 代数
        
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
            'percentile_10': [],
            'percentile_90': [],
            'death_counts': [],  # 每轮死亡数量
            'birth_counts': [],  # 每轮新生数量
            'avg_age': [],  # 平均年龄
            'avg_generation': [],  # 平均代数
            'intelligence_means': [],  # 记录智力平均值变化
            'intelligence_stds': [],   # 记录智力标准差变化
            'age_distribution': []  # 年龄分布快照
        }
        
        # 记录初始智力统计
        self.history['intelligence_means'].append(np.mean(self.intelligence))
        self.history['intelligence_stds'].append(np.std(self.intelligence))
        
        self.final_stats = {}
        
        # 记录总死亡和新生数量
        self.total_deaths = 0
        self.total_births = 0
    
    def _generate_intelligence(self, n):
        """生成智力值 - 使用截断正态分布"""
        mean, std = 100, 25
        min_val, max_val = 50, 150
        
        # 使用截断正态分布生成样本
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
        
        return samples
    
    def _generate_death_times(self, n, average_death_age):
        """生成死亡时间 - 使用指数分布，最小存活1次迭代"""
        # 指数分布模拟随机死亡时间
        # scale参数是平均值，但指数分布有长尾
        death_times = np.random.exponential(scale=average_death_age, size=n)
        
        # 确保死亡时间至少为1（不能出生就死亡）
        death_times = np.maximum(death_times, 1)
        
        # 四舍五入为整数
        death_times = np.round(death_times).astype(int)
        
        return death_times
    
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
    
    def _death_and_inheritance(self, indices):
        """死亡-继承机制：重新分配智力值，指定个体死亡"""
        if len(indices) == 0:
            return
            
        death_count = len(indices)
        self.total_deaths += death_count
        
        # 记录哪些血统死亡了
        dead_generations = self.generations[indices]
        
        if self.inheritance_type == "direct":
            # 直接继承：保持势力值不变，重新生成智力值
            self.intelligence[indices] = self._generate_intelligence(death_count)
            
        elif self.inheritance_type == "redistribute":
            # 部分重新分配：添加一些随机性，模拟遗产税或重新分配
            # 保留70%的势力值，30%重新分配
            total_wealth = np.sum(self.values[indices])
            redistribution = total_wealth * 0.3
            
            # 给每个死亡个体添加随机的小额重新分配
            random_redistribution = np.random.dirichlet(np.ones(death_count)) * redistribution
            self.values[indices] = self.values[indices] * 0.7 + random_redistribution
            
            # 重新生成智力值
            self.intelligence[indices] = self._generate_intelligence(death_count)
            
        elif self.inheritance_type == "meritocratic":
            # 精英继承：高智力后代继承更多
            # 根据当前智力值（作为遗传因素）决定继承比例
            genetic_factor = (self.intelligence[indices] - 50) / 100  # 标准化到0-1
            
            # 使用遗传因子作为权重重新分配
            weights = genetic_factor + 0.1  # 确保权重为正
            weights = weights / np.sum(weights)  # 归一化
            
            # 重新分配财富
            total_wealth = np.sum(self.values[indices])
            self.values[indices] = weights * total_wealth
            
            # 重新生成智力值（但保持一定的遗传性）
            # 新智力 = 遗传因子 * 父母智力 + (1-遗传因子) * 随机智力
            parent_factor = 0.3  # 30%受父母智力影响
            new_random_intelligence = self._generate_intelligence(death_count)
            self.intelligence[indices] = parent_factor * self.intelligence[indices] + (1-parent_factor) * new_random_intelligence
        
        # 重置年龄
        self.ages[indices] = 0
        
        # 生成新的死亡时间
        self.death_times[indices] = self._generate_death_times(death_count, self.average_death_age)
        
        # 代数增加
        self.generations[indices] += 1
        
        self.total_births += death_count
        
        return death_count
    
    def simulate_iteration(self, iteration):
        """执行一次迭代计算"""
        # 增加所有个体的年龄
        self.ages += 1
        
        # 检查哪些个体死亡
        death_indices = np.where(self.ages >= self.death_times)[0]
        death_count = 0
        
        # 执行死亡-继承机制
        if len(death_indices) > 0:
            death_count = self._death_and_inheritance(death_indices)
        
        # 运气因素
        luck_multiplier = np.where(
            np.random.random(self.num_individuals) > 0.5,
            1.1,  # 好运
            0.9   # 坏运
        )
        
        # 智力因素
        intelligence_factor = 0.9 + (self.intelligence - 50) * 0.2 / 100
        
        # 综合计算 - 使用乘性组合
        combined_factor = np.power(luck_multiplier, 0.7) * np.power(intelligence_factor, 0.3)
        
        # 更新势力值
        self.values = self.values * combined_factor
        
        # 记录统计信息
        self._record_stats(iteration, death_count)
    
    def _record_stats(self, iteration, death_count):
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
        
        # 记录死亡和新生命数据
        self.history['death_counts'].append(death_count)
        self.history['birth_counts'].append(death_count)  # 每个死亡对应一个新生
        
        # 记录年龄和代数统计
        self.history['avg_age'].append(np.mean(self.ages))
        self.history['avg_generation'].append(np.mean(self.generations))
        
        # 记录年龄分布（每20次迭代记录一次）
        if iteration % 20 == 0:
            # 记录年龄分布的直方图数据
            age_counts, age_bins = np.histogram(self.ages, bins=10, range=(0, 100))
            self.history['age_distribution'].append((age_counts, age_bins))
        
        # 记录智力统计（每20次迭代记录一次）
        if iteration % 20 == 0:
            self.history['intelligence_means'].append(np.mean(self.intelligence))
            self.history['intelligence_stds'].append(np.std(self.intelligence))
        
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
        avg_age = np.mean(self.ages)
        avg_generation = np.mean(self.generations)
        
        print(f"迭代 {iteration:3d} - "
              f"年龄: {avg_age:.1f}, 代数: {avg_generation:.1f} | "
              f"皮尔逊={correlations['pearson']:.4f}, "
              f"斯皮尔曼={correlations['spearman']:.4f}, "
              f"肯德尔={correlations['kendall']:.4f}")
    
    def run_simulation(self, verbose=True):
        """运行完整模拟"""
        if verbose:
            print(f"开始模拟: {self.num_individuals}个个体, {self.num_iterations}次迭代")
            print(f"死亡机制: 个体随机死亡，平均寿命={self.average_death_age}次迭代")
            print(f"继承类型: {self.inheritance_type}")
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
                death_count = self.history['death_counts'][-1]
                print(f"迭代 {i+1:3d}/{self.num_iterations} | "
                      f"平均势力: {avg_val:8.2f} | "
                      f"中位数: {median_val:8.2f} | "
                      f"基尼系数: {gini_val:.4f} | "
                      f"死亡数: {death_count}")
        
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
        
        # 分析死亡统计
        death_counts = np.array(self.history['death_counts'])
        avg_death_per_iteration = np.mean(death_counts) if len(death_counts) > 0 else 0
        
        # 计算各代人的势力值
        generation_stats = {}
        for gen in np.unique(self.generations):
            indices = np.where(self.generations == gen)[0]
            if len(indices) > 0:
                generation_stats[gen] = {
                    'count': len(indices),
                    'mean_value': np.mean(final_values[indices]),
                    'median_value': np.median(final_values[indices]),
                    'mean_intelligence': np.mean(self.intelligence[indices])
                }
        
        # 计算代际流动性
        intergenerational_mobility = 0
        if len(generation_stats) > 1:
            # 计算不同代之间的财富相关性
            gens = list(generation_stats.keys())
            gen_values = [generation_stats[gen]['mean_value'] for gen in gens]
            # 简单的代际变化率（从一代到下一代）
            mobility_rates = []
            for i in range(len(gen_values)-1):
                if gen_values[i] > 0:
                    rate = (gen_values[i+1] - gen_values[i]) / gen_values[i]
                    mobility_rates.append(rate)
            if mobility_rates:
                intergenerational_mobility = np.mean(mobility_rates)
        
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
            'std_rank_diff': np.std(rank_diff),
            
            # 死亡统计
            'total_deaths': self.total_deaths,
            'total_births': self.total_births,
            'avg_death_per_iteration': avg_death_per_iteration,
            'avg_final_age': np.mean(self.ages),
            'avg_final_generation': np.mean(self.generations),
            
            # 代际统计
            'num_generations': len(generation_stats),
            'intergenerational_mobility': intergenerational_mobility,
            'generation_stats': generation_stats
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
        
        print(f"\n人口动态:")
        print(f"  总死亡数: {stats['total_deaths']}")
        print(f"  总新生数: {stats['total_births']}")
        print(f"  平均每轮死亡数: {stats['avg_death_per_iteration']:.2f}")
        print(f"  最终平均年龄: {stats['avg_final_age']:.1f}")
        print(f"  最终平均代数: {stats['avg_final_generation']:.1f}")
        print(f"  总代数: {stats['num_generations']}")
        
        print(f"\n代际流动性:")
        print(f"  代际财富变化率: {stats['intergenerational_mobility']:.4f}")
        if stats['intergenerational_mobility'] > 0:
            print(f"  趋势: 后代比前代更富有")
        else:
            print(f"  趋势: 后代比前代更贫穷")
        
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
        fig.suptitle(f'世界模型模拟结果 - 图表组1 (平均寿命: {self.average_death_age})', 
                    fontsize=16, fontweight='bold')
        
        iterations = range(1, self.num_iterations + 1)
        
        # 图表1：势力值随时间变化
        ax1 = plt.subplot(2, 2, 1)
        
        avg_line, = ax1.plot(iterations, self.history['avg_values'], 
                            'b-', linewidth=2, label='平均值', alpha=0.8)
        median_line, = ax1.plot(iterations, self.history['median_values'],
                               'r-', linewidth=2, label='中位数', alpha=0.8)
        
        # 使用实际分位数
        ax1.fill_between(iterations, 
                        self.history['percentile_10'], 
                        self.history['percentile_90'],
                        alpha=0.15, color='blue', label='10%-90%范围')
        
        # 添加死亡阴影
        death_counts = np.array(self.history['death_counts'])
        if np.max(death_counts) > 0:
            # 归一化死亡数量用于透明度
            death_norm = death_counts / np.max(death_counts)
            ax1.fill_between(iterations, 
                           ax1.get_ylim()[0], 
                           ax1.get_ylim()[1],
                           where=(death_counts > 0),
                           alpha=0.1, color='red', 
                           label='死亡事件', edgecolor='none')
        
        ax1.set_xlabel('迭代次数', fontsize=10)
        ax1.set_ylabel('势力值', fontsize=10)
        ax1.set_title('1. 势力值随时间变化 (红色阴影: 死亡事件强度)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        # 图表2：三种相关系数对比
        ax2 = plt.subplot(2, 2, 2)
        
        ax2.plot(iterations, self.history['pearson_correlations'], 
                'b-', linewidth=1.5, label='皮尔逊', alpha=0.8)
        ax2.plot(iterations, self.history['spearman_correlations'], 
                'g-', linewidth=1.5, label='斯皮尔曼', alpha=0.8)
        ax2.plot(iterations, self.history['kendall_correlations'], 
                'orange', linewidth=1.5, label='肯德尔', alpha=0.8)
        
        # 添加死亡阴影
        if np.max(death_counts) > 0:
            ax2.fill_between(iterations, 
                           ax2.get_ylim()[0], 
                           ax2.get_ylim()[1],
                           where=(death_counts > 0),
                           alpha=0.1, color='red', 
                           label='死亡事件', edgecolor='none')
        
        ax2.set_xlabel('迭代次数', fontsize=10)
        ax2.set_ylabel('相关系数', fontsize=10)
        ax2.set_title('2. 三种相关系数对比', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        # 图表3：智力与最终势力值散点图，按年龄着色
        ax3 = plt.subplot(2, 2, 3)
        
        # 计算正确的排名
        intel_ranks = stats.rankdata(self.intelligence, method='average')
        value_ranks = stats.rankdata(self.values, method='average')
        
        # 排名差异（绝对值越小，相关性越高）
        rank_diff = np.abs(intel_ranks - value_ranks)
        
        # 用年龄着色
        scatter = ax3.scatter(self.intelligence, self.values, 
                             c=self.ages, cmap='viridis', 
                             alpha=0.5, s=5, edgecolors='none')
        
        ax3.set_xlabel('智力值', fontsize=10)
        ax3.set_ylabel('最终势力值', fontsize=10)
        ax3.set_title(f'3. 智力 vs 势力 (按年龄着色)', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax3, label='年龄', shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        
        # 图表4：基尼系数变化
        ax4 = plt.subplot(2, 2, 4)
        
        ax4.plot(iterations, self.history['gini_coefficients'], 
                'r-', linewidth=2, alpha=0.8)
        
        # 添加死亡阴影
        if np.max(death_counts) > 0:
            ax4.fill_between(iterations, 
                           ax4.get_ylim()[0], 
                           ax4.get_ylim()[1],
                           where=(death_counts > 0),
                           alpha=0.1, color='red', 
                           label='死亡事件', edgecolor='none')
        
        ax4.set_xlabel('迭代次数', fontsize=10)
        ax4.set_ylabel('基尼系数', fontsize=10)
        ax4.set_title('4. 不平等程度变化(基尼系数)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, 
                   label='警戒线(0.3)')
        ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, 
                   label='危险线(0.4)')
        ax4.legend(loc='lower right', fontsize=9)
        ax4.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        plt.tight_layout()
        plt.show()
    
    def plot_chart_set_2(self):
        """绘制第二组图表（4个子图）"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'世界模型模拟结果 - 图表组2 (平均寿命: {self.average_death_age})', 
                    fontsize=16, fontweight='bold')
        
        iterations = range(1, self.num_iterations + 1)
        
        # 图表5：人口动态
        ax1 = plt.subplot(2, 2, 1)
        
        # 绘制死亡和新生数量
        death_counts = np.array(self.history['death_counts'])
        birth_counts = np.array(self.history['birth_counts'])
        
        ax1.plot(iterations, death_counts, 'r-', linewidth=1.5, label='死亡数', alpha=0.8)
        ax1.plot(iterations, birth_counts, 'g-', linewidth=1.5, label='新生数', alpha=0.8)
        
        # 绘制移动平均线以显示趋势
        window_size = min(20, len(death_counts)//4)
        if window_size > 1:
            death_moving_avg = np.convolve(death_counts, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(iterations[window_size-1:], death_moving_avg, 'r--', linewidth=2, label='死亡移动平均', alpha=0.7)
        
        ax1.set_xlabel('迭代次数', fontsize=10)
        ax1.set_ylabel('个体数量', fontsize=10)
        ax1.set_title('5. 人口动态 (死亡与新生)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        # 图表6：年龄和代数分布
        ax2 = plt.subplot(2, 2, 2)
        
        # 绘制年龄和代数
        ax2_ages = ax2.twinx()  # 创建第二个y轴
        
        ax2.plot(iterations, self.history['avg_age'], 'b-', linewidth=2, label='平均年龄', alpha=0.8)
        ax2_ages.plot(iterations, self.history['avg_generation'], 'orange', linewidth=2, label='平均代数', alpha=0.8)
        
        ax2.set_xlabel('迭代次数', fontsize=10)
        ax2.set_ylabel('平均年龄', fontsize=10, color='blue')
        ax2_ages.set_ylabel('平均代数', fontsize=10, color='orange')
        ax2.set_title('6. 人口年龄和代数变化', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_ages.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        # 图表7：最终势力值分布(对数尺度)，按代数分组
        ax3 = plt.subplot(2, 2, 3)
        
        # 对势力值取对数
        log_values = np.log10(self.values + 1)
        
        # 按代数分组显示
        unique_generations = np.unique(self.generations)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_generations)))
        
        for i, gen in enumerate(unique_generations):
            if i < 5:  # 只显示前5代，避免图表过于拥挤
                gen_indices = np.where(self.generations == gen)[0]
                if len(gen_indices) > 0:
                    gen_log_values = log_values[gen_indices]
                    ax3.hist(gen_log_values, bins=20, alpha=0.5, color=colors[i], 
                            edgecolor='black', density=True, label=f'第{gen}代')
        
        ax3.set_xlabel('势力值(对数尺度)', fontsize=10)
        ax3.set_ylabel('概率密度', fontsize=10)
        ax3.set_title('7. 各代势力值分布(对数尺度)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        if len(unique_generations) <= 5:
            ax3.legend(loc='upper right', fontsize=9)
        
        # 图表8：代际财富传递
        ax4 = plt.subplot(2, 2, 4)
        
        # 获取代际统计
        generation_stats = self.final_stats['generation_stats']
        if len(generation_stats) > 1:
            gens = sorted(generation_stats.keys())
            
            # 提取数据
            gen_numbers = []
            mean_values = []
            mean_intelligence = []
            counts = []
            
            for gen in gens:
                if gen <= 10:  # 只显示前10代
                    stats = generation_stats[gen]
                    gen_numbers.append(gen)
                    mean_values.append(stats['mean_value'])
                    mean_intelligence.append(stats['mean_intelligence'])
                    counts.append(stats['count'])
            
            if len(gen_numbers) > 1:
                # 绘制代际财富变化
                ax4.plot(gen_numbers, mean_values, 'b-o', linewidth=2, label='平均势力值', alpha=0.8)
                
                # 绘制代际智力变化（使用第二个y轴）
                ax4_intel = ax4.twinx()
                ax4_intel.plot(gen_numbers, mean_intelligence, 'r-o', linewidth=2, label='平均智力值', alpha=0.8)
                
                ax4.set_xlabel('代数', fontsize=10)
                ax4.set_ylabel('平均势力值', fontsize=10, color='blue')
                ax4_intel.set_ylabel('平均智力值', fontsize=10, color='red')
                ax4.set_title('8. 代际财富和智力传递', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                # 合并图例
                lines1, labels1 = ax4.get_legend_handles_labels()
                lines2, labels2 = ax4_intel.get_legend_handles_labels()
                ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
            else:
                ax4.text(0.5, 0.5, '数据不足\n显示代际传递', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12)
                ax4.set_title('8. 代际财富传递', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '只有一代人\n无代际传递', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('8. 代际财富传递', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def print_chart_explanations(self):
        """打印图表解释"""
        print("\n" + "="*80)
        print("图表详细解释")
        print("="*80)
        
        print(f"\n死亡机制设置:")
        print(f"  平均寿命: {self.average_death_age}次迭代")
        print(f"  继承类型: {self.inheritance_type}")
        print(f"  总死亡数: {self.total_deaths}")
        print(f"  总新生数: {self.total_births}")
        
        print("\n图表组1:")
        print("1. 势力值随时间变化:")
        print("   - 蓝线: 平均势力值")
        print("   - 红线: 中位数势力值")
        print("   - 蓝区: 10%-90%分位数范围")
        print("   - 红色阴影: 死亡事件发生的强度和频率")
        
        print("\n2. 三种相关系数对比:")
        print("   - 蓝线: 皮尔逊相关系数(线性相关)")
        print("   - 绿线: 斯皮尔曼相关系数(排序相关)")
        print("   - 橙线: 肯德尔相关系数(排序和谐性)")
        print("   - 红色阴影: 死亡事件发生的强度和频率")
        
        print("\n3. 智力 vs 势力:")
        print("   - 散点图: 每个点代表一个个体")
        print("   - 颜色: 年龄(深色年轻，浅色年老)")
        print("   - 对角线趋势: 智力越高，势力越高")
        
        print("\n4. 不平等程度变化(基尼系数):")
        print("   - 红线: 基尼系数随时间变化")
        print("   - 红色阴影: 死亡事件发生的强度和频率")
        print("   - 橙线: 警戒线(0.3)")
        print("   - 红线: 危险线(0.4)")
        
        print("\n图表组2:")
        print("5. 人口动态 (死亡与新生):")
        print("   - 红线: 每轮死亡个体数量")
        print("   - 绿线: 每轮新生个体数量")
        print("   - 红色虚线: 死亡数量的移动平均线")
        
        print("\n6. 人口年龄和代数变化:")
        print("   - 蓝线: 人口平均年龄")
        print("   - 橙线: 人口平均代数")
        print("   - 年龄增长反映个体存活时间")
        print("   - 代数增长反映世代更替")
        
        print("\n7. 各代势力值分布(对数尺度):")
        print("   - 直方图: 按不同代数分组")
        print("   - 不同颜色代表不同代数")
        print("   - 显示代际财富分布差异")
        
        print("\n8. 代际财富和智力传递:")
        print("   - 蓝线: 各代平均势力值变化")
        print("   - 红线: 各代平均智力值变化")
        print("   - 显示财富和智力的代际传递模式")
        
        print("\n关键洞察:")
        print("1. 个体随机死亡使社会动态更接近现实")
        print("2. 年龄结构影响社会不平等和流动性")
        print("3. 代际传递模式反映社会固化程度")
        print("4. 持续的新生代带来社会活力的变化")


def compare_lifespan_scenarios():
    """比较不同平均寿命场景的效果"""
    print("="*80)
    print("不同平均寿命场景比较")
    print("="*80)
    
    lifespan_scenarios = [20, 40, 80, 0]  # 0表示无死亡
    results = {}
    
    for lifespan in lifespan_scenarios:
        if lifespan == 0:
            print(f"\n场景: 无死亡 (永生)")
        else:
            print(f"\n场景: 平均寿命 {lifespan} 次迭代")
        
        model = WorldModel(
            num_individuals=5000,
            initial_value=100,
            num_iterations=120,
            random_seed=42,
            average_death_age=lifespan,
            inheritance_type="direct"
        )
        model.run_simulation(verbose=False)
        results[lifespan] = model.final_stats
        
        # 简要输出
        stats = model.final_stats
        print(f"  基尼系数: {stats['gini']:.4f}")
        print(f"  斯皮尔曼系数: {stats['spearman_correlation']:.4f}")
        print(f"  总代数: {stats['num_generations']}")
        print(f"  代际流动性: {stats['intergenerational_mobility']:.4f}")
    
    # 总结比较
    print("\n" + "="*80)
    print("平均寿命场景比较总结")
    print("="*80)
    
    for lifespan, stats in results.items():
        if lifespan == 0:
            scenario_name = "永生"
        else:
            scenario_name = f"寿命{lifespan}"
        
        print(f"\n{scenario_name}:")
        print(f"  不平等程度: {stats['gini']:.4f} (基尼系数)")
        print(f"  智力-势力相关性: {stats['spearman_correlation']:.4f} (斯皮尔曼)")
        print(f"  社会代际数: {stats['num_generations']}")
        print(f"  代际财富变化率: {stats['intergenerational_mobility']:.4f}")
    
    return results


def main():
    """主函数"""
    print("世界模型模拟器 - 个体随机死亡机制")
    print("="*80)
    
    print("\n选择运行模式:")
    print("1. 单次模拟 (个体随机死亡)")
    print("2. 比较不同平均寿命场景")
    print("3. 比较不同继承类型")
    print("4. 永生社会 (基准对照)")
    
    choice = input("\n请输入选择 (1-4): ")
    
    if choice == "1":
        # 单次模拟
        print("\n单次模拟 - 个体随机死亡")
        
        # 获取用户输入参数
        avg_lifespan = int(input("请输入平均寿命 (默认40): ") or "40")
        inheritance_type = input("请输入继承类型 (direct/redistribute/meritocratic, 默认direct): ") or "direct"
        
        model = WorldModel(
            num_individuals=10000,
            initial_value=100,
            num_iterations=200,
            random_seed=42,
            average_death_age=avg_lifespan,
            inheritance_type=inheritance_type
        )
        
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
        
    elif choice == "2":
        # 比较不同平均寿命场景
        results = compare_lifespan_scenarios()
        
    elif choice == "3":
        # 比较不同继承类型
        print("\n比较不同继承类型 (平均寿命40)")
        
        inheritance_types = ["direct", "redistribute", "meritocratic"]
        for inheritance_type in inheritance_types:
            print(f"\n继承类型: {inheritance_type}")
            model = WorldModel(
                num_individuals=5000,
                initial_value=100,
                num_iterations=120,
                random_seed=42,
                average_death_age=40,
                inheritance_type=inheritance_type
            )
            model.run_simulation(verbose=False)
            
            stats = model.final_stats
            print(f"  基尼系数: {stats['gini']:.4f}")
            print(f"  斯皮尔曼系数: {stats['spearman_correlation']:.4f}")
            print(f"  代际流动性: {stats['intergenerational_mobility']:.4f}")
        
    elif choice == "4":
        # 永生社会 (基准对照)
        print("\n基准对照 - 永生社会 (无死亡)")
        
        model = WorldModel(
            num_individuals=10000,
            initial_value=100,
            num_iterations=200,
            random_seed=42,
            average_death_age=0,  # 无死亡
            inheritance_type="direct"
        )
        
        print("开始模拟...")
        model.run_simulation(verbose=True)
        
        # 生成图表
        print("\n生成图表...")
        model.plot_chart_set_1()
        model.plot_chart_set_2()
        
        # 打印图表解释
        model.print_chart_explanations()
    
if __name__ == "__main__":
    main()