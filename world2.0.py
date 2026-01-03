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
    def __init__(self, num_individuals=100000, initial_value=100, num_iterations=400,
                 random_seed=42, average_death_age=40, inheritance_type="direct",
                 growth_type="bounded", saturation_point=1000, difficulty_factor=0.5):
        """
        初始化世界模型

        参数:
        num_individuals: 个体数量
        initial_value: 初始势力值 (表示个人财富，地位，声望等的一个抽象的值)
        num_iterations: 迭代次数 (改为400)
        random_seed: 随机种子
        average_death_age: 平均死亡年龄（迭代次数），0表示无死亡
        inheritance_type: 继承类型 ("direct": 直接继承, "redistribute": 部分重新分配)
        growth_type: 增长类型 ("exponential": 指数增长, "bounded": 边际递减, "logistic": S型增长)
        saturation_point: 饱和点（边际递减开始显著的值）
        difficulty_factor: 困难因子（值越大，增长越困难）
        """
        self.num_individuals = num_individuals
        self.initial_value = initial_value
        self.num_iterations = num_iterations
        self.average_death_age = average_death_age
        self.inheritance_type = inheritance_type
        self.growth_type = growth_type
        self.saturation_point = saturation_point
        self.difficulty_factor = difficulty_factor
        
        # 设置随机种子以确保可重复性
        np.random.seed(random_seed)
        
        # 生成初始能力值
        self.ability = self._generate_ability(num_individuals)
        
        # 初始化势力值 (表示个人财富，地位，声望等的一个抽象的值)
        self.values = np.full(num_individuals, initial_value, dtype=np.float64)
        
        # 检查是否启用死亡机制
        self.death_enabled = average_death_age > 0
        
        if self.death_enabled:
            # 初始化年龄和死亡时间
            self.ages = np.zeros(num_individuals, dtype=int)  # 当前年龄
            self.death_times = self._generate_death_times(num_individuals, average_death_age)  # 死亡时间
            
            # 跟踪血统信息
            self.generations = np.zeros(num_individuals, dtype=int)  # 代数
        else:
            # 永生社会：不启用死亡机制
            self.ages = np.zeros(num_individuals, dtype=int)  # 所有个体年龄都为0或固定值
            self.death_times = np.full(num_individuals, np.inf, dtype=float)  # 死亡时间为无穷大
            self.generations = np.zeros(num_individuals, dtype=int)  # 所有人都是第0代
        
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
            'ability_means': [],  # 记录能力平均值变化
            'ability_stds': [],   # 记录能力标准差变化
            'age_distribution': [],  # 年龄分布快照
            'growth_rates': []  # 记录平均增长率
        }
        
        # 记录初始能力统计
        self.history['ability_means'].append(np.mean(self.ability))
        self.history['ability_stds'].append(np.std(self.ability))
        
        self.final_stats = {}
        
        # 记录总死亡和新生数量
        self.total_deaths = 0
        self.total_births = 0
    
    def _generate_ability(self, n):
        """生成能力值 - 使用截断正态分布"""
        mean, std = 100, 25
        min_val, max_val = 50, 150
        
        # 使用截断正态分布生成样本
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
        
        return samples
    
    def _generate_death_times(self, n, average_death_age):
        """生成死亡时间 - 使用指数分布，最小存活1次迭代"""
        if average_death_age <= 0:
            return np.full(n, np.inf)
        
        # 指数分布模拟随机死亡时间
        # scale参数是平均值，但指数分布有长尾
        death_times = np.random.exponential(scale=average_death_age, size=n)
        
        # 确保死亡时间至少为1（不能出生就死亡）
        death_times = np.maximum(death_times, 1)
        
        # 四舍五入为整数
        death_times = np.round(death_times).astype(int)
        
        return death_times
    
    def _calculate_growth_factor(self, ability_factor, luck_multiplier, current_values):
        """计算增长因子，考虑边际效益递减"""
        # 基本因子
        if self.growth_type == "exponential":
            # 原始的指数增长模型
            combined_factor = np.power(luck_multiplier, 0.7) * np.power(ability_factor, 0.3)
            return combined_factor
            
        elif self.growth_type == "bounded":
            # 边际递减增长模型
            combined_factor = np.power(luck_multiplier, 0.7) * np.power(ability_factor, 0.3)
            
            # 计算边际递减因子
            # 使用sigmoid函数实现边际递减
            # 当值很小时，增长容易；当值很大时，增长困难
            normalized_values = current_values / self.saturation_point
            difficulty = 1.0 / (1.0 + np.exp(-self.difficulty_factor * (normalized_values - 2.0)))
            
            # 调整因子：在0.8到1.2之间变化
            adjustment = 0.8 + 0.4 * (1.0 - difficulty)
            
            # 最终增长因子
            bounded_factor = np.power(combined_factor, adjustment)
            
            return bounded_factor
            
        elif self.growth_type == "logistic":
            # S型增长模型（逻辑斯蒂增长）
            combined_factor = np.power(luck_multiplier, 0.7) * np.power(ability_factor, 0.3)
            
            # 逻辑斯蒂增长：当值接近饱和点时增长变慢
            normalized_values = current_values / self.saturation_point
            
            # 避免除以零
            normalized_values = np.clip(normalized_values, 0.001, 0.999)
            
            # 逻辑斯蒂调整因子
            logistic_adjustment = 1.0 - normalized_values
            
            # 确保调整因子在合理范围内
            logistic_adjustment = np.clip(logistic_adjustment, 0.3, 1.0)
            
            # 最终增长因子
            logistic_factor = np.power(combined_factor, logistic_adjustment)
            
            return logistic_factor
            
        elif self.growth_type == "symmetric":
            # 对称增长模型：特别小和特别大都难增长，中等水平最容易增长
            combined_factor = np.power(luck_multiplier, 0.7) * np.power(ability_factor, 0.3)
            
            # 计算与理想值的距离（取对数使对称更明显）
            # 理想值设为饱和点的一半
            ideal_value = self.saturation_point / 2
            
            # 使用对数尺度计算距离
            log_current = np.log10(current_values + 1)  # 加1避免log(0)
            log_ideal = np.log10(ideal_value + 1)
            
            # 距离（绝对值）
            distance = np.abs(log_current - log_ideal)
            
            # 困难度：距离越大，增长越困难
            max_distance = np.max([np.abs(np.log10(1) - log_ideal), 
                                  np.abs(np.log10(self.saturation_point * 10) - log_ideal)])
            normalized_distance = distance / max_distance
            
            # 调整因子：距离为0时最容易增长，距离最大时最难增长
            adjustment = 1.0 - 0.5 * normalized_distance
            
            # 最终增长因子
            symmetric_factor = np.power(combined_factor, adjustment)
            
            return symmetric_factor
            
        else:
            # 默认使用边际递减模型
            return self._calculate_growth_factor(ability_factor, luck_multiplier, current_values)
    
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
    
    def calculate_spearman_correlation(self, ability, values):
        """计算斯皮尔曼等级相关系数 - 使用scipy函数"""
        try:
            # 使用scipy的spearmanr函数，它处理了相同排名的情况
            correlation, p_value = stats.spearmanr(ability, values, nan_policy='omit')
            return correlation if not np.isnan(correlation) else 0.0
        except:
            # 备用方法：使用排名后的皮尔逊相关系数
            ability_ranks = stats.rankdata(ability, method='average')
            value_ranks = stats.rankdata(values, method='average')
            correlation = np.corrcoef(ability_ranks, value_ranks)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_kendall_correlation(self, ability, values):
        """计算肯德尔等级相关系数 - 使用scipy函数"""
        try:
            correlation, p_value = stats.kendalltau(ability, values, nan_policy='omit')
            return correlation if not np.isnan(correlation) else 0.0
        except:
            # 手动计算（简略版）
            n = len(ability)
            if n <= 1:
                return 0.0
            
            concordant = 0
            discordant = 0
            
            for i in range(n):
                for j in range(i+1, n):
                    ability_compare = ability[i] - ability[j]
                    value_compare = values[i] - values[j]
                    
                    if ability_compare * value_compare > 0:
                        concordant += 1
                    elif ability_compare * value_compare < 0:
                        discordant += 1
            
            total = concordant + discordant
            if total == 0:
                return 0
            
            correlation = (concordant - discordant) / total
            return correlation
    
    def calculate_all_correlations(self, ability, values):
        """计算所有类型的相关系数"""
        correlations = {}
        
        # 皮尔逊相关系数
        if len(ability) > 1:
            corr_matrix = np.corrcoef(ability, values)
            correlations['pearson'] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            correlations['pearson'] = 0.0
        
        # 斯皮尔曼等级相关系数
        correlations['spearman'] = self.calculate_spearman_correlation(ability, values)
        
        # 肯德尔等级相关系数
        correlations['kendall'] = self.calculate_kendall_correlation(ability, values)
        
        return correlations
    
    def _death_and_inheritance(self, indices):
        """死亡-继承机制：重新分配能力值，指定个体死亡"""
        if len(indices) == 0:
            return 0
            
        death_count = len(indices)
        self.total_deaths += death_count
        
        # 记录哪些血统死亡了
        dead_generations = self.generations[indices]
        
        if self.inheritance_type == "direct":
            # 直接继承：保持势力值不变，重新生成能力值
            self.ability[indices] = self._generate_ability(death_count)
            
        elif self.inheritance_type == "redistribute":
            # 部分重新分配：添加一些随机性，模拟遗产税或重新分配
            # 保留70%的势力值 (表示个人财富，地位，声望等的一个抽象的值)，30%重新分配
            total_wealth = np.sum(self.values[indices])
            redistribution = total_wealth * 0.3
            
            # 给每个死亡个体添加随机的小额重新分配
            random_redistribution = np.random.dirichlet(np.ones(death_count)) * redistribution
            self.values[indices] = self.values[indices] * 0.7 + random_redistribution
            
            # 重新生成能力值
            self.ability[indices] = self._generate_ability(death_count)
            
        elif self.inheritance_type == "meritocratic":
            # 精英继承：高能力后代继承更多
            # 根据当前能力值（作为遗传因素）决定继承比例
            genetic_factor = (self.ability[indices] - 50) / 100  # 标准化到0-1
            
            # 使用遗传因子作为权重重新分配
            weights = genetic_factor + 0.1  # 确保权重为正
            weights = weights / np.sum(weights)  # 归一化
            
            # 重新分配财富
            total_wealth = np.sum(self.values[indices])
            self.values[indices] = weights * total_wealth
            
            # 重新生成能力值（但保持一定的遗传性）
            # 新能力 = 遗传因子 * 父母能力 + (1-遗传因子) * 随机能力
            parent_factor = 0.3  # 30%受父母能力影响
            new_random_ability = self._generate_ability(death_count)
            self.ability[indices] = parent_factor * self.ability[indices] + (1-parent_factor) * new_random_ability
        
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
        death_count = 0
        
        if self.death_enabled:
            # 增加所有个体的年龄
            self.ages += 1
            
            # 检查哪些个体死亡
            death_indices = np.where(self.ages >= self.death_times)[0]
            
            # 执行死亡-继承机制
            if len(death_indices) > 0:
                death_count = self._death_and_inheritance(death_indices)
        else:
            # 永生社会：年龄增加，但不会死亡
            self.ages += 1
        
        # 运气因素
        luck_multiplier = np.where(
            np.random.random(self.num_individuals) > 0.5,
            1.1,  # 好运
            0.9   # 坏运
        )
        
        # 能力因素
        ability_factor = 0.9 + (self.ability - 50) * 0.2 / 100
        
        # 计算增长因子（考虑边际效益递减）
        growth_factor = self._calculate_growth_factor(ability_factor, luck_multiplier, self.values)
        
        # 更新势力值 (表示个人财富，地位，声望等的一个抽象的值)
        old_values = self.values.copy()
        self.values = self.values * growth_factor
        
        # 记录增长率
        with np.errstate(divide='ignore', invalid='ignore'):
            growth_rates = np.where(old_values > 0, (self.values - old_values) / old_values, 0)
        avg_growth_rate = np.mean(growth_rates)
        self.history['growth_rates'].append(avg_growth_rate)
        
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
        
        # 记录年龄分布（每40次迭代记录一次，原来是20次）
        if iteration % 40 == 0:
            # 记录年龄分布的直方图数据
            age_counts, age_bins = np.histogram(self.ages, bins=10, range=(0, self.num_iterations))
            self.history['age_distribution'].append((age_counts, age_bins))
        
        # 记录能力统计（每40次迭代记录一次，原来是20次）
        if iteration % 40 == 0:
            self.history['ability_means'].append(np.mean(self.ability))
            self.history['ability_stds'].append(np.std(self.ability))
        
        # 基尼系数
        gini = self._calculate_gini(current_values)
        self.history['gini_coefficients'].append(gini)
        
        # 计算所有相关系数
        correlations = self.calculate_all_correlations(self.ability, current_values)
        
        self.history['pearson_correlations'].append(correlations['pearson'])
        self.history['spearman_correlations'].append(correlations['spearman'])
        self.history['kendall_correlations'].append(correlations['kendall'])
        
        # 每40次迭代汇报一次相关系数（原来是20次）
        if (iteration + 1) % 40 == 0:
            self._report_correlations(iteration + 1, correlations)
    
    def _report_correlations(self, iteration, correlations):
        """汇报相关系数"""
        avg_age = np.mean(self.ages)
        avg_generation = np.mean(self.generations)
        avg_growth = self.history['growth_rates'][-1] if self.history['growth_rates'] else 0
        
        if self.death_enabled:
            print(f"迭代 {iteration:3d} - "
                  f"年龄: {avg_age:.1f}, 代数: {avg_generation:.1f} | "
                  f"平均增长率: {avg_growth:.4f} | "
                  f"皮尔逊={correlations['pearson']:.4f}, "
                  f"斯皮尔曼={correlations['spearman']:.4f}, "
                  f"肯德尔={correlations['kendall']:.4f}")
        else:
            print(f"迭代 {iteration:3d} - "
                  f"年龄: {avg_age:.1f} | "
                  f"平均增长率: {avg_growth:.4f} | "
                  f"皮尔逊={correlations['pearson']:.4f}, "
                  f"斯皮尔曼={correlations['spearman']:.4f}, "
                  f"肯德尔={correlations['kendall']:.4f}")
    
    def run_simulation(self, verbose=True):
        """运行完整模拟"""
        if verbose:
            print(f"开始模拟: {self.num_individuals}个个体, {self.num_iterations}次迭代")
            if self.death_enabled:
                print(f"死亡机制: 个体随机死亡，平均寿命={self.average_death_age}次迭代")
                print(f"继承类型: {self.inheritance_type}")
            else:
                print(f"死亡机制: 永生社会 (无死亡)")
            print(f"增长类型: {self.growth_type}")
            print(f"饱和点: {self.saturation_point}")
            print("相关系数说明:")
            print("  皮尔逊: 线性相关 [-1,1]")
            print("  斯皮尔曼: 排序相关 [-1,1]")
            print("  肯德尔: 排序和谐性 [-1,1]")
            print("-" * 80)
        
        for i in range(self.num_iterations):
            self.simulate_iteration(i)
            
            if verbose and (i + 1) % 40 == 0:  # 改为每40次迭代汇报一次
                avg_val = self.history['avg_values'][-1]
                median_val = self.history['median_values'][-1]
                gini_val = self.history['gini_coefficients'][-1]
                death_count = self.history['death_counts'][-1]
                growth_rate = self.history['growth_rates'][-1] if i > 0 else 0
                
                if self.death_enabled:
                    print(f"迭代 {i+1:3d}/{self.num_iterations} | "
                          f"平均势力: {avg_val:8.2f} | "
                          f"中位数: {median_val:8.2f} | "
                          f"基尼系数: {gini_val:.4f} | "
                          f"增长率: {growth_rate:.4f} | "
                          f"死亡数: {death_count}")
                else:
                    print(f"迭代 {i+1:3d}/{self.num_iterations} | "
                          f"平均势力: {avg_val:8.2f} | "
                          f"中位数: {median_val:8.2f} | "
                          f"基尼系数: {gini_val:.4f} | "
                          f"增长率: {growth_rate:.4f}")
        
        # 计算最终统计
        self._calculate_final_stats()
        
        if verbose:
            self.print_summary()
    
    def _calculate_final_stats(self):
        """计算最终统计信息"""
        final_values = self.values
        
        # 计算最终相关系数
        final_correlations = self.calculate_all_correlations(self.ability, final_values)
        
        # 计算十分位数
        deciles = np.percentile(final_values, range(0, 101, 10))
        
        # 计算排名数据
        ability_ranks = stats.rankdata(self.ability, method='average')
        value_ranks = stats.rankdata(final_values, method='average')
        rank_diff = ability_ranks - value_ranks
        
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
                    'mean_ability': np.mean(self.ability[indices])
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
        
        # 计算平均增长率
        growth_rates = np.array(self.history['growth_rates'])
        avg_growth_rate = np.mean(growth_rates) if len(growth_rates) > 0 else 0
        
        # 计算饱和程度（有多少个体超过饱和点）
        saturation_ratio = np.sum(final_values > self.saturation_point) / len(final_values)
        
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
            'death_enabled': self.death_enabled,
            'total_deaths': self.total_deaths,
            'total_births': self.total_births,
            'avg_death_per_iteration': avg_death_per_iteration,
            'avg_final_age': np.mean(self.ages),
            'avg_final_generation': np.mean(self.generations),
            
            # 代际统计
            'num_generations': len(generation_stats),
            'intergenerational_mobility': intergenerational_mobility,
            'generation_stats': generation_stats,
            
            # 增长统计
            'avg_growth_rate': avg_growth_rate,
            'saturation_ratio': saturation_ratio,
            'growth_type': self.growth_type,
            'saturation_point': self.saturation_point
        }
    
    def print_summary(self):
        """打印模拟总结"""
        print("\n" + "="*80)
        print("模拟结果总结")
        print("="*80)
        
        stats = self.final_stats
        
        print(f"\n整体统计:")
        print(f"  平均势力值 (表示个人财富，地位，声望等的一个抽象的值): {stats['mean_value']:.2f}")
        print(f"  中位数势力值 (表示个人财富，地位，声望等的一个抽象的值): {stats['median_value']:.2f}")
        print(f"  标准差: {stats['std_value']:.2f}")
        print(f"  范围: {stats['min_value']:.2f} - {stats['max_value']:.2f}")
        print(f"  基尼系数: {stats['gini']:.4f}")
        
        print(f"\n增长统计:")
        print(f"  增长类型: {stats['growth_type']}")
        print(f"  平均增长率: {stats['avg_growth_rate']:.4f}")
        print(f"  饱和点: {stats['saturation_point']}")
        print(f"  超过饱和点的个体比例: {stats['saturation_ratio']:.2%}")
        
        print(f"\n分位数:")
        print(f"  10%分位数: {stats['percentile_10']:.2f}")
        print(f"  90%分位数: {stats['percentile_90']:.2f}")
        if stats['median_value'] > 0:
            print(f"  平均值/中位数比率: {stats['mean_value']/stats['median_value']:.2f}")
        else:
            print(f"  平均值/中位数比率: 无穷大")
        
        if stats['death_enabled']:
            print(f"\n人口动态:")
            print(f"  总死亡数: {stats['total_deaths']}")
            print(f"  总新生数: {stats['total_births']}")
            print(f"  平均每轮死亡数: {stats['avg_death_per_iteration']:.2f}")
            print(f"  最终平均年龄: {stats['avg_final_age']:.1f}")
            print(f"  最终平均代数: {stats['avg_final_generation']:.1f}")
            print(f"  总代数: {stats['num_generations']}")
            
            if stats['num_generations'] > 1:
                print(f"\n代际流动性:")
                print(f"  代际财富变化率: {stats['intergenerational_mobility']:.4f}")
                if stats['intergenerational_mobility'] > 0:
                    print(f"  趋势: 后代比前代更富有")
                else:
                    print(f"  趋势: 后代比前代更贫穷")
        else:
            print(f"\n人口动态:")
            print(f"  永生社会，无死亡事件")
            print(f"  最终平均年龄: {stats['avg_final_age']:.1f}")
        
        print(f"\n最终相关系数 (能力 vs 势力):")
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
        
        if self.death_enabled:
            fig.suptitle(f'世界模型模拟结果 - 图表组1 (增长类型: {self.growth_type}, 寿命: {self.average_death_age})', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'世界模型模拟结果 - 图表组1 (增长类型: {self.growth_type}, 永生社会)', 
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
        
        # 添加饱和线
        ax1.axhline(y=self.saturation_point, color='orange', linestyle='--', 
                   alpha=0.7, label=f'饱和点({self.saturation_point})')
        
        # 添加死亡阴影（如果有死亡）
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
        ax1.set_ylabel('势力值 (表示个人财富，地位，声望等的一个抽象的值)', fontsize=10)
        if self.death_enabled:
            ax1.set_title('1. 势力值随时间变化 (红色阴影: 死亡事件强度)', fontsize=12, fontweight='bold')
        else:
            ax1.set_title('1. 势力值随时间变化', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)
        # 调整x轴刻度，因为400次迭代需要更多的刻度
        ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
        
        # 图表2：增长率变化
        ax2 = plt.subplot(2, 2, 2)
        
        ax2.plot(iterations, self.history['growth_rates'], 
                'g-', linewidth=1.5, label='平均增长率', alpha=0.8)
        
        # 添加死亡阴影（如果有死亡）
        if np.max(death_counts) > 0:
            ax2.fill_between(iterations, 
                           ax2.get_ylim()[0], 
                           ax2.get_ylim()[1],
                           where=(death_counts > 0),
                           alpha=0.1, color='red', 
                           label='死亡事件', edgecolor='none')
        
        ax2.set_xlabel('迭代次数', fontsize=10)
        ax2.set_ylabel('平均增长率', fontsize=10)
        ax2.set_title('2. 平均增长率变化', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(8))
        
        # 图表3：能力与最终势力值散点图，按年龄着色
        ax3 = plt.subplot(2, 2, 3)
        
        # 计算正确的排名
        ability_ranks = stats.rankdata(self.ability, method='average')
        value_ranks = stats.rankdata(self.values, method='average')
        
        # 用年龄着色
        scatter = ax3.scatter(self.ability, self.values, 
                             c=self.ages, cmap='viridis', 
                             alpha=0.5, s=5, edgecolors='none')
        
        ax3.set_xlabel('能力值', fontsize=10)
        ax3.set_ylabel('最终势力值 (表示个人财富，地位，声望等的一个抽象的值)', fontsize=10)
        ax3.set_title(f'3. 能力 vs 势力 (按年龄着色)', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax3, label='年龄', shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        
        # 图表4：基尼系数变化
        ax4 = plt.subplot(2, 2, 4)
        
        ax4.plot(iterations, self.history['gini_coefficients'], 
                'r-', linewidth=2, alpha=0.8)
        
        # 添加死亡阴影（如果有死亡）
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
        ax4.xaxis.set_major_locator(plt.MaxNLocator(8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_chart_set_2(self):
        """绘制第二组图表（4个子图）"""
        fig = plt.figure(figsize=(16, 12))
        
        if self.death_enabled:
            fig.suptitle(f'世界模型模拟结果 - 图表组2 (增长类型: {self.growth_type}, 寿命: {self.average_death_age})', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'世界模型模拟结果 - 图表组2 (增长类型: {self.growth_type}, 永生社会)', 
                        fontsize=16, fontweight='bold')
        
        iterations = range(1, self.num_iterations + 1)
        
        # 图表5：三种相关系数对比
        ax1 = plt.subplot(2, 2, 1)
        
        ax1.plot(iterations, self.history['pearson_correlations'], 
                'b-', linewidth=1.5, label='皮尔逊', alpha=0.8)
        ax1.plot(iterations, self.history['spearman_correlations'], 
                'g-', linewidth=1.5, label='斯皮尔曼', alpha=0.8)
        ax1.plot(iterations, self.history['kendall_correlations'], 
                'orange', linewidth=1.5, label='肯德尔', alpha=0.8)
        
        # 添加死亡阴影（如果有死亡）
        death_counts = np.array(self.history['death_counts'])
        if np.max(death_counts) > 0:
            ax1.fill_between(iterations, 
                           ax1.get_ylim()[0], 
                           ax1.get_ylim()[1],
                           where=(death_counts > 0),
                           alpha=0.1, color='red', 
                           label='死亡事件', edgecolor='none')
        
        ax1.set_xlabel('迭代次数', fontsize=10)
        ax1.set_ylabel('相关系数', fontsize=10)
        ax1.set_title('5. 三种相关系数对比 (能力 vs 势力)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
        
        # 图表6：人口动态（如果有死亡）
        ax2 = plt.subplot(2, 2, 2)
        
        if self.death_enabled:
            # 绘制死亡和新生数量
            death_counts = np.array(self.history['death_counts'])
            birth_counts = np.array(self.history['birth_counts'])
            
            ax2.plot(iterations, death_counts, 'r-', linewidth=1.5, label='死亡数', alpha=0.8)
            ax2.plot(iterations, birth_counts, 'g-', linewidth=1.5, label='新生数', alpha=0.8)
            
            # 绘制移动平均线以显示趋势
            window_size = min(40, len(death_counts)//4)  # 窗口大小调整为40
            if window_size > 1:
                death_moving_avg = np.convolve(death_counts, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(iterations[window_size-1:], death_moving_avg, 'r--', linewidth=2, label='死亡移动平均', alpha=0.7)
            
            ax2.set_xlabel('迭代次数', fontsize=10)
            ax2.set_ylabel('个体数量', fontsize=10)
            ax2.set_title('6. 人口动态 (死亡与新生)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right', fontsize=9)
        else:
            # 永生社会没有死亡事件
            ax2.text(0.5, 0.5, '永生社会\n无死亡事件', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('6. 人口动态 (永生社会)', fontsize=12, fontweight='bold')
        
        ax2.xaxis.set_major_locator(plt.MaxNLocator(8))
        
        # 图表7：最终势力值分布(对数尺度)
        ax3 = plt.subplot(2, 2, 3)
        
        # 对势力值取对数
        log_values = np.log10(self.values + 1)
        
        if self.death_enabled and self.final_stats['num_generations'] > 1:
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
            
            ax3.set_xlabel('势力值(对数尺度) (表示个人财富，地位，声望等的一个抽象的值)', fontsize=10)
            ax3.set_ylabel('概率密度', fontsize=10)
            ax3.set_title('7. 各代势力值分布(对数尺度)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            if len(unique_generations) <= 5:
                ax3.legend(loc='upper right', fontsize=9)
        else:
            # 永生社会或只有一代
            ax3.hist(log_values, bins=50, color='steelblue', 
                    edgecolor='black', alpha=0.7, density=True)
            
            # 计算对数平均值和中位数
            mean_log = np.mean(log_values)
            median_log = np.median(log_values)
            
            ax3.axvline(mean_log, color='red', linestyle='--', 
                       label=f'均值: {10**mean_log:.1f}(取对数后{mean_log:.2f})')
            ax3.axvline(median_log, color='orange', linestyle='--',
                       label=f'中位数: {10**median_log:.1f}(取对数后{median_log:.2f})')
            
            ax3.set_xlabel('势力值(对数尺度) (表示个人财富，地位，声望等的一个抽象的值)', fontsize=10)
            ax3.set_ylabel('概率密度', fontsize=10)
            ax3.set_title('7. 最终势力值分布(对数尺度)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.legend(loc='upper right', fontsize=9)
        
        # 图表8：年龄和代数分布
        ax4 = plt.subplot(2, 2, 4)
        
        if self.death_enabled:
            # 绘制年龄和代数
            ax4_ages = ax4.twinx()  # 创建第二个y轴
            
            ax4.plot(iterations, self.history['avg_age'], 'b-', linewidth=2, label='平均年龄', alpha=0.8)
            ax4_ages.plot(iterations, self.history['avg_generation'], 'orange', linewidth=2, label='平均代数', alpha=0.8)
            
            ax4.set_xlabel('迭代次数', fontsize=10)
            ax4.set_ylabel('平均年龄', fontsize=10, color='blue')
            ax4_ages.set_ylabel('平均代数', fontsize=10, color='orange')
            ax4.set_title('8. 人口年龄和代数变化', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_ages.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        else:
            # 永生社会只有年龄增加
            ax4.plot(iterations, self.history['avg_age'], 'b-', linewidth=2, label='平均年龄', alpha=0.8)
            ax4.set_xlabel('迭代次数', fontsize=10)
            ax4.set_ylabel('平均年龄', fontsize=10)
            ax4.set_title('8. 人口年龄变化 (永生社会)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper left', fontsize=9)
        
        ax4.xaxis.set_major_locator(plt.MaxNLocator(8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_growth_comparison(self):
        """绘制不同增长模型的比较图"""
        fig = plt.figure(figsize=(12, 8))
        
        fig.suptitle(f'不同势力值下的增长因子比较 (增长类型: {self.growth_type})', 
                    fontsize=16, fontweight='bold')
        
        # 生成测试数据：不同的势力值
        test_values = np.logspace(0, 5, 100)  # 从1到100,000
        ability_factor = 1.0  # 假设能力因子为1
        luck_multiplier = 1.0  # 假设运气因子为1
        
        # 计算不同势力值下的增长因子
        growth_factors = []
        for value in test_values:
            # 创建一个假的能力因子数组和运气因子数组
            ability_factors = np.full(1, ability_factor)
            luck_multipliers = np.full(1, luck_multiplier)
            values_array = np.full(1, value)
            
            # 计算增长因子
            growth_factor = self._calculate_growth_factor(ability_factors[0], luck_multipliers[0], values_array)[0]
            growth_factors.append(growth_factor)
        
        ax = plt.subplot(1, 1, 1)
        ax.plot(test_values, growth_factors, 'b-', linewidth=2, alpha=0.8)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='无增长线(y=1)')
        ax.axvline(x=self.saturation_point, color='orange', linestyle='--', 
                  alpha=0.7, label=f'饱和点({self.saturation_point})')
        
        ax.set_xlabel('当前势力值', fontsize=12)
        ax.set_ylabel('增长因子', fontsize=12)
        ax.set_title(f'增长因子随势力值的变化 ({self.growth_type}模型)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # 使用对数尺度
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加注解
        if self.growth_type == "exponential":
            ax.text(0.05, 0.95, '指数增长模型\n(无边际递减)', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif self.growth_type == "bounded":
            ax.text(0.05, 0.95, '边际递减模型\n(值越大增长越难)', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif self.growth_type == "logistic":
            ax.text(0.05, 0.95, '逻辑斯蒂模型\n(S型增长)', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif self.growth_type == "symmetric":
            ax.text(0.05, 0.95, '对称增长模型\n(中等水平最易增长)', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def print_chart_explanations(self):
        """打印图表解释"""
        print("\n" + "="*80)
        print("图表详细解释")
        print("="*80)
        
        if self.death_enabled:
            print(f"\n死亡机制设置:")
            print(f"  平均寿命: {self.average_death_age}次迭代")
            print(f"  继承类型: {self.inheritance_type}")
            print(f"  总死亡数: {self.total_deaths}")
            print(f"  总新生数: {self.total_births}")
        else:
            print(f"\n社会类型: 永生社会 (无死亡机制)")
        
        print(f"\n增长机制设置:")
        print(f"  增长类型: {self.growth_type}")
        print(f"  饱和点: {self.saturation_point}")
        print(f"  困难因子: {self.difficulty_factor}")
        
        print("\n图表组1:")
        print("1. 势力值随时间变化:")
        print("   - 蓝线: 平均势力值 (表示个人财富，地位，声望等的一个抽象的值)")
        print("   - 红线: 中位数势力值 (表示个人财富，地位，声望等的一个抽象的值)")
        print("   - 蓝区: 10%-90%分位数范围")
        print("   - 橙线: 饱和点 (边际递减开始显著的值)")
        if self.death_enabled:
            print("   - 红色阴影: 死亡事件发生的强度和频率")
        
        print("\n2. 平均增长率变化:")
        print("   - 绿线: 平均增长率随时间变化")
        print("   - 反映了边际递减效应: 随着势力值增大，增长率趋于0")
        if self.death_enabled:
            print("   - 红色阴影: 死亡事件发生的强度和频率")
        
        print("\n3. 能力 vs 势力:")
        print("   - 散点图: 每个点代表一个个体")
        print("   - 颜色: 年龄(深色年轻，浅色年老)")
        print("   - 对角线趋势: 能力越高，势力越高")
        
        print("\n4. 不平等程度变化(基尼系数):")
        print("   - 红线: 基尼系数随时间变化")
        if self.death_enabled:
            print("   - 红色阴影: 死亡事件发生的强度和频率")
        print("   - 橙线: 警戒线(0.3)")
        print("   - 红线: 危险线(0.4)")
        
        print("\n图表组2:")
        print("5. 三种相关系数对比 (能力 vs 势力):")
        print("   - 蓝线: 皮尔逊相关系数(线性相关)")
        print("   - 绿线: 斯皮尔曼相关系数(排序相关)")
        print("   - 橙线: 肯德尔相关系数(排序和谐性)")
        if self.death_enabled:
            print("   - 红色阴影: 死亡事件发生的强度和频率")
        
        if self.death_enabled:
            print("\n6. 人口动态 (死亡与新生):")
            print("   - 红线: 每轮死亡个体数量")
            print("   - 绿线: 每轮新生个体数量")
            print("   - 红色虚线: 死亡数量的移动平均线")
        else:
            print("\n6. 人口动态:")
            print("   - 永生社会没有死亡事件")
        
        if self.death_enabled and self.final_stats['num_generations'] > 1:
            print("\n7. 各代势力值分布(对数尺度):")
            print("   - 直方图: 按不同代数分组")
            print("   - 不同颜色代表不同代数")
            print("   - 显示代际财富分布差异")
        else:
            print("\n7. 最终势力值分布(对数尺度):")
            print("   - 使用对数尺度显示分布")
            print("   - 大多数个体势力值很低(左侧)")
            print("   - 少数个体势力值极高(右侧长尾)")
        
        print("\n8. 人口年龄和代数变化:")
        print("   - 蓝线: 人口平均年龄")
        if self.death_enabled:
            print("   - 橙线: 人口平均代数")
            print("   - 年龄增长反映个体存活时间")
            print("   - 代数增长反映世代更替")
        else:
            print("   - 永生社会年龄持续增长")
        
        print(f"\n增长模型解释 ({self.growth_type}):")
        if self.growth_type == "exponential":
            print("  指数增长: 势力值呈指数增长，模拟科技无限发展的世界")
        elif self.growth_type == "bounded":
            print("  边际递减: 势力值越大增长越困难，模拟现实世界的瓶颈效应")
            print("  小势力值: 容易增长，容易脱贫")
            print("  大势力值: 增长困难，接近饱和")
        elif self.growth_type == "logistic":
            print("  逻辑斯蒂增长: S型曲线，模拟资源有限环境下的增长")
            print("  初期: 快速增长")
            print("  中期: 增长减缓")
            print("  后期: 接近饱和，几乎无增长")
        elif self.growth_type == "symmetric":
            print("  对称增长: 中等水平最容易增长，特别小和特别大都难增长")
            print("  模拟: 中等势力最容易发展，赤贫和巨富都面临增长困难")
            print("  赤贫: 缺乏初始资源，难以起步")
            print("  巨富: 体量太大，边际效益递减")
            print("  中等: 既有资源又有增长空间")


def compare_growth_types():
    """比较不同增长类型的效果"""
    print("="*80)
    print("不同增长类型比较")
    print("="*80)
    
    growth_types = ["exponential", "bounded", "logistic", "symmetric"]
    results = {}
    
    for growth_type in growth_types:
        print(f"\n增长类型: {growth_type}")
        
        model = WorldModel(
            num_individuals=5000,
            initial_value=100,
            num_iterations=200,  # 比较场景中使用200次迭代以加速
            random_seed=42,
            average_death_age=40,
            inheritance_type="direct",
            growth_type=growth_type,
            saturation_point=1000,
            difficulty_factor=0.5
        )
        model.run_simulation(verbose=False)
        results[growth_type] = model.final_stats
        
        # 简要输出
        stats = model.final_stats
        print(f"  平均势力值: {stats['mean_value']:.2f}")
        print(f"  最大势力值: {stats['max_value']:.2f}")
        print(f"  平均增长率: {stats['avg_growth_rate']:.4f}")
        print(f"  基尼系数: {stats['gini']:.4f}")
        print(f"  斯皮尔曼系数: {stats['spearman_correlation']:.4f}")
    
    # 总结比较
    print("\n" + "="*80)
    print("增长类型比较总结")
    print("="*80)
    
    for growth_type, stats in results.items():
        print(f"\n{growth_type}:")
        print(f"  最终平均势力: {stats['mean_value']:.2f}")
        print(f"  最终最大势力: {stats['max_value']:.2f}")
        print(f"  平均增长率: {stats['avg_growth_rate']:.4f}")
        print(f"  不平等程度: {stats['gini']:.4f} (基尼系数)")
        print(f"  能力-势力相关性: {stats['spearman_correlation']:.4f} (斯皮尔曼)")
    
    return results


def compare_lifespan_scenarios():
    """比较不同平均寿命场景的效果"""
    print("="*80)
    print("不同平均寿命场景比较")
    print("="*80)
    
    lifespan_scenarios = [20, 40, 80, 0]  # 0表示永生社会
    results = {}
    
    for lifespan in lifespan_scenarios:
        if lifespan == 0:
            print(f"\n场景: 永生社会")
        else:
            print(f"\n场景: 平均寿命 {lifespan} 次迭代")
        
        model = WorldModel(
            num_individuals=5000,
            initial_value=100,
            num_iterations=200,  # 比较场景中使用200次迭代以加速
            random_seed=42,
            average_death_age=lifespan,
            inheritance_type="direct",
            growth_type="bounded",
            saturation_point=1000
        )
        model.run_simulation(verbose=False)
        results[lifespan] = model.final_stats
        
        # 简要输出
        stats = model.final_stats
        print(f"  基尼系数: {stats['gini']:.4f}")
        print(f"  斯皮尔曼系数: {stats['spearman_correlation']:.4f}")
        if lifespan > 0:
            print(f"  总代数: {stats['num_generations']}")
            print(f"  代际流动性: {stats['intergenerational_mobility']:.4f}")
    
    # 总结比较
    print("\n" + "="*80)
    print("平均寿命场景比较总结")
    print("="*80)
    
    for lifespan, stats in results.items():
        if lifespan == 0:
            scenario_name = "永生社会"
        else:
            scenario_name = f"寿命{lifespan}"
        
        print(f"\n{scenario_name}:")
        print(f"  不平等程度: {stats['gini']:.4f} (基尼系数)")
        print(f"  能力-势力相关性: {stats['spearman_correlation']:.4f} (斯皮尔曼)")
        if lifespan > 0:
            print(f"  社会代际数: {stats['num_generations']}")
            print(f"  代际财富变化率: {stats['intergenerational_mobility']:.4f}")
    
    return results


def main():
    """主函数"""
    print("世界模型模拟器 - 边际效益递减机制")
    print("="*80)
    
    print("\n选择运行模式:")
    print("1. 单次模拟 (边际递减增长)")
    print("2. 比较不同增长类型")
    print("3. 比较不同平均寿命场景")
    print("4. 比较不同继承类型")
    print("5. 永生社会 (基准对照)")
    
    choice = input("\n请输入选择 (1-5): ")
    
    if choice == "1":
        # 单次模拟
        print("\n单次模拟 - 边际递减增长")
        
        # 获取用户输入参数
        avg_lifespan = int(input("请输入平均寿命 (默认40): ") or "40")
        inheritance_type = input("请输入继承类型 (direct/redistribute/meritocratic, 默认direct): ") or "direct"
        growth_type = input("请输入增长类型 (exponential/bounded/logistic/symmetric, 默认bounded): ") or "bounded"
        saturation_point = float(input("请输入饱和点 (默认1000): ") or "1000")
        difficulty_factor = float(input("请输入困难因子 (0.1-1.0, 默认0.5): ") or "0.5")
        
        model = WorldModel(
            num_individuals=100000,
            initial_value=100,
            num_iterations=400,
            random_seed=42,
            average_death_age=avg_lifespan,
            inheritance_type=inheritance_type,
            growth_type=growth_type,
            saturation_point=saturation_point,
            difficulty_factor=difficulty_factor
        )
        
        print("开始模拟...")
        model.run_simulation(verbose=True)
        
        # 生成第一组图表
        print("\n生成第一组图表...")
        model.plot_chart_set_1()
        
        # 生成第二组图表
        print("\n生成第二组图表...")
        model.plot_chart_set_2()
        
        # 生成增长模型比较图
        print("\n生成增长模型比较图...")
        model.plot_growth_comparison()
        
        # 打印图表解释
        model.print_chart_explanations()
        
    elif choice == "2":
        # 比较不同增长类型
        results = compare_growth_types()
        
    elif choice == "3":
        # 比较不同平均寿命场景
        results = compare_lifespan_scenarios()
        
    elif choice == "4":
        # 比较不同继承类型
        print("\n比较不同继承类型 (平均寿命40, 边际递减增长)")
        
        inheritance_types = ["direct", "redistribute", "meritocratic"]
        for inheritance_type in inheritance_types:
            print(f"\n继承类型: {inheritance_type}")
            model = WorldModel(
                num_individuals=5000,
                initial_value=100,
                num_iterations=200,  # 比较场景中使用200次迭代以加速
                random_seed=42,
                average_death_age=40,
                inheritance_type=inheritance_type,
                growth_type="bounded",
                saturation_point=1000
            )
            model.run_simulation(verbose=False)
            
            stats = model.final_stats
            print(f"  基尼系数: {stats['gini']:.4f}")
            print(f"  斯皮尔曼系数: {stats['spearman_correlation']:.4f}")
            if stats['death_enabled']:
                print(f"  代际流动性: {stats['intergenerational_mobility']:.4f}")
        
    elif choice == "5":
        # 永生社会 (基准对照)
        print("\n基准对照 - 永生社会 (无死亡)")
        
        growth_type = input("请输入增长类型 (exponential/bounded/logistic/symmetric, 默认bounded): ") or "bounded"
        
        model = WorldModel(
            num_individuals=10000,
            initial_value=100,
            num_iterations=400,
            random_seed=42,
            average_death_age=0,  # 永生社会
            inheritance_type="direct",
            growth_type=growth_type,
            saturation_point=1000
        )
        
        print("开始模拟...")
        model.run_simulation(verbose=True)
        
        # 生成图表
        print("\n生成图表...")
        model.plot_chart_set_1()
        model.plot_chart_set_2()
        model.plot_growth_comparison()
        
        # 打印图表解释
        model.print_chart_explanations()

if __name__ == "__main__":
    main()