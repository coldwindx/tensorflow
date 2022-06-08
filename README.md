# 聚类分析
## 离群点（Outlier）
- $distance_k(O)$表示点O到k近邻点的距离
- 可达距离：$distance_k(A,B) = max\{distance_k(B), d(A,B)\}$
- 局部可达密度，k最近邻点的平均可达密度的倒数：
$lrd(A) = 1/(\frac{\sum_{B\in{N_k(A)}}distance_k(A,B)}{|N_k(A)|})$
- 局部离群点因子，代表离群点程度：$LOF(A)=\frac{\sum_{B\in{N_k(A)}}{\frac{lrd(B)}{lrd(A)}}}{|N_k(A)|}=\frac{\sum_{B\in{N_k(A)}}{lrd(B)}}{|N_k(A)|}/lrd(A)$

# 准确率
- $G-mean = \sqrt{Acc^+ * Acc^-}$
- $F-measure = \frac{2 * precision * recall}{precision + recall}$

# 标准化
- $v^1=\frac{v-min}{max-min}(new\_max-new\_min)+new\_min$ 
- Z-score normalization: $v^1=\frac{v-\mu}{\sigma}$

# 熵(Entropy)
- $H(X)=-\sum_{i=1}^n p(x_i)\log_2{p(x_i)}$
