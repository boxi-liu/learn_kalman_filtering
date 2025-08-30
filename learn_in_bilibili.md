**一.入门**
卡尔曼滤波clang
适用系统：线性高斯系统
1. 线性
    1. 叠加性
   两个输入x1,x2作用于一个系统产生一个输出y,等价于两个输入x1,x2作用于两个系统产生y1,y2,y1和y2再作用于一个系统产生y
   即：
   ```
   y = ax1+bx2
   ```
    2. 齐次性
   输入x增大k倍，y也增大k倍
2. 高斯： 噪声满足正态分布
宏观意义：滤波即加权

理想状态： 信号*1 + 噪声 *0 

卡尔曼滤波： 修正值= 估计权重 + 观测权重

**二.进阶**
1. 状态空间表达式
   状态方程： $x_k=Ax_{k-1} + Bu_k +w_k$
   观测方程： $y_k=Cx_k + v_k$
2. 高斯分布
   1. 参数分析
      1. wk 过程噪声 符合正态分布：后验估计值均值为0 方差Qk
      2. vk 观测噪声 符合正态分布：均值为0 方差Rk
        wk vk统称为高斯白噪声
        >高斯白噪声的高斯指的是概率分布为正态后验估计值分布，白噪声指的是其二阶矩不相关一阶矩为常数。 故把瞬时值的概率分布服从高斯分布，功率谱密度服从均匀分布的噪声后验估计值称为高斯白噪声。 这两个条件是判断高斯白噪声性能的标准。 功率谱密度：每赫兹的瓦特数 高斯白噪声 高斯分布：噪声的幅度服从高斯（正态）分布，这意味着噪声的统计特性可以通过其均值和方差来完全描述。
         >
   2. 方差
   一维： 噪声：Qk ,Rk 状态：xt自身有个方差（？wk，Qk）
   二维： xt= $$\begin{bmatrix}
xt1（wk1）\\
xt2（wk2）\\
\end{bmatrix}$$
    协方差cov（xt1,xt2）=$$\begin{bmatrix}
cov(x1,x1)&cov(x1,x2)\\
cov(x2,x1)&cov(x2,x2)\\
\end{bmatrix}$$

   3. 超参数
   Q（Qk） R（RK） ～ PID 
   *需要自己调的参数*
   4. 卡尔曼直观图解
    ![error](./截图%202025-08-27%2018-54-01.png)
    xk-1^:修正值（后验估计值），上一时刻卡尔曼滤波输出的自由估计值
    xk_ ^:先验估计值，基于xk-1估计的当前估计值
    yk(xk)： 观测值
    xk^:当前的自由估计值
    *^表示估计*


**三.放弃**
1. 卡尔曼公式理解
*实现过程：使用上一次的最优结果预测当前值，同时使用观测值修正当前值，得到最优结果*
   1. 先验估计
   $\widehat x_t =F\widehat x_{t-1}+Bu_{i-1}$ 
   $$x_t= \begin{bmatrix}
p(位置)\\
v（速度）\\
\end{bmatrix}$$
   *假设为匀加速模型*
   $pi=p_{i-1}+v_{i-1}\Delta t+a/2\Delta t^2$
   $vi=v_{i-1}+a\Delta t$
   等价于
   $$ \begin{bmatrix}
Pi\\
vi\\
\end{bmatrix}=   
   \begin{bmatrix}
1&\Delta t\\
0&1\\
\end{bmatrix}  \begin{bmatrix} p_{i-1}\\
v_{i-1}\end{bmatrix}+   \begin{bmatrix}
\Delta t^2/2\\
\Delta t\\
\end{bmatrix} ai$$
   即
   $ \widehat x_t =F \widehat x_{t-1} +Bu_{i1} +w_t$
   *两式上下具有一一对应关系*
<br>
   1. 先验估计协方差
   $Pt^{-} =F*P_{t-1}*F^T+Q$
   $cov(\widehat x_t,\widehat x_t)=cov(F \widehat x_{t-1} +Bu_{i1}+w_t,F \widehat x_{t-1} +Bu_{i1}+w_t)$
   即：
   $F*cov(\widehat x_{t-1},\widehat x_{t-1})*F^T+cov(w_t,w_t)$
   <br>
   2. 测量方程
   测量 $z_p=p_t+\Delta p_t(观测器误差)$
   模型$z_v=0$
   等价于
   $\begin{bmatrix}
z_p\\
0\\
\end{bmatrix}=\begin{bmatrix}
1&0\\
\end{bmatrix} \begin{bmatrix}
p_t\\
v_t\\
\end{bmatrix}+\begin{bmatrix}
1&0\\
\end{bmatrix}\begin{bmatrix}
\Delta p_t\\
\Delta v_t\\
\end{bmatrix}$
   即：
   $z_t=Hx_t +v$
   <br>

   1. 修正估计
   $\widehat x_t=\widehat x_t ^{ -}+K_t(z_t-H\widehat x_t ^{ -})$
   $K_t为卡尔曼增益,\widehat x_t为卡尔曼滤波最终值，$
   $\widehat x_t ^{ -}为先验估计值，z_t 为观测值$
   <br>
   2. 更新卡尔曼增益
   $K_t=\frac{P_t^{ -}H^T}{HP_t^{ -}H^T+R}$
   一维时等于
   $K_t=\frac{P_t^{ -}}{P_t^{ -}+R}=\frac{P_{t-1}^{ -}+Q}{P_{t-1}^{ -}+Q+R}$
   即：$K_t$与Q，R都有关
   <br>
   3. 更新后验估计协方差
   $P_t=（I-K_tH）P_t^{ -}$
   I是单位矩阵
2. 调节超参数
   1. Q与R的取值
   公式层面理解
   $Pt^{-} =F*P_{t-1}*F^T+Q$
   $K_t=\frac{P_t^{ -}H^T}{HP_t^{ -}H^T+R}$
   （一维）化简：
   $K_t=\frac{P_t^{ -}}{P_t^{ -}+R}=\frac{P_{t-1}^{ -}+Q}{P_{t-1}^{ -}+Q+R}$
   结合：
   $\widehat x_t=\widehat x_t ^{ -}+K_t(z_t-H\widehat x_t ^{ -})$
   *当观测器精度高时（更信任观测值），调高K，即降低R，反之，当运动模型理想（没有摩擦），降低K，即降低Q或调高R*
   *Q是过程噪声方差，R是观测噪声方差*
   <br>
   2. $P_0与\widehat x _0的取值$
   $习惯取\widehat x_0=0,P往小的取，方便收敛（一般取1）（不可为0）$
3. 卡尔曼滤波的使用
   1. 选择状态量，观测量
   2. 构建方程
   3. 初始化参数
   4. 代入公式迭代
   ![error](./截图%202025-08-28%2009-54-55.png)
   5. 调节超参数
**精通**
1. 机器人应用举例
   1. 陀螺仪滤波
      1. 选择状态量，观测量
      状态量：$\begin{bmatrix}\
         angle\\
         \text{Q\_bias}(陀螺仪漂移)       
      \end{bmatrix}$
      观测量：
      $\begin{bmatrix}
         newAngle
      \end{bmatrix}$
      ![error](./截图%202025-08-30%2009-38-32.png)
      2. 构建方程
         1. 预测先验估计值
         $angle_i=angle_{i-1}-\text{Q\_ bias}dt+newGryro*dt$
         Q_bias=Q_bias
         ![error](./截图%202025-08-30%2009-48-48.png)
         2. 预测先验估计协方差
         ![error](./截图%202025-08-30%2009-55-41.png)
         其中A的值由先验估计值式子相同的参数继承验估计协方差而来
         ![error](./截图%202025-08-30%2010-03-35.png)
         ![error](./截图%202025-08-30%2010-05-38.png)
         3. 建立测量方程
         ![error](./截图%202025-08-30%2010-08-34.png)
         4. 计算卡尔曼增益
         ![error](./截图%202025-08-30%2010-10-01.png)
         5. 计算当前最优估计值
         ![error](./截图%202025-08-30%2010-12-26.png)
         6. 更新协方差矩阵
         ![error](./截图%202025-08-30%2010-13-24.png)
      （理论）代码总览：
      ![error](./截图%202025-08-30%2010-14-37.png)