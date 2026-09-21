# 顶盖驱动方腔流：Re=100

目的：展示经典主涡/角涡结构，检查速度中心线的网格敏感性与终态残差。单位方腔，上壁切向速度 U=1，另外三壁静止，所有壁面法向速度为零；ν=0.01，Re=UL/ν=100。由静止启动，采用 SDIRK2-mr-ccSAV，dt=0.02，T=40。配置可编辑，Re由实际参数导出。

移动壁仍满足相对于壁面的无滑移。顶角速度不连续是标准方腔算例的特征；面自由度不包含顶角的切向速度。以 u_ghost=2U-u_inside 消去顶部虚拟点，只需在最上排自由 u 面添加 2νU/hy²。原 K 不变。当前守恒/反对称对流的墙面通量乘以法向速度零，故顶盖的切向虚拟值不改变这项通量。它是边界黏性载荷，不是物理体力。

这项扩展独立于原齐次壁面的理论证明。已有 h1_seminorm_squared 是齐次矩阵二次型；在移动壁算例中不能直接解释为带边界贡献的物理耗散。

```sh
uv run python experiments/cavity/run.py --config experiments/cavity/configs/grid_check.json
uv run python experiments/cavity/analyze.py --runs runs/cavity/<32网格运行编号> runs/cavity/<64网格运行编号>
```

计算和绘图分离。图包含速度大小/流函数等值线、内点涡量，以及两条中心线的网格比较。主涡位置从顶点流函数极小值估计。稳态指标是投影后的连续半离散 NS 右端 L2 范数；没有仅凭 T 或小散度宣称稳态。两网格差是敏感性检查，不是完整网格收敛证明。原始结果、边界配置和来源随运行记录保存。

经典文献：Ghia, Ghia & Shin (1982), High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method. https://doi.org/10.1016/0021-9991(82)90058-4 。本次不把两网格比较称为已通过 Ghia 数据验证。
