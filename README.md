# Pref_Route

We aim to construct a general learning-based model that can generate different proper routes according to the preferences of users, such as shorter path, faster path, or general path.



code

- model.py: 路线生成模型原始代码
- train.py: 原模型训练代码
- valid.py: 验证代码
- pref_train.py: 考虑多偏好场景的路线生成模型训练代码
- pref_valid.py: 验证代码
- stat.py: 数据统计分析



Res（最后没添加非线性激活层）

| Model/Metric | 数据          | Loss                                      | Overlap | Length  |
| ------------ | ------------- | ----------------------------------------- | ------- | ------- |
| model        | 全偏好        | 交叉熵loss_ce                             | 0.904   | 4082.05 |
| pref_model   | 全长度        | loss_ce+10*loss_length                    | 0.606   | 3112.57 |
| pref_model   | 全长度        | loss_ce+e^-loss_length                    | 0.904   | 4069.54 |
| pref_model   | 8:2偏好：长度 | loss_ce+e^-loss_length                    | 0.896   | 4074.65 |
| pref_model   | 全长度        | loss_ce+loss_length^0.2 (放大loss_length) |         |         |



Res，添加sigmoid层（目前感觉sigmoid层好像效果很差）

| Model/Metric | 数据   | Loss                    | Overlap | Length  |
| ------------ | ------ | ----------------------- | ------- | ------- |
| pref_model   | 全长度 | loss_ce                 | 0.498   | 2778.35 |
| pref_model   | 全长度 | loss_ce+10*loss_length  | 0.498   | 2789.31 |
| pref_model   | 全长度 | loss_ce+loss_length^0.2 | 0.508   | 2557.61 |



Res，添加Relu层

| Model/Metric | 数据   | Loss                    | Overlap | Length  |
| ------------ | ------ | ----------------------- | ------- | ------- |
| model        | 全偏好 | 交叉熵loss_ce           | 0.883   | 4026.11 |
| pref_model   | 全长度 | loss_ce+30*loss_length  | 0.524   | 2506.71 |
| pref_model   | 全长度 | loss_ce+loss_length^0.2 | 0.896   | 4062.63 |



模型微调思路：https://s.craft.me/EJijIQB3y8TRGo

Code调试：https://s.craft.me/7eSsucs5cS4lLF
