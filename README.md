# Pref_Route

We aim to construct a general learning-based model that can generate different proper routes according to the preferences of users, such as shorter path, faster path, or general path.



code

- model.py: 路线生成模型原始代码
- train.py: 原模型训练代码
- valid.py: 验证代码
- pref_train.py: 考虑多偏好场景的路线生成模型训练代码
- pref_valid.py: 验证代码
- stat.py: 数据统计分析



Res

| Model/Metric | Overlap | Length  |
| ------------ | ------- | ------- |
| model        | 0.904   | 4082.05 |
| pref_model   | 0.606   | 3112.57 |



模型微调思路：https://s.craft.me/EJijIQB3y8TRGo

Code调试：https://s.craft.me/7eSsucs5cS4lLF
