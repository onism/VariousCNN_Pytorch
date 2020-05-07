    ref:https://medium.com/heuritech/learning-deep-neural-networks-incrementally-3e005e4fb4bc

# Learning Deep Neural Networks Incrementally
为什么让深度学习持续学习？

1. 我们的模型不能在每次需要学习新事实时都回顾所有以前的知识。例如作为一个九年级的孩子，你没有必要复习八年级的所有教学大纲。

2. 我们的模型需要不断地学习，而不忘记任何以前学过的知识。

论文 [CORe50: a New Dataset and Benchmark for Continuous Object Recognition](https://arxiv.org/abs/1705.03550) 提出了三种场景

1. Learning new data of known classes (online learning)
2. Learning new classes (class-incremental learning)
3. The union of the two previous scenarios

这篇博客主要讨论第二种情况即每次新数据出现时模型只能看到新类别的数据，但是经过学习之后不能忘掉之前的类别信息。

最原始的思路时迁移学习,但是迁移学习存在灾难性遗忘(catastrophic forgetting)问题，为了解决这一问题，需要在 rigidity (being good on old tasks) 和 plasticity (being good on new tasks)  之间寻求平衡。

论文[Continual Lifelong Learning with Neural Networks: A Review](https://arxiv.org/abs/1802.07569)提出了三种策略：

1. **External Memory** storing a small amount of previous tasks data
2. **Constraints-based methods** avoiding forgetting on previous tasks
3. **Model Plasticity** extending the capacity

External Memory：即放宽一点不能接触之前数据的限制。

Constraints-based methods：最直观的思路是保持当前模型和之前模型的相似性可以避免遗忘信息。目前的工作主要集中在
balance a rigidity (encouraging similarity between the two model versions) and plasticity (letting enough slack to the new model to learn new classes).主要分为三类：
1. Those enforcing a similarity of the activations
2. Those enforcing a similarity of the weights
3. And those enforcing a similarity of the gradients

Plasticity：即修改网络结构，第一种策略是增加神经元到当前的模型中。

If its loss is not good enough, new neurons are added at several layers and they will be dedicated to learn on the new task. Furthermore the authors choose to freeze some of the already-existing neurons. Those neurons, that are particularly important for the old tasks, must not change in order to reduce forgetting.