# gbdt
gdbt implement by scikit-learn

参数说明：
     ls：    loss默认是deviance,包括（deviance、exponential）,deviance 其实就是logistic regression，exponential 指数损失函数，其实这时候就编程adaboost了
     lr：    learning_rate,代表每个tree的学习速率，默认是0.1,具体可以自己试验看效果
     ns:     n_estimators，代表 boosting的次数，默认是100，次数越多效果肯定越好，而且也不用担心over-fitting的问题
     md:     max_depth, 代表每一棵树的深度，  默认是3.
    sub：    subsample，每次随机选取训练数据集的比率，默认是1.就是全部用来训练。如果sub小于1的话，就变成stochastic gradient boosting.
    cv:      cross Validation,交叉验证的大小，这个k-fold要设置.
    dataset： 训练数据集，注意这边目前只输入libsvm格式的数据

     只要dataset放最后面，其他参数顺序无所谓！效果不好，就ns设高点~
                
     command：
           （1）  python gbdt.py -cv 10 heart_scale
           （2）  python gbdt.py -ns 100 -md 5  -cv 10 heart_scale
    output：
           （1）会输出交叉验证的Acc
           （2）为了应对不平衡数据，还输出了Auc。
		   （3）输出混淆矩阵，虽然样式没那么好看，但是还是能ok的。
           （4）输出feature的重要性打分，这个还是挺有用的，分值越高表示这个feature的在这个过程起到的


	2.对文件进行predict
	    当我们有测试文件 test 和 训练文化train的时候，我们用train训练model，用这个model来预测test的文件，输出文件为当前目录下的predict文件。
	    python gbdt.py -p testFile trainFile
                                

