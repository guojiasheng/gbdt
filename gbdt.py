from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import *
from sklearn import metrics  
import numpy as np
import sys  
import os  
import time  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class gbdt_paramter():
	_nams = ["loss","learning_rate","n_estimators","max_depth","subsample","cv","p"]
	
	def __init__(self,option = None):
		if option == None:
		    option = ''
		self.parse_option(option);

	def parse_option(self,option):
		if isinstance(option,list):
			argv = option
		elif isinstance(option,str):
			argv = option.split();
		else:
			raise TypeError("arg 1 should be a list or a str .");
		self.set_to_default_values();
		
		i=0
		while i< len(argv):
			if argv[i] == "-ls":
				i = i+1
				self.loss = argv[i]
			elif argv[i] == "-lr":
				i = i+1
				self.learning_rate = float(argv[i])
			elif argv[i] == "-ns":
				i = i+1
				self.n_estimators = int(argv[i])	
			elif argv[i] == "-md":
				i = i+1
				self.max_depth = int(argv[i])
			elif argv[i] == "-sub":
				i = i+1
				self.subsample = float(argv[i])
			elif argv[i] == "-cv":
				i = i+1
				self.cv = int(argv[i])
			elif argv[i] == "-p":
				i = i+1
				self.p = argv[i]
			else:
				raise ValueError("Wrong options.Only -ls(loss) -lr(learning_rate) -ns(n_estimators) -md(max_depth) -sub(subssample),-cv,-p(testFile)")
			i += 1

	def set_to_default_values(self):
		self.loss = "deviance"
		self.learninig_rate = 0.1
		self.n_estimators = 100
		self.max_depth = 3
		self.subsample = 1
		self.cv=0
		self.p=""

def read_data(data_file):
	try:
		
		t_X,t_y=load_svmlight_file(data_file)
		return t_X,t_y
	except ValueError as e:
		print(e)


# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y,para):  
    model = GradientBoostingClassifier(n_estimators=para.n_estimators)
    model.fit(train_x, train_y)  
    return model 

if __name__ == '__main__':
	def exit_with_help():
		print("Usage: gbdt.py  [-ls (loss: deviance,exponential),-lr(learning_rate 0.1),-ns(n_estimators 100),-md(max_depth 3),-sub(subsample 1),-cv (10),-p testFile] dataset")
		sys.exit(1)
	if len(sys.argv)<2:
		exit_with_help();
	dataset_path_name = sys.argv[-1]
	option = sys.argv[1:-1]
	try:
		train_X,train_Y=read_data(dataset_path_name)
		train_X=train_X.todense()
		para = gbdt_paramter(option)
		gbdt=gradient_boosting_classifier(train_X, train_Y,para)
	
		if para.cv>0:
			accuracy = cross_val_score(gbdt, train_X, train_Y, cv=10, scoring='accuracy')
                	roc = cross_val_score(gbdt, train_X, train_Y, cv=10, scoring='roc_auc')
			print "10 cross validation result"
			print "ACC:"+str(accuracy.mean());
			print "AUC:"+str(roc.mean());
			predicted = cross_val_predict(gbdt, train_X,train_Y, cv=10)
			print "confusion_matrix"
			print metrics.confusion_matrix(train_Y, predicted)
			print "The feature importances (the higher, the more important the feature)"
			print gbdt.feature_importances_
		if para.p!="":
			test_x,test_y = read_data(para.p);
			predict = gbdt.predict(test_x.todense());
			prob = gbdt.predict_proba(test_x.todense());
			out = open('predict','wb');
			out.write("origin"+"\t"+"predict"+"\t"+"prob"+"\n")
			for i in range(predict.shape[0]):
				if (i%1000==0):
					print "instance:"+str(i);
				out.write(str(test_y[i])+"\t"+str(predict[i])+"\t"+str(prob[i])+'\n')	
			out.close();
        except(IOError,ValueError) as e:
		sys.stderr.write(str(e) + '\n')
		sys.exit(1)
