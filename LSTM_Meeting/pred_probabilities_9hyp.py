import numpy as np

meeting2 = np.load('./Preds/pred_meeting2.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
meeting5 = np.load('./Preds/pred_meeting5.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
meeting10 = np.load('./Preds/pred_meeting10.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
meeting20 = np.load('./Preds/pred_meeting20.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
meeting50 = np.load('./Preds/pred_meeting50.npy', mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]

# intervals - 9 parts with open lower bound and closed upper bound
# [0.01, 0.142, 0.2804, 0.415, 0.5451, 0.6701, 0.7889, 0.8999, 1.0]

# 1. [0.0000 - 0.0760] -- 0.0100
# 2. (0.0760 - 0.2112] -- 0.1420
# 3. (0.2112 - 0.3477] -- 0.2804
# 4. (0.3477 - 0.4800] -- 0.4150
# 5. (0.4800 - 0.6076] -- 0.5451
# 6. (0.6076 - 0.7295] -- 0.6701
# 7. (0.7295 - 0.8444] -- 0.7889
# 8. (0.8444 - 0.9499] -- 0.8999
# 9. (0.9499 - 1.0000] -- 1.0000

prob_list_meeting2 = []
prob_list_meeting5 = []
prob_list_meeting10 = []
prob_list_meeting20 = []
prob_list_meeting50 = []

upper_bound_list = [0.0760, 0.2112, 0.3477, 0.4800, 0.6076, 0.7295, 0.8444, 0.9499, 1.0000]

for i in xrange(100):

	c_meeting2 = [1,1,1,1,1,1,1,1,1]
	total_meeting2 = 9
	prob_meeting2 = []

	c_meeting5 = [1,1,1,1,1,1,1,1,1]
	total_meeting5 = 9
	prob_meeting5 = []

	c_meeting10 = [1,1,1,1,1,1,1,1,1]
	total_meeting10 = 9
	prob_meeting10 = []

	c_meeting20 = [1,1,1,1,1,1,1,1,1]
	total_meeting20 = 9
	prob_meeting20 = []

	c_meeting50 = [1,1,1,1,1,1,1,1,1]
	total_meeting50 = 9
	prob_meeting50 = []


	for j in xrange(99):

		temp = meeting2[i][j]
		for index in xrange(9):
			if temp <= upper_bound_list[index]:
				meeting2[i][j] = upper_bound_list[index]
				c_meeting2[index] += 1
				break
		total_meeting2 += 1
		for index in xrange(9):
			prob_meeting2.append(float(c_meeting2[index])/total_meeting2)

		temp = meeting5[i][j]
		for index in xrange(9):
			if temp <= upper_bound_list[index]:
				meeting5[i][j] = upper_bound_list[index]
				c_meeting5[index] += 1
				break
		total_meeting5 += 1
		for index in xrange(9):
			prob_meeting5.append(float(c_meeting5[index])/total_meeting5)

		temp = meeting10[i][j]
		for index in xrange(9):
			if temp <= upper_bound_list[index]:
				meeting10[i][j] = upper_bound_list[index]
				c_meeting10[index] += 1
				break
		total_meeting10 += 1
		for index in xrange(9):
			prob_meeting10.append(float(c_meeting10[index])/total_meeting10)

		temp = meeting20[i][j]
		for index in xrange(9):
			if temp <= upper_bound_list[index]:
				meeting20[i][j] = upper_bound_list[index]
				c_meeting20[index] += 1
				break
		total_meeting20 += 1
		for index in xrange(9):
			prob_meeting20.append(float(c_meeting20[index])/total_meeting20)

		temp = meeting50[i][j]
		for index in xrange(9):
			if temp <= upper_bound_list[index]:
				meeting50[i][j] = upper_bound_list[index]
				c_meeting50[index] += 1
				break
		total_meeting50 += 1
		for index in xrange(9):
			prob_meeting50.append(float(c_meeting50[index])/total_meeting50)

	prob_list_meeting2.append(prob_meeting2)
	prob_list_meeting5.append(prob_meeting5)
	prob_list_meeting10.append(prob_meeting10)
	prob_list_meeting20.append(prob_meeting20)
	prob_list_meeting50.append(prob_meeting50)

prob_list_meeting2 = np.array(prob_list_meeting2)
prob_list_meeting5 = np.array(prob_list_meeting5)
prob_list_meeting10 = np.array(prob_list_meeting10)
prob_list_meeting20 = np.array(prob_list_meeting20)
prob_list_meeting50 = np.array(prob_list_meeting50)

np.savetxt("./Class_Pred_Probs/pred_meeting2.csv", prob_list_meeting2, delimiter=",")
np.savetxt("./Class_Pred_Probs/pred_meeting5.csv", prob_list_meeting5, delimiter=",")
np.savetxt("./Class_Pred_Probs/pred_meeting10.csv", prob_list_meeting10, delimiter=",")
np.savetxt("./Class_Pred_Probs/pred_meeting20.csv", prob_list_meeting20, delimiter=",")
np.savetxt("./Class_Pred_Probs/pred_meeting50.csv", prob_list_meeting50, delimiter=",")