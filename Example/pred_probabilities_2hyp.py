import numpy as np

# dir_list = [ 'Data_20_2hyp/', 'Data_50_2hyp/', 'Data_100_2hyp/' ]
# # Iterate through dir_list using dir_name

# for dir_name in dir_list:

# 	fire2 = np.load(str(dir_name + 'Preds/pred_fire2.npy'), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
# 	fire5 = np.load(str(dir_name + 'Preds/pred_fire5.npy'), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
# 	fire10 = np.load(str(dir_name + 'Preds/pred_fire10.npy'), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
# 	fire20 = np.load(str(dir_name + 'Preds/pred_fire20.npy'), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]
# 	fire50 = np.load(str(dir_name + 'Preds/pred_fire50.npy'), mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII').reshape(300,99)[0:100,:]

# 	# intervals - halfs with open lower bound and closed upper bound
# 	# [0.25 - 0.5] -- 0.5
# 	# (0.75 - 1.0] -- 1.0

# 	prob_list_fire2 = []
# 	prob_list_fire5 = []
# 	prob_list_fire10 = []
# 	prob_list_fire20 = []
# 	prob_list_fire50 = []

# 	for i in xrange(100):

# 		# classes: 1,2 COUNTS
# 		c1_fire2 = 1
# 		c2_fire2 = 1
# 		total_fire2 = 2
# 		prob_fire2 = []

# 		c1_fire5 = 1
# 		c2_fire5 = 1
# 		total_fire5 = 2
# 		prob_fire5 = []

# 		c1_fire10 = 1
# 		c2_fire10 = 1
# 		total_fire10 = 2
# 		prob_fire10 = []

# 		c1_fire20 = 1
# 		c2_fire20 = 1
# 		total_fire20 = 2
# 		prob_fire20 = []

# 		c1_fire50 = 1
# 		c2_fire50 = 1
# 		total_fire50 = 2
# 		prob_fire50 = []

# 		for j in xrange(99):

# 			if fire2[i][j] <= 0.5:
# 				fire2[i][j] = 0.5
# 				c1_fire2 += 1
# 			elif fire2[i][j] <= 1.0:
# 				fire2[i][j] = 1.0
# 				c2_fire2 += 1
# 			total_fire2 += 1
# 			prob_fire2.append(float(c1_fire2)/total_fire2)
# 			prob_fire2.append(float(c2_fire2)/total_fire2)
# 			# float("{0:.4f}".format(x))

# 			if fire5[i][j] <= 0.5:
# 				fire5[i][j] = 0.5
# 				c1_fire5 += 1
# 			elif fire5[i][j] <= 1.0:
# 				fire5[i][j] = 1.0
# 				c2_fire5 += 1
# 			total_fire5 += 1
# 			prob_fire5.append(float(c1_fire5)/total_fire5)
# 			prob_fire5.append(float(c2_fire5)/total_fire5)

# 			if fire10[i][j] <= 0.5:
# 				fire10[i][j] = 0.5
# 				c1_fire10 += 1
# 			elif fire10[i][j] <= 1.0:
# 				fire10[i][j] = 1.0
# 				c2_fire10 += 1
# 			total_fire10 += 1
# 			prob_fire10.append(float(c1_fire10)/total_fire10)
# 			prob_fire10.append(float(c2_fire10)/total_fire10)

# 			if fire20[i][j] <= 0.5:
# 				fire20[i][j] = 0.5
# 				c1_fire20 += 1
# 			elif fire20[i][j] <= 1.0:
# 				fire20[i][j] = 1.0
# 				c2_fire20 += 1
# 			total_fire20 += 1
# 			prob_fire20.append(float(c1_fire20)/total_fire20)
# 			prob_fire20.append(float(c2_fire20)/total_fire20)

# 			if fire50[i][j] <= 0.5:
# 				fire50[i][j] = 0.5
# 				c1_fire50 += 1
# 			elif fire50[i][j] <= 1.0:
# 				fire50[i][j] = 1.0
# 				c2_fire50 += 1
# 			total_fire50 += 1
# 			prob_fire50.append(float(c1_fire50)/total_fire50)
# 			prob_fire50.append(float(c2_fire50)/total_fire50)

# 		prob_list_fire2.append(prob_fire2)
# 		prob_list_fire5.append(prob_fire5)
# 		prob_list_fire10.append(prob_fire10)
# 		prob_list_fire20.append(prob_fire20)
# 		prob_list_fire50.append(prob_fire50)

# 	prob_list_fire2 = np.array(prob_list_fire2)
# 	prob_list_fire5 = np.array(prob_list_fire5)
# 	prob_list_fire10 = np.array(prob_list_fire10)
# 	prob_list_fire20 = np.array(prob_list_fire20)
# 	prob_list_fire50 = np.array(prob_list_fire50)

# 	np.savetxt(str(dir_name + "Class_Pred_Probs/pred_fire2.csv"), prob_list_fire2, delimiter=",")
# 	np.savetxt(str(dir_name + "Class_Pred_Probs/pred_fire5.csv"), prob_list_fire5, delimiter=",")
# 	np.savetxt(str(dir_name + "Class_Pred_Probs/pred_fire10.csv"), prob_list_fire10, delimiter=",")
# 	np.savetxt(str(dir_name + "Class_Pred_Probs/pred_fire20.csv"), prob_list_fire20, delimiter=",")
# 	np.savetxt(str(dir_name + "Class_Pred_Probs/pred_fire50.csv"), prob_list_fire50, delimiter=",")

pred_list = [0.15391952,0.24134205,0.29485783,0.34662443,0.38030717,0.38399696,0.38482928,0.40323919,
			0.41694078,0.40913364,0.40191504,0.41442332,0.42443463,0.41447333,0.4056372,0.41689038,
			0.42610329,0.41567343,0.40647888,0.41744986,0.42648256,0.41594672,0.40667084,0.41757756,
			0.42656922,0.41600925,0.40671474,0.41760677,0.42658898,0.41602349,0.40672478,0.41761342,
			0.42659351,0.41602671,0.40672699,0.41761491,0.4265945,0.41602746,0.40672752,0.41761526,
			0.42659479,0.41602761,0.4067277,0.41761535,0.42659482,0.41602767,0.4067277,0.41761538,
			0.42659485,0.41602769,0.4067277,0.41761538,0.42659485,0.41602769,0.4067277,0.41761538,
			0.42659485,0.41602769,0.4067277,0.41761538,0.42659485,0.41602769,0.4067277,0.41761538,
			0.42659485,0.41602769,0.4067277,0.41761538,0.42659485,0.41602769,0.4067277,0.41761538,
			0.42659485,0.41602769,0.4067277,0.41761538,0.42659485,0.41602769,0.4067277,0.41761538,
			0.42659485,0.41602769,0.4067277,0.41761538,0.42659485,0.41602769,0.4067277,0.41761538,
			0.42659485,0.41602769,0.4067277,0.41761538,0.42659485,0.41602769,0.4067277,0.41761538,
			0.42659485,0.41602769,0.4067277]

c1 = 1
c2 = 1
total = 2
prob1 = []
prob2 = []
for j in xrange(99):
	if pred_list[j] <= 0.5:
		c1 += 1
	elif pred_list[j] <= 1.0:
		c2 += 1
	total += 1
	prob1.append(float(c1)/total)
	prob2.append(float(c2)/total)

print prob1
print prob2