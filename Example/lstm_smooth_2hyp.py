import random
import numpy as np
import math
import numpy.polynomial.polynomial as poly
import matplotlib
import matplotlib.pyplot as plt
import operator
import matplotlib.backends.backend_pdf
from scipy.stats import *
import csv

############# Generating Utilities according to Boulware #########
def boulwareUtilities (rv,Deadline):
	ut = []
	beta = 5.2
	beta = float(1)/beta
	for i in range(1,Deadline+1):
		minm = min(i,Deadline)
		time = float(minm)/Deadline
		curr_ut = rv + (1-rv)*(math.pow(time,beta))
		# print "================"
		# print minm
		# print time
		# print beta
		# print "================"
		ut.append(float("{0:.4f}".format(curr_ut)))
	return ut

#############################################################

########## Generating Utilities according to Tim Barslaag #########

def GenerateTimUtility( rv,rounds):
	l=[]
	l.append(rv);
	for i in range(1,rounds):
		l.append(float((l[i-1]+1)*(l[i-1]+1))/4)
	return l

###################################################################

def getflag(direction,Gridcoords,GridSize):
	flag=0
	if(direction==1):
		if(Gridcoords[0]!=0):
			Gridcoords[0]-=1                 ### Moving North
		if(Gridcoords[1]==GridSize-1):
			flag=3
		elif(Gridcoords[1]==0):
			flag=2
		elif(Gridcoords[0]+1==GridSize-1):
			flag=4
	elif(direction==2):
		if(Gridcoords[1]!=0):
			Gridcoords[1]-=1                 ### Moving West
		if(Gridcoords[0]==0):
			flag=1
		elif(Gridcoords[1]+1==GridSize-1):
			flag==3
		elif(Gridcoords[0]==GridSize-1):
			flag=4

	elif(direction==3):
		if(Gridcoords[1]!=GridSize-1):
			Gridcoords[1]+=1                 ### Moving East
		if(Gridcoords[0]==GridSize-1):
			flag=4
		elif(Gridcoords[0]==0):
			flag=1
		elif(Gridcoords[1]-1==0):
			flag=2
	else:
		if(Gridcoords[0]!=GridSize-1):
			Gridcoords[0]+=1                 ### Moving South
		if(Gridcoords[0]-1==0):
			flag=1
		elif(Gridcoords[1]==GridSize-1):
			flag=3
		elif(Gridcoords[1]==0):
			flag=2
	return flag
	
#############################################################
def Firerv(RV,roundnum,Deadline,UpdateRate,GridSize,Gridcoords):
	ManPower=[12,10,7,4]
	Utilities=[0.75,0.57,0.321,0.12]
	
	if(roundnum==0):
		# print "---round 1: =="
		direction=random.randint(1,4)
		# direction=random.choice([1,4])
		### Gridcoords Updation 
		flag=getflag(direction,Gridcoords,GridSize)
		# print "------"
		# print direction
		# print Gridcoords
		# print "------"
		if(flag==0 ):
			return Utilities[direction-1]
			#return getReservationUtility(ManPower[direction-1])

		else:	

			### commented by Kritika
			#print "This case: "+str(flag) + " "+ str(ManPower[direction-1] )
			###

			# return getReservationUtility( max (ManPower[flag-1], ManPower[direction-1] )) 
			return max (Utilities[flag-1], Utilities[direction-1] )

	elif(roundnum%UpdateRate==0):
		# print "---update == " + str(roundnum)   
		direction=random.randint(1,4)
		# direction=random.choice([1,4])
		flag=getflag(direction,Gridcoords,GridSize)
		# print "------"
		# print direction
		# print Gridcoords
		# print "------"
		if(flag==0 ):
			# return getReservationUtility(ManPower[direction-1])
			return Utilities[direction-1]

		else:	
			# print "This case: "+str(flag) + " "+ str(ManPower[direction-1] )
			# return getReservationUtility( max (ManPower[flag-1], ManPower[direction-1] ) )
			return Utilities[direction-1]

	else:
		return RV[len(RV)-1]


#############################################################

def getprobability( rows ):
	#probabilities=[[0.25,0.25,0.25,0.25]]
	probabilities=[[0.5,0.5]]
	cnt =0
	l=[]
	for i in rows:
		l.append(float(i))
		cnt=cnt+1
		if(cnt%2==0):
			probabilities.append(l)
			l=[]
	return probabilities


if __name__ == '__main__':


	Average_rv=[]
	AverageUtilities_Tims=[]
	AverageUtilities_lstm=[]

	####----- CSV parsing ---########3

	rows=[]
	fields=[]

	prob_example_1 = [0.6666666666666666, 0.75, 0.8, 0.8333333333333334, 0.8571428571428571, 0.875, 0.8888888888888888, 
						0.9, 0.9090909090909091, 0.9166666666666666, 0.9230769230769231, 0.9285714285714286, 0.9333333333333333, 
						0.9375, 0.9411764705882353, 0.9444444444444444, 0.9473684210526315, 0.95, 0.9523809523809523, 0.9545454545454546, 
						0.9565217391304348, 0.9583333333333334, 0.96, 0.9615384615384616, 0.9629629629629629, 0.9642857142857143, 
						0.9655172413793104, 0.9666666666666667, 0.967741935483871, 0.96875, 0.9696969696969697, 0.9705882352941176, 
						0.9714285714285714, 0.9722222222222222, 0.972972972972973, 0.9736842105263158, 0.9743589743589743, 0.975, 
						0.975609756097561, 0.9761904761904762, 0.9767441860465116, 0.9772727272727273, 0.9777777777777777, 
						0.9782608695652174, 0.9787234042553191, 0.9791666666666666, 0.9795918367346939, 0.98, 0.9803921568627451, 
						0.9807692307692307, 0.9811320754716981, 0.9814814814814815, 0.9818181818181818, 0.9821428571428571, 
						0.9824561403508771, 0.9827586206896551, 0.9830508474576272, 0.9833333333333333, 0.9836065573770492, 
						0.9838709677419355, 0.9841269841269841, 0.984375, 0.9846153846153847, 0.9848484848484849, 0.9850746268656716, 
						0.9852941176470589, 0.9855072463768116, 0.9857142857142858, 0.9859154929577465, 0.9861111111111112, 
						0.9863013698630136, 0.9864864864864865, 0.9866666666666667, 0.9868421052631579, 0.987012987012987, 
						0.9871794871794872, 0.9873417721518988, 0.9875, 0.9876543209876543, 0.9878048780487805, 0.9879518072289156, 
						0.9880952380952381, 0.9882352941176471, 0.9883720930232558, 0.9885057471264368, 0.9886363636363636, 
						0.9887640449438202, 0.9888888888888889, 0.989010989010989, 0.9891304347826086, 0.989247311827957, 
						0.9893617021276596, 0.9894736842105263, 0.9895833333333334, 0.9896907216494846, 0.9897959183673469, 
						0.98989898989899, 0.99, 0.9900990099009901]

	prob_example_2 = [0.3333333333333333, 0.25, 0.2, 0.16666666666666666, 0.14285714285714285, 0.125, 0.1111111111111111, 0.1, 
						0.09090909090909091, 0.08333333333333333, 0.07692307692307693, 0.07142857142857142, 0.06666666666666667, 0.0625, 
						0.058823529411764705, 0.05555555555555555, 0.05263157894736842, 0.05, 0.047619047619047616, 0.045454545454545456, 
						0.043478260869565216, 0.041666666666666664, 0.04, 0.038461538461538464, 0.037037037037037035, 0.03571428571428571, 
						0.034482758620689655, 0.03333333333333333, 0.03225806451612903, 0.03125, 0.030303030303030304, 0.029411764705882353, 
						0.02857142857142857, 0.027777777777777776, 0.02702702702702703, 0.02631578947368421, 0.02564102564102564, 0.025, 
						0.024390243902439025, 0.023809523809523808, 0.023255813953488372, 0.022727272727272728, 0.022222222222222223, 
						0.021739130434782608, 0.02127659574468085, 0.020833333333333332, 0.02040816326530612, 0.02, 0.0196078431372549, 
						0.019230769230769232, 0.018867924528301886, 0.018518518518518517, 0.01818181818181818, 0.017857142857142856, 
						0.017543859649122806, 0.017241379310344827, 0.01694915254237288, 0.016666666666666666, 0.01639344262295082, 
						0.016129032258064516, 0.015873015873015872, 0.015625, 0.015384615384615385, 0.015151515151515152, 0.014925373134328358, 
						0.014705882352941176, 0.014492753623188406, 0.014285714285714285, 0.014084507042253521, 0.013888888888888888, 
						0.0136986301369863, 0.013513513513513514, 0.013333333333333334, 0.013157894736842105, 0.012987012987012988, 
						0.01282051282051282, 0.012658227848101266, 0.0125, 0.012345679012345678, 0.012195121951219513, 0.012048192771084338, 
						0.011904761904761904, 0.011764705882352941, 0.011627906976744186, 0.011494252873563218, 0.011363636363636364, 
						0.011235955056179775, 0.011111111111111112, 0.01098901098901099, 0.010869565217391304, 0.010752688172043012, 
						0.010638297872340425, 0.010526315789473684, 0.010416666666666666, 0.010309278350515464, 0.01020408163265306, 
						0.010101010101010102, 0.01, 0.009900990099009901]

	rows = []
	for _i in xrange(len(prob_example_1)):
		rows.append(prob_example_1[_i])
		rows.append(prob_example_2[_i])

	print len(rows)

	rows = np.array(rows)
	print rows.shape

	for iterations in xrange(1,2):
		probabilities=getprobability(rows)
		# probabilities=getprobability(rows[22])
		# print probabilities[1]

		RV=[0]
		Deadline = 100
		intervals=2
		UpdateRate=2     ##### keep updating the updaterate according to csv file parsed
		random_rv=[0.1,0.9]

		# iterations=1

		Utilities=[]
		actual_utility=[]

		for rv in random_rv:
			Utilities.append(GenerateTimUtility(rv,Deadline))
			# Utilities.append(boulwareUtilities(rv,Deadline))
		new_probability = probabilities
		lstmUtilities=[]

		x=[]
		for i in xrange(1,Deadline+1):
			x.append(i)
		x_belief=[]
		for i in xrange(0,Deadline+1):
			x_belief.append(i)

		GridSize=20
		Gridcoords=[GridSize/2 ,GridSize/2]

		####------ Negotiation starts ------######
		for roundnum in xrange(1,Deadline+1):
			new_CombinedUtility=0
			for i in xrange(0,len(new_probability[0])):
				new_CombinedUtility+=new_probability[roundnum-1][i]*Utilities[i][len(Utilities[i])-roundnum]

			lstmUtilities.append(float("{0:.4f}".format(new_CombinedUtility)))
			# actual_utility.append(float("{0:.4f}".format(utility_RV[len(utility_RV)-roundnum])))
		print lstmUtilities

		##### --------- aVerages over iterations ---#####
		if(iterations==1):
			Average_rv=RV
			# AverageUtilities_Tims=actual_utility
			AverageUtilities_lstm=lstmUtilities

		else:
			Average_rv=np.array(Average_rv,dtype=float)*(iterations-1)
			# AverageUtilities_Tims=np.array(AverageUtilities_Tims,dtype=float)*(iterations-1)
			AverageUtilities_lstm=np.array(AverageUtilities_lstm,dtype=float)*(iterations-1)

			# print Average_rv
			Average_rv=map(operator.add,Average_rv,RV)
			# AverageUtilities_Tims=map(operator.add,AverageUtilities_Tims,actual_utility)
			AverageUtilities_lstm=map(operator.add,AverageUtilities_lstm,lstmUtilities)

			Average_rv=np.array(Average_rv)/iterations
			# AverageUtilities_Tims=np.array(AverageUtilities_Tims)/iterations
			AverageUtilities_lstm=np.array(AverageUtilities_lstm)/iterations

		### Commented by Kritika
		# print "---- " + str(iterations) + " -----"
		###

	
	lstmError=0

	for i in xrange(2,6):
		lstm_fit=np.polyfit(x,AverageUtilities_lstm,i,full=True)

		if(i==2):
			
			lstmError=lstm_fit[1]

			lstm_index=i

		else:
			
			if(lstm_fit[1]<lstmError):
				lstmError=lstm_fit[1]
				lstm_index=i


	legend_properties = {'weight':'bold', 'size':20}			
	

	plt.figure('AverageUtilities lstm')
	plt.title('LSTM',fontsize=20, fontweight='bold')
	coefs=poly.polyfit(x,AverageUtilities_lstm,lstm_index)
	ffit=poly.polyval(x,coefs)

	
	Bay,=plt.plot(x,AverageUtilities_lstm, linestyle='-', color='k', linewidth=1.5)
	Bayfit,=plt.plot(x,ffit, linestyle='--', color='g', linewidth=3.5)
	plt.legend([Bay,Bayfit],["LSTM Utilities","Fitted Utilities"],loc=6,ncol=1, handlelength=4,prop=legend_properties)

	plt.yticks(fontsize=20,fontweight='bold')
	plt.xticks(fontsize=20,fontweight='bold')
	# plt.plot(Average_rv,'ro')
	# plt.plot(AverageUtilities_Normalised,'r--',ffit,'g--')
	plt.xlabel('Rounds',fontsize=20, fontweight='bold')
	plt.ylabel('Utilities',fontsize=20, fontweight='bold')
	plt.savefig('lstm.pdf',format='pdf', dpi=1000)

	### Commented by Kritika
	# print '######################'
	# print "Smoothness"
	###
	print lstmError

	### Commented by Kritika
	# print '######################'
	###