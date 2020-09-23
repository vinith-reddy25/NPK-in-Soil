def ConvertToClassification(df):
 #columns_name = ['N1','N2','N3','P1','P2','P3','K1','K2','K3']
 #tempdf=pd.DataFrame(columns = columns_name)
counter = 0
arrN = np.zeros([df.shape[0],5])
 arrP = np.zeros([df.shape[0],5])
 arrK = np.zeros([df.shape[0],5])
 for i,j in df.iterrows():
 if(j['N']<140):
 arrN[counter][0]=1
 elif (j['N']<300):
 arrN[counter][1]=1
 elif (j['N']<500):
 arrN[counter][2]=1
 else:
 arrN[counter][3]=1
 if(j['P']<12):
 arrP[counter][0]=1
 elif (j['P']<20):
 arrP[counter][1]=1
 elif (j['P']<59):
 arrP[counter][2]=1
 elif(j['P']<90):
 arrP[counter][3]=1
 else:
 arrP[counter][4]=1

 if(j['K']<100):
 arrK[counter][0]=1
 elif (j['K']<144):
 arrK[counter][1]=1
 elif (j['K']<338):
 arrK[counter][2]=1
 elif(j['K']<658):
 arrK[counter][3]=1
 else:
 arrK[counter][4]=1
 counter= counter+1
 return arrN,arrP,arrK