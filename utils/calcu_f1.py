recall=[0.25273,0.37657,0.46032,0.51771]
precision=[0.34251,0.26386,0.21801,0.18554]
recall=[0.19774, 0.44355]
precision=[0.27401, 0.15917]
for i in range(len(recall)):
    f1=2*precision[i]*recall[i]/(precision[i]+recall[i])
    print(f"F1@{i+1}: {f1:.5f}")
