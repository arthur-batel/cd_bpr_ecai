import os
dPath = "../../data/cdbpr_format/"
embDirPath = "../../results/table_2/"
datasets = ['assist09_tkde', 'assist17_tkde', 'algebra','math_1', 'math_2']
epochs = [70, 95, 75, 75, 90]
batchSize =[ 512, 512,512, 512,512]
learningRate = [0.01,0.01,0.01,0.01,0.01]
mode = [1,1,1,1,1]
for i in range(len(datasets)):
    print(datasets[i])
    cmd = 'python main.py --dataTrain ' + datasets[i] +'/train_valid_0.csv --dataTest ' + datasets[i] +'/test_0.csv --epochs ' + str(epochs[i]) +' --batchSize 512 --learningRate 0.01 --mode 1 --dataPath ' + dPath + " --embPath " + embDirPath
    os.system(cmd)
