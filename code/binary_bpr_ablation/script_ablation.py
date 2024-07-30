import os

epochs = [70, 95, 75, 75, 90]
datasets = ["assist09_tkde", "assist17_tkde", "algebra", "math_1", "math_2"]
for d,dataset in enumerate(datasets) :
    print(f'----------{dataset}----------')
    for i in range(1,4):
        print("Ablation (0 no ablation, 1 ablation L2, 2 ablation init, 3 both) ",i)
        for a in range(5):
            cmd = "python main.py --epochs " + str(epochs[d]) +" --dataTrain ../../data/cdbpr_format/" + str(dataset) + "/train_valid_" + str(a) + ".csv --dataTest ../../data/cdbpr_format/" + str(dataset) + "/test_" + str(a) + ".csv --ablation " + str(i)
            os.system(cmd)
