
# Using readlines()
file1 = open('logger/logs/log_GRU_with_attention.txt', 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
loss=[]
accuracy =[]
precision_0  =[]
precision_1  =[]
precision_2  =[]
precision_3 =[]
precision_4  =[]
precision_5  =[]
precision_6  =[]
recall_0 =[]
recall_1 =[]
recall_2 =[]
recall_3 =[]
recall_4 =[]
recall_5 =[]
recall_6 =[]
f1_0 =[]
f1_1 =[]
f1_2 =[]
f1_3 =[]
f1_4 =[]
f1_5 =[]
f1_6 =[]
val_loss =[]
val_accuracy =[]
val_precision_0 =[]
val_precision_1 =[]
val_precision_2 =[]
val_precision_3 =[]
val_precision_4 =[]
val_precision_5 =[]
val_precision_6 =[]
val_recall_0 =[]
val_recall_1 =[]
val_recall_2 =[]
val_recall_3 =[]
val_recall_4 =[]
val_recall_5 =[]
val_recall_6 =[]
val_f1_0 =[]
val_f1_1 =[]
val_f1_2 =[]
val_f1_3 =[]
val_f1_4 =[]
val_f1_5 =[]
val_f1_6 =[]
no = [":"," ",",",'','|',':']
for line in Lines:
    count += 1

    # print("Line{}: {}".format(count, line.strip("")))
    line = line.strip()
    # line.split("")
    str_list = line.split(" ")
    # print(str_list)
    str_list = [x for x in str_list if x not in no]
    # print(str_list)
    if len(str_list) > 0:
        # print(str_list[-1])
        if str_list[0] == 'loss':
            loss.append(str_list[-1])
        if str_list[0] =='accuracy':
            accuracy.append(str_list[-1])
        if str_list[0] == 'precision_0':
            precision_0.append(str_list[-1])
        if str_list[0] ==   'precision_1':
            precision_1.append(str_list[-1])
        if str_list[0] == 'precision_2':
            precision_2.append(str_list[-1])
        if str_list[0] == 'precision_3':
            precision_3.append(str_list[-1])
        if str_list[0] == 'precision_4':
            precision_4.append(str_list[-1])
        if str_list[0] == 'precision_5':
            precision_5.append(str_list[-1])
        if str_list[0] == 'precision_6':
            precision_6.append(str_list[-1])
        if str_list[0] == 'recall_0':
            recall_0.append(str_list[-1])
        if str_list[0] == 'recall_1':
            recall_1.append(str_list[-1])
        if str_list[0] == 'recall_2':
            recall_2.append(str_list[-1])
        if str_list[0] == 'recall_3':
            recall_3.append(str_list[-1])
        if str_list[0] == 'recall_4':
            recall_4.append(str_list[-1])
        if str_list[0] == 'recall_5':
            recall_5.append(str_list[-1])
        if str_list[0] == 'recall_6':
            recall_6.append(str_list[-1])
        if str_list[0] == 'f1_0':
            f1_0.append(str_list[-1])
        if str_list[0] == 'f1_1':
            f1_1.append(str_list[-1])
        if str_list[0] == 'f1_2':
            f1_2.append(str_list[-1])
        if str_list[0] == 'f1_3':
            f1_3.append(str_list[-1])
        if str_list[0] == 'f1_4':
            f1_4.append(str_list[-1])
        if str_list[0] == 'f1_5':
            f1_5.append(str_list[-1])
        if str_list[0] == 'f1_6':
            f1_6.append(str_list[-1])
        if str_list[0] == 'val_loss':
            val_loss.append(str_list[-1])
        if str_list[0] == 'val_accuracy':
            val_accuracy.append(str_list[-1])
        if str_list[0] == 'val_precision_0':
            val_precision_0.append(str_list[-1])
        if str_list[0] == 'val_precision_1':
            val_precision_1.append(str_list[-1])
        if str_list[0] == 'val_precision_2':
            val_precision_2.append(str_list[-1])
        if str_list[0] == 'val_precision_3':
            val_precision_3.append(str_list[-1])
        if str_list[0] == 'val_precision_4':
            val_precision_4.append(str_list[-1])
        if str_list[0] == 'val_precision_5':
            val_precision_5.append(str_list[-1])
        if str_list[0] == 'val_precision_6':
            val_precision_6.append(str_list[-1])
        if str_list[0] == 'val_recall_0':
            val_recall_0.append(str_list[-1])
        if str_list[0] == 'val_recall_1':
            val_recall_1.append(str_list[-1])
        if str_list[0] == 'val_recall_2':
            val_recall_2.append(str_list[-1])
        if str_list[0] == 'val_recall_3':
            val_recall_3.append(str_list[-1])
        if str_list[0] == 'val_recall_4':
            val_recall_4.append(str_list[-1])
        if str_list[0] == 'val_recall_5':
            val_recall_5.append(str_list[-1])
        if str_list[0] == 'val_recall_6':
            val_recall_6.append(str_list[-1])
        if str_list[0] == 'val_f1_0':
            val_f1_0.append(str_list[-1])
        if str_list[0] == 'val_f1_1':
            val_f1_1.append(str_list[-1])
        if str_list[0] == 'val_f1_2':
            val_f1_2.append(str_list[-1])
        if str_list[0] == 'val_f1_3':
            val_f1_3.append(str_list[-1])
        if str_list[0] == 'val_f1_4':
            val_f1_4.append(str_list[-1])
        if str_list[0] == 'val_f1_5':
            val_f1_5.append(str_list[-1])
        if str_list[0] == 'val_f1_6':
            val_f1_6.append(str_list[-1])
print(loss)
