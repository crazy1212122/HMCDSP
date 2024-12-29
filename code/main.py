import os
import time
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.pyplot as plt
from random import random
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier
import dataget
from model import GCN_circ, GCN_dis, GCN_drug
import evaluation_scores
from param import parameter_parser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run(n_fold):
    args = parameter_parser()
    dataset, cd_pairs = dataget.dataset(args)

    kf = KFold(n_splits = n_fold, shuffle = True)


    # model_cir = GCN_circ(args)
    model_drug = GCN_drug(args)
    # model_dis = GCN_dis(args)
    
    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
    ave_spec = 0
    localtime = time.asctime( time.localtime(time.time()) )
    fprs, tprs, roc_aucs = [], [], []
    precisions, recalls, aps = [], [], []

    
    with open('/path/result/results/cir_drug_fea_16dim.txt', 'a') as f:
        f.write('time:\t'+ str(localtime)+"\n")
        
        for fold_index, (train_index, test_index) in enumerate(kf.split(cd_pairs)):
            c_drugmatix,train_cd_pairs,test_cd_pairs = dataget.C_Dmatix(cd_pairs,train_index,test_index)
            dataset['c_d']=c_drugmatix
        
            cir_fea, drugfea = dataget.feature_representation(model_drug, args, dataset)
            print(cir_fea.shape)  ##1885,256
            print(drugfea.shape)  ##27, 256
            # print(dis_fea.shape)
            #加入表征
            drug_smile_fea = np.load('/path/drug_embeddings.npy')
            if drugfea.shape == drug_smile_fea.shape:
                drugfea = drugfea + drug_smile_fea/2
            else:
                print("Drug_fea dis-match")
            
            circ_seq_fea = np.load('/path/circ_embeddings_matrix.npy')
            if cir_fea.shape == circ_seq_fea.shape:
                cir_fea = cir_fea + circ_seq_fea
            else:
                print("circ_fea dis-match")
            train_dataset = dataget.new_dataset(cir_fea, drugfea,train_cd_pairs)
            test_dataset = dataget.new_dataset(cir_fea, drugfea,test_cd_pairs)

            X_train, y_train = train_dataset[:,:-2], train_dataset[:,-2:][:,0]
            X_test, y_test = test_dataset[:,:-2], test_dataset[:,-2:][:,0]

            print(X_train.shape,X_test.shape)
            #####different classifiers' comparsion

            clf = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam', max_iter=360, random_state=42,alpha=0.001,learning_rate='constant')
            clf.fit(X_train, y_train)


            y_pred = clf.predict(X_test) 
            y_prob = clf.predict_proba(X_test)
            y_prob = y_prob[:, 1]
            tp, fp, tn, fn, acc, prec, sens, f1_score, MCC, AUC,AUPRC, spec = evaluation_scores.calculate_performace(len(y_pred), y_pred, y_prob, y_test) 
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            roc_aucs.append(roc_auc)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            precisions.append(precision)
            recalls.append(recall)
            aps.append(ap)
       
            
            ### output the result metrics
            print('RF: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  sens = \t', sens, '\n  f1_score = \t', f1_score, '\n  MCC = \t', MCC, '\n  AUC = \t', AUC,'\n  AUPRC = \t', AUPRC,'\n  spec = \t', spec)
            f.write('RF: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(acc)+'\t  prec = \t'+ str(prec)+ '\t  sens = \t'+str(sens)+'\t  f1_score = \t'+str(f1_score)+ '\t  MCC = \t'+str(MCC)+'\t  AUC = \t'+ str(AUC)+'\t  AUPRC = \t'+ str(AUPRC)+'\n'+'\t  spec = \t'+ str(spec)+'\n')
            ave_acc += acc
            ave_prec += prec
            ave_sens += sens
            ave_f1_score += f1_score
            ave_mcc += MCC
            ave_auc += AUC
            ave_auprc  += AUPRC
            ave_spec += spec

        
        ave_acc /= n_fold
        ave_prec /= n_fold
        ave_sens /= n_fold
        ave_f1_score /= n_fold
        ave_mcc /= n_fold
        ave_auc /= n_fold
        ave_auprc /= n_fold
        ave_spec /= n_fold
        print('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n'+'\t  spec = \t'+ str(ave_spec)+'\n')
        f.write('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n'+'\t  spec = \t'+ str(ave_spec)+'\n')


if __name__ == "__main__":
    args = parameter_parser()

    k_fold = args.fold ## default is 5
    rounds = args.round ## default is 10

    for i in range(rounds):
        run(k_fold)