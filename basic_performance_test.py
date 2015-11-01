import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklearn_pr
import sklearn.cross_validation as sklearn_cv
import sklearn.linear_model as sklearn_lm
import pdb

def error_measure(predicted, gtruth):
    '''
    Negative log loss 
    -1/N \sum_{i=1}{N}\sum_{j=1}{j=M} y_ij log(p_ij)
    '''
    data_num, data_dim = predicted.shape
    gtruth_num, _ = gtruth.shape
    if data_num != gtruth_num:
        raise
    error_loss = 0.
    for i in range(0, data_num):
        # normalize
        cur_predict = predicted[i, :] / sum(predicted[i, :])

def do_classification(train_feat, train_label):
    '''
    Do simple test classification
    '''
    np.random.seed(0)
    normed_feat = sklearn_pr.StandardScaler().fit_transform(train_feat)
    stratified_kfolds = sklearn_cv.StratifiedKFold(train_label, n_folds=5, shuffle=True)

    l2_logistic_rg = sklearn_lm.LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial')

    for train_index, test_index in stratified_kfolds:
        print 'Train {0}, Test {1}'.format(len(train_index), len(test_index))
        train_feat = normed_feat[train_index,:]
        test_feat = normed_feat[test_index, :]
        #pdb.set_trace()
        cur_train_label = train_label[[train_index]]
        cur_test_label = train_label[[test_index]]

        l2_logistic_rg.fit(train_feat, cur_train_label)
        y_pred = l2_logistic_rg.predict(test_feat)
        classif_rate = np.mean(y_pred.ravel() == cur_test_label.ravel())*100
        print('classif_rate %f\n' % classif_rate)

if __name__ == '__main__':
    '''
    main
    '''
    fid = open('train_data.p', 'rb')
    train_data = pickle.load(fid)
    fid.close()
    np_train_label = train_data[0:len(train_data), 0]    
    np_train_feats = train_data[0:len(train_data), 1:]
    do_classification(np_train_feats, np_train_label)
