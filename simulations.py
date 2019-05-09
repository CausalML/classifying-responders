import numpy as np
import scipy
import scipy.stats
import Resp
from sklearn import svm, ensemble, metrics, linear_model
from collections import OrderedDict

# Linear
def makedata1(n,d):
    X = np.random.randn(n,d)
    probR = 0.15+0.7*(X.sum(axis=1)>0)
    R = np.random.rand(n)<=probR
    probNonRespType = 1.-scipy.stats.beta.cdf(scipy.stats.chi.cdf(np.linalg.norm(X,axis=1),d),4.,4.)
    NonRespType = np.random.rand(n)<=probNonRespType
    Tb = np.random.randint(0,2,n).astype('bool')
    Yb = R*Tb | ((~R)&NonRespType)
    return X,R,probR,Tb,Yb

# Spherical
def makedata2(n,d):
    X = np.random.randn(n,d)
    probR = scipy.stats.beta.cdf(scipy.stats.chi.cdf(np.linalg.norm(X,axis=1),d),4.,4.)
    R = np.random.rand(n)<=probR
    probNonRespType = 0.15+0.7*np.bitwise_xor.reduce(X[:,::2]+X[:,1::2]>0,axis=1)
    NonRespType = np.random.rand(n)<=probNonRespType
    Tb = np.random.randint(0,2,n).astype('bool')
    Yb = R*Tb | ((~R)&NonRespType)
    return X,R,probR,Tb,Yb

scenarios = OrderedDict((('linear',makedata1), ('spherical',makedata2)))
ntest     = 10000

ds = [2,10,20]

methods = OrderedDict((

('Linear_SVC+CV',
Resp.RespClassifier(svm.LinearSVC,cv=5,cvparams={'C':np.logspace(-3,3,7)})
),

('RBF_SVC+CV',
{d:
Resp.RespClassifier(svm.SVC,cv=5,cvparams={'C':np.logspace(-3,3,7), 'gamma': (1. / d)*np.logspace(-2,2,9)})
for d in ds}
),

('RespLR_gen',
{d: 
Resp.RespNet(input_dim=d)
for d in ds}
),

('RespLR_disc',
{d: 
Resp.RespNet(input_dim=d, loss=0.5)
for d in ds}
),

('RespNet_gen',
{d: 
Resp.RespNet(input_dim=d, hiddenlayers=(2*d,d))
for d in ds}
),

('RespNet_disc',
{d: 
Resp.RespNet(input_dim=d, hiddenlayers=(2*d,d), loss=0.5)
for d in ds}
),

('TARNet',
{d: 
Resp.TARNet(input_dim=d, hiddenlayers0=(2*d,), hiddenlayers1=(d,))
for d in ds}
),

('RF',
Resp.DiffCATE(ensemble.RandomForestClassifier,params={'n_estimators':100})
),

('CF',
Resp.CF()
),

))

import random

# e.g. doRun('spherical', 20, 10240, 0, 'RBF_SVC+CV')
def doRun(scenario, d, n, seed, method):
    np.random.seed(seed)
    random.seed(seed)
    X_test,R_test,probR_test,Tb_test,Yb_test = scenarios[scenario](ntest,d)
    X,R,probR,Tb,Yb = scenarios[scenario](n,d)
    m = methods[method][d] if type(methods[method]) is dict else methods[method]
    m.fit(X,Tb,Yb)
    Rhat = m.predict(X)>0
    Rhat_test = m.predict(X_test)>0
    acc1 = metrics.accuracy_score(R, Rhat)
    acc2 = metrics.accuracy_score(probR>0.5, Rhat)
    acc1_test = metrics.accuracy_score(R_test, Rhat_test)
    acc2_test = metrics.accuracy_score(probR_test>0.5, Rhat_test)
    result = (scenario, d, n, seed, method, acc1, acc2, acc1_test, acc2_test)
    return result
