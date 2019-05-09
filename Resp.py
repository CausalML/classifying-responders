import numpy as np
from sklearn import linear_model, svm, model_selection, metrics
from scipy.special import logit

## Assume RCT data everywhere (Q=1/2)
## All binary variables are boolean vectors (not +-1)
## All objects return decision function; for some this is a logit of responder probability

class Object(object):
    pass

def RespLoss(z_true, z_pred, theta=0.5):
    return (2*(1-theta)*((z_true) & (~z_pred)) + 2*(1+theta)*((~z_true) & (z_pred))).mean()

class defaultdictkeyed(dict):
    def __init__(self, f):
        super(defaultdictkeyed, self).__init__()
        self.f = f
    def __missing__(self, k):
        v = self.f(k)
        self[k] = v
        return v
    def __call__(self, k):
        return self[k]

RespLoss_scorer = defaultdictkeyed(
    lambda theta:
        metrics.make_scorer(
            lambda z_true, z_pred: RespLoss(z_true, z_pred, theta),
            greater_is_better=False)
)

class RespClassifier:
    def __init__(self, model, theta=0.5, params={}, cv=None, cvparams={}, weighting='class', usedecisionfunction=True):
        self.model = model
        self.theta = theta
        self.params = params
        self.cv = cv
        self.cvparams = cvparams
        self.usedecisionfunction = usedecisionfunction
        self.weighting = weighting
    def fit(self, X, T, Y):
        params = self.params
        Z = T==Y
        if Z.all():
            self.fit0 = Object()
            self.fit0.decision_function = lambda x: np.inf*np.ones(len(x))
            self.fit0.predict_proba     = lambda x: np.vstack((np.zeros(len(x)),np.ones(len(x)))).T
        elif (~Z).all():
            self.fit0 = Object()
            self.fit0.decision_function = lambda x: -np.inf*np.ones(len(x))
            self.fit0.predict_proba     = lambda x: np.vstack((np.ones(len(x)),np.zeros(len(x)))).T
        else:
            if self.weighting == 'class':
                params['class_weight'] = {False:2*(1+self.theta), True:2*(1-self.theta)}
                fitparams = {}
            elif self.weighting == 'sample':
                fitparams = {'sample_weight': 2*(1-self.theta)*Z + 2*(1+self.theta)*(~Z)}
            else:
                raise Exception("Weighting type not implemented")
            self.fit0 = (
                    self.model(**params)
                if self.cv is None else
                    model_selection.GridSearchCV(self.model(**params), self.cvparams, scoring=RespLoss_scorer(self.theta), cv=self.cv, iid=False, error_score=np.nan)
                ).fit(X,Z,**fitparams)
    def predict(self, x):
        return self.fit0.decision_function(x) if self.usedecisionfunction else logit(self.fit0.predict_proba(x)[:,1])

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
_EPSILON = K.epsilon()

def layer_init(layer):
    w0 = layer.get_weights()
    fan_in, fan_out = w0[0].shape
    limit = np.sqrt(6. / (fan_in + fan_out))
    layer.set_weights([
        (np.random.rand(*w0[0].shape)*2.-1.)*limit,
        np.zeros_like(w0[1])
    ])

def RespLoss_keras_helper(theta):
    def loss(zbTrue,rPred):
        rPred = K.round(rPred)
        return K.mean(2*(1-theta)*zbTrue * (1.0-rPred) + 2*(1+theta)*(1.0 - zbTrue) *  rPred)
    return loss

def RespEntropy_keras_helper(theta):
    def loss(zbTrue,rPred):
        rPred = K.clip(rPred, _EPSILON, 1.0-_EPSILON)
        return -K.sum(2*(1-theta)*(zbTrue) * K.log(rPred) + 2*(1+theta)*(1.0 - zbTrue) * K.log(1.0 - rPred))
    return loss

RespLoss_keras = defaultdictkeyed(RespLoss_keras_helper)

RespEntropy_keras = defaultdictkeyed(RespEntropy_keras_helper)

def RespNLL_keras(zbTrue,rPred):
    rPred = K.clip(rPred, _EPSILON, 1.0-_EPSILON)
    return -K.sum(K.log(0.5+(1.0*zbTrue-0.5)*rPred))

class RespNet:
    def __init__(self, input_dim, hiddenlayers=(), loss='generative', optimizer='adam', epochs=200, verbose=0):
        self.model = keras.Sequential([
            keras.layers.Dense(h, activation=tf.nn.sigmoid if i==len(hiddenlayers) else tf.nn.elu, **({'input_dim': input_dim} if i==0 else {}))
        for i,h in enumerate(hiddenlayers+(1,))])
        self.epochs = epochs
        self.verbose = verbose
        self.model.compile(
            optimizer = optimizer,
            loss = RespNLL_keras if loss=='generative' else RespEntropy_keras(loss)
        )
    def fit(self, X, T, Y, reset_layers = True, reset_optimizer = True, epochs = None, verbose = None):
        if reset_layers:
            for layer in self.model.layers:
                layer_init(layer)
        if reset_optimizer:
            self.model.optimizer.set_weights([0*x for x in self.model.optimizer.get_weights()])
        self.model.fit(X, T==Y, epochs=self.epochs if epochs is None else epochs, verbose=self.verbose if verbose is None else verbose)
    def predict(self, x):
        return logit(self.model.predict(x)[:,0])

from itertools import chain

class TARNet:
    def __init__(self, input_dim, hiddenlayers0, hiddenlayers1, optimizer='adam', epochs=200, verbose=0):
        self.epochs = epochs
        self.verbose = verbose
        inputX = keras.layers.Input(shape=(input_dim,))
        inputT = keras.layers.Input(shape=(1,))
        self.phi = keras.Sequential([
            keras.layers.Dense(h, activation=tf.nn.elu)
            for h in hiddenlayers0])
        phi = self.phi(inputX)
        self.f0 = keras.Sequential([
            keras.layers.Dense(h, activation=tf.nn.elu)
            for h in hiddenlayers1] + [keras.layers.Dense(1, activation=tf.nn.sigmoid),])
        f0 = self.f0(phi)
        self.f1 = keras.Sequential([
            keras.layers.Dense(h, activation=tf.nn.elu)
            for h in hiddenlayers1] + [keras.layers.Dense(1, activation=tf.nn.sigmoid),])
        f1 = self.f1(phi)
        f0t = keras.layers.Multiply()([f0,keras.layers.Lambda(lambda t: 1.-t)(inputT)])
        f1t = keras.layers.Multiply()([f1,inputT])
        out = keras.layers.Add()([f0t,f1t])
        self.model = keras.models.Model(inputs=[inputX, inputT], outputs=out)
        self.model.compile(
            optimizer = optimizer,
            loss = 'binary_crossentropy',
        )
    def fit(self, X, T, Y, reset_layers = True, reset_optimizer = True, epochs = None, verbose = None):
        if reset_layers:
            for layer in chain(self.phi.layers, self.f0.layers, self.f1.layers):
                layer_init(layer)
        if reset_optimizer:
            self.model.optimizer.set_weights([0*x for x in self.model.optimizer.get_weights()])
        self.model.fit([X,T], Y, epochs=self.epochs if epochs is None else epochs, verbose=self.verbose if verbose is None else verbose)
    def predict(self, x):
        return logit(
            np.clip(
                (self.model.predict([x,np.ones(x.shape[0])])-self.model.predict([x,np.zeros(x.shape[0])]))[:,0]
            ,0.,1.)
        )

import rpy2
import rpy2.robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.vectors
R=rpy2.robjects.r
R.library("grf")

class CF:
    def __init__(self, params={}):
        self.params = params
    def fit(self, X, T, Y):
        self.cf = R.causal_forest(X, Y, T,**self.params)
    def predict(self, x):
        return logit(np.clip(np.array(R.predict(self.cf,x)).ravel(),0.,1.))

class DiffCATE:
    def __init__(self, model, params={}, cv=None, cvparams={}):
        self.model = model
        self.params = params
        self.cv = cv
        self.cvparams = cvparams
    def fit(self, X, T, Y):
        if Y[~T].all():
            self.fit0 = Object()
            self.fit0.decision_function = lambda x: np.inf*np.ones(len(x))
            self.fit0.predict_proba     = lambda x: np.vstack((np.zeros(len(x)),np.ones(len(x)))).T
        elif (~Y[~T]).all():
            self.fit0 = Object()
            self.fit0.decision_function = lambda x: -np.inf*np.ones(len(x))
            self.fit0.predict_proba     = lambda x: np.vstack((np.ones(len(x)),np.zeros(len(x)))).T
        else:
            self.fit0 = (
                self.model(**self.params)
                if self.cv is None else
                model_selection.GridSearchCV(self.model(**self.params), self.cvparams, cv=self.cv, iid=False, error_score=np.nan)
                ).fit(X[~T],Y[~T])
        if Y[T].all():
            self.fit1 = Object()
            self.fit1.decision_function = lambda x: np.inf*np.ones(len(x))
            self.fit1.predict_proba     = lambda x: np.vstack((np.zeros(len(x)),np.ones(len(x)))).T
        elif (~Y[T]).all():
            self.fit1 = Object()
            self.fit1.decision_function = lambda x: -np.inf*np.ones(len(x))
            self.fit1.predict_proba     = lambda x: np.vstack((np.ones(len(x)),np.zeros(len(x)))).T
        else:
            self.fit1 = (
                self.model(**self.params)
                if self.cv is None else
                model_selection.GridSearchCV(self.model(**self.params), self.cvparams, cv=self.cv, iid=False, error_score=np.nan)
                ).fit(X[T],Y[T])
    def predict(self, x):
        return logit(np.clip((self.fit1.predict_proba(x)-self.fit0.predict_proba(x))[:,1],0.,1.))
