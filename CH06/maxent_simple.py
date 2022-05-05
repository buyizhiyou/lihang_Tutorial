from collections import defaultdict
import math
#https://vimsky.com/article/714.html
#https://vimsky.com/article/776.html

class MaxEnt(object):

    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = []
        self.labels = set()

    def load_data(self, file):
        import pdb;pdb.set_trace()
        for line in open(file):
            fields = line.strip().split()

            # at least two columns
            if len(fields) < 2:
                continue
            # the first column is label
            label = fields[0]
            self.labels.add(label)

            for f in set(fields[1:]):
                # (label,f) tuple is feature
                self.feats[(label, f)] += 1
            self.trainset.append(fields)
        
        print(self.trainset)

    def _initparams(self):

        self.size = len(self.trainset)#15
        # M param for GIS training algorithm
        self.M = max([len(record) - 1 for record in self.trainset])#3
        self.ep_ = [0.0] * len(self.feats)#len:12个特征函数

        for i, f in enumerate(self.feats):
            # calculate feature expectation on empirical distribution，经验分布
            # E_p*(f_i)
            self.ep_[i] = float(self.feats[f]) / float(self.size)
            # each feature function correspond to id
            self.feats[f] = i

        # init weight for each feature
        self.w = [0.0] * len(self.feats)

        self.lastw = self.w

    def probwgt(self, features, label):
        wgt = 0.0
        #计算sum_i(w_i*f_i(x,y))
        for f in features:
            if (label, f) in self.feats:
                wgt += self.w[self.feats[(label, f)]]
  
        #exp(sum_i(w_i*f_i(x,y)))
        return math.exp(wgt)

    """
    calculate feature expectation on model distribution,计算关于模型分布的期望
    """
    def Ep(self):
        ep = [0.0] * len(self.feats)
        for record in self.trainset:
            features = record[1:]
            # calculate p(y|x)
            prob = self.calprob(features)#[(0.5, 'Outdoor'), (0.5, 'Indoor')]

            for f in features:
                for w, l in prob:
                    # only focus on features from training data.
                    if (l, f) in self.feats:
                        # get feature id
                        idx = self.feats[(l, f)]
                        # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
                        ep[idx] += w * (1.0 / self.size)

        #E_P(f_i)
        return ep

    def _convergence(self, lastw, w):

        for w1, w2 in zip(lastw, w):
            if abs(w1 - w2) >= 0.01:
                return False
        return True

    def train(self, max_iter=1000):
        import pdb;pdb.set_trace()
        self._initparams()
        for i in range(max_iter):
            print('iter %d ...' % (i + 1))
            # calculate feature expectation on model distribution
            self.ep = self.Ep()
            self.lastw = self.w[:]
            for i, w in enumerate(self.w):
                #GIS算法更新公式
                delta = 1.0 / self.M * math.log(self.ep_[i] / self.ep[i])
                # update w
                self.w[i] += delta

            print(self.w)
            # test if the algorithm is convergence
            if self._convergence(self.lastw, self.w):
                break

    def calprob(self, features):

        wgts = [(self.probwgt(features, l), l) for l in self.labels]#[(1.0, 'Outdoor'), (1.0, 'Indoor')]
        Z = sum([w for w, l in wgts])#计算Z_w=sum_y{sum_i[w_i*f_i(x,y)]}
        prob = [(w / Z, l) for w, l in wgts]#p(y|x)

        return prob

    def predict(self, input):

        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse=True)

        return prob


model = MaxEnt()
'''
Outdoor Sunny Happy
Outdoor Sunny Happy Dry
Outdoor Sunny Happy Humid
Outdoor Sunny Sad Dry
Outdoor Sunny Sad Humid
Outdoor Cloudy Happy Humid
Outdoor Cloudy Happy Humid
Outdoor Cloudy Sad Humid
Outdoor Cloudy Sad Humid
Indoor Rainy Happy Humid
Indoor Rainy Happy Dry
Indoor Rainy Sad Dry
Indoor Rainy Sad Humid
Indoor Cloudy Sad Humid
Indoor Cloudy Sad Humid
'''
#label={outdoor,indoor}
#words={Sunny,Happy,Dry,Humid,Sad,Cloudy,Rainy}
#context:15条

model.load_data('Input/gameLocation.dat')
model.train()
