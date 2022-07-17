import minpy.numpy as np
import numpy as nnp
from minpy.context import gpu
from itertools import chain
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans


class PLSA(object):
    def __init__(self, docs, K=10, iters=10, min_delta=0.01):
        vocabulary = Counter(chain(*docs))
        vocab_idx = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
        vocab_idx = dict([(w, idx) for idx, (w, n) in enumerate(vocab_idx)])

        self._k = K
        self._iters = iters
        self._vocab_n = len(vocab_idx)
        self._n_docs = len(docs)
        self._min_delta = min_delta

        _docs = [[vocab_idx[w] for w in doc] for doc in docs]
        self._theta_m_k = PLSA.normalize(np.random.random([self._n_docs, self._k]))
        self._psi_k_j = PLSA.normalize(np.random.random([self._k, self._vocab_n]))
        self._Nm = np.asarray([len(doc) for doc in docs])
        self._p_dm = 1.0 / self._n_docs
        self._n_mj = pd.DataFrame([Counter(doc) for doc in _docs]).fillna(0).values
        self._vocab = dict([(idx, w) for w, idx in vocab_idx.items()])

    @staticmethod
    def normalize(vec):
        amax = np.sum(vec, axis=1)
        normalized = np.mat(vec) / np.mat(amax).T
        return np.asarray(normalized)

    @staticmethod
    def cross_entropy(predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
        return float(ce[0])

    def qz_mjk_m(self, mj, m):
        mmat = np.tile(self._theta_m_k[m, :], self._vocab_n)
        mmat = np.mat(mmat.reshape(self._vocab_n, self._k))
        p = np.multiply(mmat, np.mat(self._psi_k_j).T)
        p = self._p_dm * p
        mm = np.tile(mj[m, :], self._k).reshape(self._k, self._vocab_n)
        mm = np.transpose(mm)
        p /= mm
        return p

    def qz_mjk_k(self, mj, k):
        m = np.tile(np.transpose(self._theta_m_k[:, k]), self._vocab_n)
        j = np.repeat(self._psi_k_j[k, :], self._n_docs)
        tmj = np.multiply(m, j).reshape(self._vocab_n, self._n_docs)
        tmj = np.transpose(tmj)
        return tmj / mj

    def run(self):
        lase_te = 0.0
        lase_pe = 0.0
        te = 0.0
        pe = 0.0

        for i in range(self._iters):
            theta_mk = np.asarray([[0.0] * self._k for _ in range(self._n_docs)])
            psi_kj = np.asarray([[0.0] * self._vocab_n for _ in range(self._k)])
            mj = self._p_dm * np.matmul(self._theta_m_k, self._psi_k_j)
            for m in range(self._n_docs):
                mk = np.matmul(self._n_mj[m, :], self.qz_mjk_m(mj, m))
                p = mk / self._Nm[m]
                theta_mk[m] = np.asarray(p).reshape(self._k)

            den = [0.0] * self._k
            for k in range(self._k):
                p = np.multiply(self._n_mj, np.mat(self.qz_mjk_k(mj, k)))
                dk = np.sum(p)
                den[k] = dk

            for k in range(self._k):
                p = np.multiply(self._n_mj, self.qz_mjk_k(mj, k))
                p = np.sum(p, axis=0) / den[k]
                psi_kj[k] = p

            lase_pe = pe
            lase_te = te
            pe = PLSA.cross_entropy(psi_kj, self._psi_k_j)
            te = PLSA.cross_entropy(theta_mk, self._theta_m_k)

            if (abs(lase_pe - pe) < self._min_delta and abs(lase_te - te) < self._min_delta) or np.isnan(pe) or np.isnan(te):
                break

            self._psi_k_j = psi_kj
            self._theta_m_k = theta_mk
            yield i, te, pe



if __name__ == '__main__':
    docs = []

    with open('stopwords.txt', 'r', encoding='utf-8') as fin:
        stopwords = set([line.strip() for line in fin])

    n = 500
    with open('corpus.txt', 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split('\t')
            if len(line) != 4:
                continue

            words = line[3]
            words = [w for w in words.split(' ') if w not in stopwords]
            if len(words) < 10:
                continue

            docs.append(words)
            if n > 0:
                n -= 1
            else:
                break

    with gpu(0):
        plsa = PLSA(docs, 25, 5)
        for i,te, pe in plsa.run():
            print('iter %d, te: %f, pe: %f' % (i, te, pe))

        topic_words = np.argsort(plsa._psi_k_j)
        for t, tw in enumerate(topic_words):
            tw = tw[:20]
            words = [plsa._vocab[int(i)] for i in tw]
            print('topic %d: %s' % (t, ' '.join(words)))


