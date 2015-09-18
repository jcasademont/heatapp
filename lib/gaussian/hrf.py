import numpy as np
from .gbn import GBN

class HRF():
    def __init__(self, k, k_star, variables_names):
        self.k = k
        self.k_star = k_star
        self.bns = []
        self.variables_names = np.array(variables_names)

    def _corr_ratio(self, x1, x2):
        n = x1.shape[0]
        mu1 = np.mean(x1)
        mu2 = np.mean(x2)
        mu = 1/2 * (mu1 + mu2)

        num = n * (mu1 - mu) ** 2 + n * (mu2 - mu) ** 2
        den = np.sum(np.square(x1 - mu) + np.square(x2 - mu))

        return np.sqrt(num / den)

    def fit(self, X):
        n = len(self.variables_names)
        R = np.empty(len(self.variables_names), dtype=object)
        for i in range(n):
            ratios = np.array([self._corr_ratio(X[:, i], X[:, j])
                                for j in np.arange(n)])
            ratios[i] = np.inf
            ratios = np.argsort(ratios)[:self.k]
            ratios = np.array(list(filter(lambda x: x != i, ratios)))
            R[i] = np.array(self.variables_names[ratios], dtype=str)

        ite = 0
        refined = np.array([True] * len(self.variables_names))
        bns = np.empty(n, dtype=object)
        while np.any(refined):

            for i in range(n):
                if refined[i]:
                    other = [np.where(self.variables_names == n)[0][0] for n in R[i]]
                    bn = GBN(self.variables_names[np.append([i], other)])
                    bn.fit(X[:, np.append([i], other)])

                    bns[i] = bn
                    refined[i] = False

            u = np.empty(n, dtype=object)
            for i in range(n):
                u[i] = set()
                for j in range(n):
                    u[i] = u[i].union(bns[j].markov_blanket(self.variables_names[i]))

            for i in range(n):
                if(len(u[i]) <= self.k_star):
                    u_i = np.array(list(u[i]), dtype=str)
                    other = np.array([np.where(self.variables_names == n)[0][0] for n in u_i], dtype=int)
                    bn_p = GBN(self.variables_names[np.append([i], other)])
                    bn_p.fit(X[:, np.append([i], other)])

                    mb = bns[i].markov_blanket(self.variables_names[i])
                    mb_p = bn_p.markov_blanket(self.variables_names[i])

                    proba = bns[i].proba(self.variables_names[i],
                            data=X, given=mb)
                    proba_p = bn_p.proba(self.variables_names[i],
                            data=X, given=mb_p)
                    if(proba_p > proba):
                        R[i] = np.array(u_i, dtype=str)
                        refined[i] = True

            ite += 1

        self.bns = bns

    def predict(self, X, names):
        predictions = np.empty((X.shape[0], len(names)))

        for i, n in enumerate(self.variables_names):
            if n in names:
                idx = np.where(np.array(names) == n)[0][0]

                nodes_names = np.array(
                                list(filter(lambda x: x in names,
                                     self.bns[i].variables_names))
                                )
                indices = [np.where(self.variables_names == n)[0][0]
                           for n in self.bns[i].variables_names]
                node_idx = np.where(nodes_names == n)[0][0]
                preds = self.bns[i].predict(X[:, indices], nodes_names)
                predictions[:, idx] = preds[:, node_idx]

        return predictions

    def variances(self, names):
        variances = np.zeros(len(names))

        for i, n in enumerate(self.variables_names):
            if n in names:
                idx = np.where(np.array(names) == n)[0][0]
                var = self.bns[i].variances(n)
                variances[idx] = var

        return variances
