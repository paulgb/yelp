
from numpy import array, zeros
from collections import defaultdict

class CategoryAverage:
    def fit(self, x, y):
        x = array(x)
        y = array(y)
        self.tot = (0,0)
        self.vs = dict()
        for est, val in zip(x, y):
            try:
                c, s = self.vs[est]
            except KeyError:
                c, s = (0, 0)
            c += 1
            s += val
            self.vs[est] = c, s
            self.tot = (self.tot[0] + 1, self.tot[1] + val)
        return self
    
    def transform(self, x):
        x = array(x)
        l = zeros((len(x), 1))
        for i, h in enumerate(x):
            t_c, t_s = self.tot
            overall_avg = float(t_s) / t_c
            if h in self.vs:
                c, s = self.vs[h]
                avg = float(s) / c
            else:
                avg = 0
                c = 0
            slce = 1. / (c+1)
            l[i, 0] = (slce * overall_avg) + ((1 - slce) * avg)
        return l

