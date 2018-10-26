import numpy as np

class SupervisedBaseClass():

    def _format_batch(self, *data):
        """
        convert list to np.ndarray, and get rid of the redundant dimension
        """
        result = []
        for item in data:
            if isinstance(item, list):
                item = np.array(item)
            if isinstance(item, np.ndarray) and item.ndim > 1:
                shape = item.shape
                while shape[0] == 1:
                    item.reshape(shape[1:])
                    shape = item.shape
            result.append(item)
        if len(result) == 1:
            return result[0]
        return result

    def _error_calc(self, pred, y):
        '''
        defined by sub-class
        '''
        pass

    def _update(self, X, pred, y, err):
        '''
        defined by sub-class
        '''
        pass

    def _optimize_gd(self, X, y, max_iter, tol):
        error = 0
        for t in xrange(max_iter):
            pred = self.predict(X)
            err = self._error_calc(pred, y)
            error = err[0]
            self._update(X, pred, y, err)
            if error < tol:  break

    def predict(self, X):
        """
        output list
        defined by sub-class
        """
        pass

    def eval(self, X, y, output=True):
        """
        calculate the accurary of the model
        """
        _y = self.predict(X)
        y = np.array(y); _y = np.array(_y)
        precious = 1-(1.0*len(np.nonzero(y-_y)[0])/len(y))
        if output:
            print "Accurary: %.2f" % precious
        return precious
