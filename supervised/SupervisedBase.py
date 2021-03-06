import numpy as np

class SupervisedBaseClass():

    def _format_batch(self, *args):
        """
        convert list to np.ndarray, and get rid of the redundant dimension
        """
        result = [np.array(item) for item in args]
        if len(result) == 1:
            return result[0]
        return result
    
    def _sign(self, y, thresh=0):
        y[y>=thresh] = 1
        y[y<thresh] = 0
        return y
    
    def _batch_generator(self, X, y, batch_size):
        n_samples = X.shape[0]
        rand_index = np.arange(n_samples)
        np.random.shuffle(rand_index)
        s = 0
        while s < n_samples:
            t = min(n_samples, s+batch_size)
            yield X[rand_index[s:t]], y[rand_index[s:t]]
            s += batch_size

    def _error_calc(self, pred, y):
        '''
        defined by sub-class
        '''
        raise NotImplementedError

    def _update(self, X, y, err):
        '''
        defined by sub-class
        '''
        raise NotImplementedError

    def _optimize_gd(self, X, y, batch_size, epoch):
        '''
        gradient descent optimization
        '''
        for _ in range(epoch):
            next_batch = self._batch_generator(X, y, batch_size)
            while True:
                try:
                    batch_x, batch_y = next(next_batch)
                except StopIteration:
                    break
                pred = self._predict(batch_x)
                err = self._error_calc(pred, batch_y)
                self._update(batch_x, batch_y, err)

    def _predict(self, X):
        """
        output prediction
        defined by sub-class
        """
        raise NotImplementedError
    
    def predict(self, X):
        """
        output prediction
        defined by sub-class
        """
        raise NotImplementedError

    def _cls_score(self, X, y, output=True):
        """
        calculate the accurary of the model
        """
        _y = self.predict(X)
        y = np.array(y); _y = np.array(_y)
        precious = 1-(1.0*len(np.nonzero(y-_y)[0])/len(y))
        if output:
            print(f"Accuracy: {precious}")
        return precious
    
    def _reg_score(self, X, y, output=True):
        """
        calculate the accurary of the model
        """
        X, y = self._format_batch(X, y)
        _y = self.predict(X)
        mse = np.sum((_y-y)**2)/len(y)
        if output:
            print(f"Mean Squared Error: {mse}")
        return mse
