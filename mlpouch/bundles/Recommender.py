import numpy as np

class Recommender:
    
    def __init__(self, recommenders):
        self.recommenders = recommenders

    def recommend(self, U, rec_num=5):
        results = []

        for U_item in U:
            results.append(self._recommend(U_item, rec_num))

        return results
        

    def _recommend(self, U_item, rec_num=5):

        r = [U_item, np.NaN]

        for rec in self.recommenders:
            result = rec.recommend([U_item], rec_num)

            if result[0][1] != np.NaN:
                r = result[0][1]
                break

        return r