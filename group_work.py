import numpy as np
class ECDF:
    def get_ecdf(x):
        Z = np.sort(x.unique()) # Extract and sort unique values for x
        compare = x.to_numpy().reshape(-1,1) <= Z.reshape(1,-1) # Compare x and Z values
        ecdf = np.mean(compare,axis=0) # Average over x indices for each z

        return Z, ecdf

    def get_quantiles(x):
        median = np.quantile(x, 0.5)
        q25, q75 = np.quantile(x, [0.25, 0.75])

        return q25, q75

    def get_iqr_whiskers(x):
        q25, q75 = ECDF.get_quantiles(x)
        iqr = q75 - q25
        lower_bound = q25 - (1.5 * iqr)
        upper_bound = q75 + (1.5 * iqr)

        return iqr, lower_bound, upper_bound


def get_five_number_summary(x):
    minimum = np.min(x)
    q25, q75 = ECDF.get_quantiles(x)
    iqr = ECDF.get_iqr_whiskers(q25, q75)[0]
    lower_whisker = ECDF.get_iqr_whiskers(q25, q75)[1]
    upper_whisker = ECDF.get_iqr_whiskers(q25, q75)[2]
    median = np.median(x)
    maximum = np.max(x)

    return {
        "min:": print(minimum),
        "Q1:": print(q25), 
        "median": print(median), 
        "Q3": print(q75), 
        "upper_whisker": print(upper_whisker), 
        "max": print(maximum), 
        "iqr": print(iqr)
    }

def get_outliers(x):
    lower_bound = ECDF.get_iqr_whiskers(x)[1]
    upper_bound = ECDF.get_iqr_whiskers(x)[2]
    outliers = x[(x < lower_bound) | (x > upper_bound)]
    return outliers



    



