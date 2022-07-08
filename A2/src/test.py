import numpy as np

if __name__ == '__main__':
    df1 = np.array([2,2,3])
    df2 = np.array([[1,2,3],[1,2,3]])

    df3 = df1* df2
    df3 = df3.sum(axis=0)
    df3 = np.zeros(df2.shape[1])
    df3 = np.concatenate((np.atleast_2d(df3),np.atleast_2d(df1)), axis=0)
    df3 = np.concatenate((df3,df2))
    df3 = np.delete(df3,0,0)
    #df4 = np.dot(df1,df2)

    print(df1)
    print(df2)
    print(df3)