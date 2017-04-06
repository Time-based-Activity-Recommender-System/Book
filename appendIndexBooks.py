import numpy as np
import pandas as pd

def main():
    c_cols = ['2user_id','3ISBN','4bookrating']
    current_user_data = pd.read_csv('book-dataset/BX-Book-Ratings.csv', sep=';', names=c_cols,encoding='latin-1') 
    print current_user_data['2user_id'][0]
    
    d = {'1slno': [0],  '2user_id': current_user_data['2user_id'][0], '3ISBN': current_user_data['3ISBN'][0], '4bookrating':current_user_data['4bookrating'][0]}
    df = pd.DataFrame(d)
    df.to_csv('book-dataset/book_ratings.txt',sep=';',index=False, header=False)

    for i in range(1,len(current_user_data)):
        d = {'1slno': [i],  '2user_id': current_user_data['2user_id'][i], '3ISBN': current_user_data['3ISBN'][i], '4bookrating':current_user_data['4bookrating'][i]}
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/book_ratings.txt',mode='a' ,sep=';',index=False, header=False)     


if __name__ == "__main__":
    main()
