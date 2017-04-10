from bookRatingsUI import Ui_MainWindow
import sys
from PyQt4 import QtGui,QtCore
import os
import signal
import numpy as np
import pandas as pd
from scipy import optimize

num_movies = 1682
num_users = 943 #updated by temp.data

class BookRatings(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initUI()

	#to get newuser_id
        c_cols = ['current_user']
        current_user_data = pd.read_csv('session.data', sep='\t', names=c_cols, encoding='latin-1') 
        name = current_user_data['current_user'][0]
        
        p_cols = ['1user_id', '2Password', '3user_id'] #first user_id is user name, 3rd column is system generated
        passwords_data = pd.read_csv('passwords.data', sep='\t', names=p_cols, encoding='latin-1')
        for i in range(len(passwords_data)):
            if( passwords_data['1user_id'][i] == name ):
                self.newuser_id = passwords_data['3user_id'][i]
                break

	print "newuser_id=",self.newuser_id 

        self.ui.save_next_pushButton.clicked.connect(self.back)
        self.connections()
	self.books()
        
    def initUI(self):
        self.setWindowTitle('Login')
        self.center()
        self.show()

    def center(self):
        frameGm = self.frameGeometry()
        centerPoint = QtGui.QDesktopWidget().availableGeometry().center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def back(self):
        self.hide()
        os.system('python rate_UI_run.py')
    
    def appendNewRatings(self):
        r_cols = ['1user_id', '2ISBN', '3rating']
        self.ratings_data = pd.read_csv('book-dataset/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1')
	#print ratings_data['2ISBN'][613610]
        d = {'1user_id': [self.newuser_id],  '2ISBN': [16], '3rating': self.newuser_ratings[16]}
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [39], '3rating': self.newuser_ratings[39] }
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [253], '3rating': self.newuser_ratings[253]}
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [43], '3rating': self.newuser_ratings[43] }
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [283], '3rating': self.newuser_ratings[283]}
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [750], '3rating': self.newuser_ratings[750]}
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [135], '3rating': self.newuser_ratings[135]}
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [34400], '3rating': self.newuser_ratings[34400] }
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [105711], '3rating': self.newuser_ratings[105711]}
        df = pd.DataFrame(d)
        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 

        d = {'1user_id': [self.newuser_id],  '2ISBN': [107290], '3rating': self.newuser_ratings[107290]}
        df = pd.DataFrame(d)

        df.to_csv('book-dataset/BX-Book-Ratings.csv',mode='a' ,sep=';',index=False, header=False) 
        self.ratings_data = pd.read_csv('book-dataset/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1')
 
    def back(self):    
        self.appendNewRatings()
        #self.recommenderSystem()

        self.hide()
        os.system('python rate_UI_run.py')

    def connections(self):
        self.connect(self.ui.horizontalSlider_1,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_2,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_3,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_4,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_5,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_6,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_7,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_8,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal)
        self.connect(self.ui.horizontalSlider_9,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal) 
        self.connect(self.ui.horizontalSlider_10,QtCore.SIGNAL("valueChanged(int)"),self.sliderVal) 

    def sliderVal(self):
        self.ui.lineEdit_11.setText(str(self.ui.horizontalSlider_1.value()))
        self.ui.lineEdit_12.setText(str(self.ui.horizontalSlider_2.value()))
        self.ui.lineEdit_13.setText(str(self.ui.horizontalSlider_3.value()))
        self.ui.lineEdit_14.setText(str(self.ui.horizontalSlider_4.value()))
        self.ui.lineEdit_15.setText(str(self.ui.horizontalSlider_5.value()))
        self.ui.lineEdit_16.setText(str(self.ui.horizontalSlider_6.value()))
        self.ui.lineEdit_17.setText(str(self.ui.horizontalSlider_7.value()))
        self.ui.lineEdit_18.setText(str(self.ui.horizontalSlider_8.value()))
        self.ui.lineEdit_19.setText(str(self.ui.horizontalSlider_9.value()))
        self.ui.lineEdit_20.setText(str(self.ui.horizontalSlider_10.value()))
        self.newRatings()
    
    def books(self):
	print "asmita"
        i_cols = ['1ISBN','2Book','3Author','4Year','5Publisher','6S', '7M', '8L']
        items = pd.read_csv('book-dataset/BX-Books.csv', sep=';', names=i_cols, dtype = 'unicode')
	print 'boo'
	print items['2Book'][5]
	'''        
	self.ui.lineEdit_1.setText(items['2Book'][16]) #16
        self.ui.lineEdit_2.setText(items['2Book'][39]) #39
        self.ui.lineEdit_3.setText(items['2Book'][253]) #253
        self.ui.lineEdit_4.setText(items['2Book'][43])
        self.ui.lineEdit_5.setText(items['2Book'][283]) #283
        self.ui.lineEdit_6.setText(items['2Book'][750]) #750
        self.ui.lineEdit_7.setText(items['2Book'][135]) 
        self.ui.lineEdit_8.setText(items['2Book'][34400])
        self.ui.lineEdit_9.setText(items['2Book'][105711]) #105711
        self.ui.lineEdit_10.setText(items['2Book'][107290]) #107290
	'''
    
    def newRatings(self):
        global num_movies
        self.newuser_ratings = np.zeros((num_movies, 1))
        self.newuser_ratings[16] = int(self.ui.horizontalSlider_1.value())
        self.newuser_ratings[39] = int(self.ui.horizontalSlider_10.value())
        self.newuser_ratings[253] = int(self.ui.horizontalSlider_2.value())
        self.newuser_ratings[43] = int(self.ui.horizontalSlider_3.value())   
        self.newuser_ratings[283] = int(self.ui.horizontalSlider_4.value())
        self.newuser_ratings[750] = int(self.ui.horizontalSlider_5.value())
        self.newuser_ratings[135] = int(self.ui.horizontalSlider_6.value())
        self.newuser_ratings[34400] =int(self.ui.horizontalSlider_7.value())
        self.newuser_ratings[105711] = int(self.ui.horizontalSlider_8.value())
        self.newuser_ratings[107290] = int(self.ui.horizontalSlider_9.value())
        
        
    def recommenderSystem(self):
        global num_movies
        global num_users

        #update num_users
        cols = ['count']
        count_data = pd.read_csv('temp.data', sep=';', names=cols, encoding='latin-1') 
        num_users = count_data['count'][0] - 1
        print "num_users=",num_users

        self.ratings = np.zeros((num_movies, num_users), dtype = np.uint8) #num_users updated 
        #Create 2D ratings matrix
        for i in range(len(self.ratings_data)):
	        col = (int)(self.ratings_data['1user_id'][i])-1
	        row = (int)(self.ratings_data['2ISBN'][i])-1
	        self.ratings[row][col]=(int)(self.ratings_data['3rating'][i])
        
        self.did_rate = (self.ratings != 0) * 1

        self.ratings, ratings_mean = self.normalize_ratings()
        num_users = self.ratings.shape[1] #num_users gets updated i.e. increases by 1
        num_features = 3

        movie_features = np.random.randn( num_movies, num_features )
        user_prefs = np.random.randn( num_users, num_features )
        initial_X_and_theta = np.r_[movie_features.T.flatten(), user_prefs.T.flatten()]

        reg_param = 30
        minimized_cost_and_optimal_params = optimize.fmin_cg(self.calculate_cost, fprime=self.calculate_gradient, x0=initial_X_and_theta, args=(self.ratings, self.did_rate, num_users, num_movies, num_features, reg_param), maxiter=100, disp=True, full_output=True ) 
        cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]

        movie_features, user_prefs = self.unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)
        # Make some predictions (movie recommendations). Dot product
	all_predictions = movie_features.dot( user_prefs.T )
        # add back the ratings_mean column vector to my (our) predictions
        predictions_for_newuser = all_predictions[:, 0:1] + ratings_mean
        
        i_cols = ['1ISBN', '2bookname']
        items = pd.read_csv('book-dataset/BX-Books.csv', sep=';', names=i_cols,encoding='latin-1')
        ind = np.argpartition(predictions_for_newuser, -1)[-5:]
        for i in range(len(ind)):
            ind2 = self.ratings_data['1ISBN'][i]
            #print items['movie title'][ind2]
            d = { 'Bookname': [ items['2book_name'][ind2] ] }
            df = pd.DataFrame(d)
	    df.to_csv('book_reco.data',mode='a' ,sep=';',index=False, header=False)
       
    def normalize_ratings(self):
	global num_movies
        num_movies = self.ratings.shape[0]
        
        ratings_mean = np.zeros(shape = (num_movies, 1))
        ratings_norm = np.zeros(shape = self.ratings.shape)
        
        for i in range(num_movies):
            # Get all the indexes where there is a 1
            idx = np.where(self.did_rate[i] == 1)[0]
            #  Calculate mean rating of ith movie only from user's that gave a rating
            ratings_mean[i] = np.mean(self.ratings[i, idx])
            ratings_norm[i, idx] = self.ratings[i, idx] - ratings_mean[i]
            
        return ratings_norm, ratings_mean

    def unroll_params(self, X_and_theta, num_users, num_movies, num_features):
	    # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	    # --------------------------------------------------------------------------------------------------------------
	    # Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	    first_30 = X_and_theta[:num_movies * num_features]
	    # Reshape this column vector into a 10 X 3 matrix
	    X = first_30.reshape((num_features, num_movies)).transpose()
	    # Get the rest of the 18 the numbers, after the first 30
	    last_18 = X_and_theta[num_movies * num_features:]
	    # Reshape this column vector into a 6 X 3 matrix
	    theta = last_18.reshape(num_features, num_users ).transpose()
	    return X, theta

    def calculate_gradient(self, X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	    X, theta = self.unroll_params(X_and_theta, num_users, num_movies, num_features)

	    # we multiply by did_rate because we only want to consider observations for which a rating was given
	    difference = X.dot( theta.T ) * did_rate - ratings
	    X_grad = difference.dot( theta ) + reg_param * X
	    theta_grad = difference.T.dot( X ) + reg_param * theta

	    # wrap the gradients back into a column vector 
	    return np.r_[X_grad.T.flatten(), theta_grad.T.flatten()]
    
    def calculate_cost(self, X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	    X, theta = self.unroll_params(X_and_theta, num_users, num_movies, num_features)
	    # we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	    cost = np.sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	    # '**' means an element-wise power
	    regularization = (reg_param / 2) * (np.sum( theta**2 ) + np.sum(X**2))
	    return cost + regularization

    


def main():
    app=QtGui.QApplication(sys.argv) 
    ui = BookRatings() 
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
