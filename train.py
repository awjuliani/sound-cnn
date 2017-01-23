import tensorflow as tf
import scipy.io.wavfile
import numpy as np
import matplotlib.mlab
from os import listdir
from os.path import isfile, join
import sys
import utilities as util

from model import SoundCNN

arguments = sys.argv

bpm = int(arguments[1])
samplingRate = int(arguments[2])
mypath = str(arguments[3])
iterations = int(arguments[4])
batchSize = int(arguments[5])

classes,trainX,trainYa,valX,valY,testX,testY = util.processAudio(bpm,samplingRate,mypath)

def trainNetConv(maxIter):

	myModel = SoundCNN(classes)
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())
		myIters = 0
		fullTrain = np.concatenate((trainX,trainYa),axis=1)
		while myIters < maxIter:
			perms = np.random.permutation(fullTrain)
			for i in range(perms.shape[0]/batchSize):
				batch = perms[i *batchSize:(i+1) * batchSize,:]
				batchX,batchYa = np.hsplit(batch,[-1])
				batchY = util.oneHotIt(batchYa)
				sess.run(myModel.train_step,feed_dict={myModel.x: batchX, myModel.y_: batchY, myModel.keep_prob: 0.5})
				if myIters%100 == 0:
					train_accuracy = myModel.accuracy.eval(session=sess,feed_dict={myModel.x:batchX, myModel.y_: batchY, myModel.keep_prob: 1.0})
					print("Step %d, Training accuracy: %g"%(myIters, train_accuracy))
				if myIters%500 == 0:
					val_accuracy = myModel.accuracy.eval(session=sess,feed_dict={myModel.x:valX, myModel.y_: valY, myModel.keep_prob: 1.0})
					print("Step %d, Validation accuracy: %g"%(myIters, val_accuracy))
				myIters+= 1
		test_accuracy = myModel.accuracy.eval(session=sess,feed_dict={myModel.x:testX, myModel.y_: testY, myModel.keep_prob: 1.0})
		print("Test accuracy: %g"%(test_accuracy))
		save_path = saver.save(sess, "./model.ckpt")

trainNetConv(iterations)

