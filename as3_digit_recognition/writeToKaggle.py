import csv
import os
def write_tokaggle(fname,pred):
	dir = os.path.dirname(fname)
	if not os.path.exists(dir):
		os.makedirs(dir) 
	test_output_file = open(fname, "wb")
	writer =csv.writer(test_output_file, delimiter = ',')
	writer.writerow(['ID','Prediction'])
	for i,p in enumerate(pred):
	    row = [i+1, p]
	    writer.writerow(row)
	test_output_file.close()