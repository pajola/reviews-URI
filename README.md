# A Novel Review Helpfulness Measure based on the User-Review-Item Paradigm
Official repository for the paper "A Novel Review Helpfulness Measure based on the User-Review-Item Paradigm" accepted at  ACM Transactions on the Web (TWEB).

--------------------------
## Execution steps
Once you download the Amazon Core dataset (see instructions inside the Dataset folder), you can replicate our resuts as follows.

You can skip steps (1) and (2) by downloading our extracted features. 
Download the resources from: https://forms.gle/imWLi2z63sbgqXU78
Plance both *LIWC* and *Dataset* folders in the main directory. 


### (1) Unzip the datasets
For instance, suppose you aim training a model over *Toys_and_Games* dataset. 
The first step loads the dataset from the tar.gz files, and it converts it into a pandas object. 

    python load_dataset.py -d Toys_and_Games

### (2) Extract the features
This is the most time consuming step. 
It extract a set of features from a pre-processed dataset (output of the previous point). 

    python features_extraction.py -d Toys_and_Games

Note that the code is nothing more than a *handler*, i.e., it contains a bunch of *if-else* conditions. 
Each of this extracts a target set of features, so feel free to comment some branches if you need to focus on a restrict set of features. 

Do you have an unexpected inerruption of your code? 
That's not a problem! Our script incrementally save the extracted features: therefore, you just need to restart and it will automatically resume. 
Inside the script you can further tune the number of cores, based on your machine capabilities. 

### (3) It's time for training
We experimented we many models. 
With the following script, you automatically trains a set of naive models (e.g., Logistic Regression, Random Forest). 

    python train.py -d Toys_and_Games --m lr -c 8

For instance, the previous instruction trains a Logistic Regression over the Toys_and_Games dataset, using 8 cores (to speed up the execution). 




