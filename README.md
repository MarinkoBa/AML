# AML

#### The goal is to build a binary/multi class classificator, which will be able to distinguish between COVID-19, Pneumonia and Normal classes of X-ray chest images. Due to the current COVID-19 situation the radiologists could need help while recognizing positive COVID-19 cases.

#### Dataset
For training of the network we used the Chest X-ray dataset from kaggle (https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia). Dataset is organized into 2 folders (train, test) and both train and test contain 3 subfolders (COVID19, PNEUMONIA, NORMAL). DataSet contains total 6432 x-ray images and test data have 20% of total images.

#### Chest X Ray images data directory structure:
	- data 
		- test
			- COVID19
       				- *.jpg
			- NORMAL
       				- *.jpg
			- PNEUMONIA
       				- *.jpg
		- train
			- COVID19
       				- *.jpg
			- NORMAL
       				- *.jpg
			- PNEUMONIA
       				- *.jpg


#### Method

To Achieve the objective, the follwoing three neural network approaches are used:

- DarkCovidNet by Ozturk (2020): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7187882/pdf/main.pdf
- DenseNet by Huang (2016): https://arxiv.org/abs/1608.06993
- Xception by Jain (2020): https://link.springer.com/content/pdf/10.1007/s10489-020-01902-1.pdf

#### Results

![plot](plots/BestArchitecturesComparison_20_font_cut.png)

- The best DarkCovidNet was trained on a balanced dataset with augmentation and a learning rate of 0.003.
- The best DenseNet was trained on a balanced dataset without augmentation and a learning rate of 0.001.
- The best Xception Network was trained on a balanced dataset with augmentation and a learning rate of 0.001.

In case of the evaluation metrics like Precision, Recall, F1-Score and Accuracy, the trained models achieved following results:

![plot](Metriken.PNG)

#### Done

- Preprocessing including reshaping
- Implementation Xception network
- Implementation DarkCovid network
- First Training/Performance test with unbalanced data set
- Implementation of 2D DenseNet
- Implementation of Metrics (Recall, Secificity, Precision, F1-Score, Accuracy)
- Data Augmentation: Random rotation angle  = 10°, Horisontal Flip and Zoom Range = 0.4
- Balanced Dataset: ~500x Covid, ~500x Normal, ~500x Pneunomia (randomly picked from the sets) and increase dataset with augmentation to 1500x Covid, 1500x Normal, 1500x Pneunomia. The original train set includes 460 Covid (300 RGB, 124 L, 36 RGBA -> for differences see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes), 1266 Normal and 3418 Pneunomia images.
- Train different DenseNet Architecutres based on balanced without augmentation dataset and learning_rate = 0.001 (-> choose best)
- Train DarkCovidNet to show which dataset is most suitable (no_balancing_no_aug, balancing_no_aug, balancing_and_aug)
- Train Xception based on best dataset and with different learning rates (evaluate best one)

#### FUTURE WORK
- GAN for higher resolution	
- build Ensemble of the Models and test performance
