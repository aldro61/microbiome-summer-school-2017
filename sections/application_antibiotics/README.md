# Application: predicting antibiotic resistance

In this part of the tutorial, we will model the resistance of 353 ***Clostridium difficile*** isolates to clindamycin, a lincosamide antibiotic. We will use the Set Covering Machine algorithm (Marchand and Shawe-Taylor, 2002), which produces sparse interpretable models, and a Support Vector Machine, which produces a *black-box* model.

To apply the Set Covering Machine algorithm, we will use [Kover](https://github.com/aldro61/kover/) (Drouin et al., 2016) a disk-based implementation of this algorithm designed to learn from large genomic datasets. Kover uses reference-free genome comparisons, based on k-mers to learn sparse and interpretable models of phenotypes. The models produced by Kover make predictions based on the presence/absence of k-mers. 

## Data

To use Kover, the data must be packaged into a specical, compressed, format, which relies on the [HDF5 library](https://support.hdfgroup.org/HDF5/whatishdf5.html). A Kover dataset can easily be created from reads, contigs or a k-mer matirx ([see the documentation](http://aldro61.github.io/kover/doc_dataset.html)). In the interest of time, this was done for you.

### Verify the data
First, verify that the Kover dataset was successfully downloaded by running the following command.

```bash
make download.verify
```

You should get the following output:

```
f0b7bcf295f370da3440198ada98e104  ./data/antibiotics/cdiff_clindamycin.kover
```

### Exploring the data

The dataset was created from the whole genomes of 353 *C. difficile* isolates, which were partitioned into k-mers using [Ray Surveyor](https://github.com/zorino/raysurveyor-tutorial).

Run the following command to print the number of k-mers in the dataset:

```bash
kover dataset info --dataset ./data/antibiotics/cdiff_clindamycin.kover --kmer-count
```

Run the following command to print the number of isolates in the dataset:

```bash
kover dataset info --dataset ./data/antibiotics/cdiff_clindamycin.kover --genome-count
```

Notice the enourmous imbalance between the number of learning examples (353) and the number of features (32 823 803). This setting is very challenging for learning algorithms and is know as **fat data**.

### Create a training and testing set

Kover takes care of running correct machine learning protocols for you. It offers a command called `kover dataset split`, which allows to partition a dataset into training and testing sets, as well as cross-validation folds.

Create a data partition using the following command:

```bash
kover dataset split --dataset ./data/antibiotics/cdiff_clindamycin.kover --id tutorial_split --train-size 0.666 --folds 5 --random-seed 42 --progress
```

You can use the following command to list the splits in a dataset:

```bash
kover dataset info --dataset ./data/antibiotics/cdiff_clindamycin.kover --splits
```

This should print the split that you just created.


## Learning a model

Now that we have created a partition of our data, we are ready to learn a model. Run the following command to start learning:

```bash
kover learn --dataset ./data/antibiotics/cdiff_clindamycin.kover --split tutorial_split --model-type conjunction disjunction --p 0.1 1.0 10.0 --max-rules 5 --hp-choice cv --n-cpu 2 --progress
```

This will perform cross-validation to select the value of the SCM's hyperparameters ([see here](http://aldro61.github.io/kover/doc_learning.html#understanding-the-hyperparameters)), train using the best hyperparameter values, evaluate the model on the testing set, and create a report once it is done.