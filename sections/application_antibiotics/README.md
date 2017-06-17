# Application: predicting antibiotic resistance

In this part of the tutorial, we will use two machine learning algorithms to predict antibiotic resistance in ***Clostridium difficile*** isolates. Specifically, we will use the whole genomes of **353 isolates** along with their measured susceptibility to **clindamycin**, a lincosamide antibiotic.

## Data

First, verify that the data was successfully downloaded by running the following command.

```bash
make download.verify
```

You should get the following output:

```
f0b7bcf295f370da3440198ada98e104  ./data/antibiotics/cdiff_clindamycin.kover
```
