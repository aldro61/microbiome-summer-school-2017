<a href="../../#table-of-contents">&larr; Back to table of contents</a>

# Application: predicting antibiotic resistance

In this part of the tutorial, we will model the resistance of 141 [***Mycobacterium tuberculosis***](https://en.wikipedia.org/wiki/Mycobacterium_tuberculosis) isolates to the [rifampicin](https://en.wikipedia.org/wiki/Rifampicin) antibiotic. We will use the Set Covering Machine algorithm (Marchand and Shawe-Taylor, 2002), which produces sparse interpretable models, and a Support Vector Machine, which produces a *black-box* model.

To apply the Set Covering Machine algorithm, we will use [Kover](https://github.com/aldro61/kover/) (Drouin et al., 2016) a disk-based implementation of this algorithm designed to learn from large genomic datasets. Kover uses reference-free genome comparisons, based on k-mers, to learn sparse and interpretable models of phenotypes. The models produced by Kover make predictions based on the presence/absence of k-mers. 

## Data

Download and uncompress the data using the following command:

```bash
make applications.antibiotics.data
```

Now, move to the data directory using the following command:

```bash
cd kover-example
```

## Kover example

Once this is done, follow [the example](http://aldro61.github.io/kover/doc_example.html) given in the Kover documentation, but skip the part about downloading the data.

## Comparison to SVM

Use the `cd ..` command to go back to the `exercise` directory. Then, run the following command to train a Support Vector Machine on this dataset and compare it to Kover.

```bash
make applications.antibiotics.svm
```

![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Exercise:** Which of the models is the most accurate (SVM or Kover)? Can you guess why?

**Solution:** [click me](./solutions/why_scm_better_svm/)


## References

Drouin, A., Giguère, S., Déraspe, M., Marchand, M., Tyers, M., Loo, V. G., ... & Corbeil, J. (2016). Predictive computational phenotyping and biomarker discovery using reference-free genome comparisons. BMC genomics, 17(1), 754. [[link]](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2889-6)


<br />

<a href="../../#table-of-contents">&larr; Back to table of contents</a>
