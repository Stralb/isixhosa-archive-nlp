# Findings

## Sampling changes the representation trade-off

The main comparison changes how the fixed training budget is allocated between the archive's English and isiXhosa text views, independently of how individual pages are selected. We used BPE and Unigram with four policies and five seeds each.

In the 1888 held-out period, word-based allocation reduced English-view BPE segmentation cost by 1.8775 pieces per 100 alphabetic characters (95% conditional issue-bootstrap interval [-1.9716, -1.7818]). Unigram gave -1.7948 ([-1.9081, -1.6706]). The isiXhosa-view BPE cost increased by 0.4526 ([+0.4068, +0.4974]). The results therefore show a trade-off, not uniform improvement.

[Original segmentation comparisons](../results/segmentation.csv).

## Bigram modelling gives a similar but narrower conclusion

We measured held-out bits per alphabetic character with a smoothed count-bigram model. The validation-only comparison of alpha 1, 10 and 100 selected alpha 100.

The isolated allocation contrast for English-view BPE was -0.0372167 BPC in 1888 (95% interval [-0.0423644, -0.0321188]), approximately 1.29% of the PP baseline. All five seeds had the predicted sign. It was -0.0251924 in the 1889 to 1894 check.

But the whole-policy change from PP to WW did **not** consistently help: the later English-view BPC contrast was +0.0045033. This negative result matters because changing allocation and page selection together is not equivalent to isolating allocation.

[Original bigram comparisons](../results/bigram.csv).

## Repeated-text check

We fitted 20 additional BPE models to test whether exact repeated newspaper material explained the results. After filtering repeated 13-word sequences, the 1888 English-view PP minus WW segmentation gap remained +2.3891 pieces per 100 characters (interval [+2.1824, +2.5898]). The unfiltered, matched-support comparison was +2.0974. Shorter or paraphrased repetition could still remain.

[Repetition contrasts](../results/repetition.csv) and [full raw scores](../results/raw/repetition.csv).

## Limits

The data come from one newspaper, and the text-view language assignments were not independently labelled. Intervals use paired publication-issue bootstraps conditional on the available pages and trained seeds. Later years were seen during earlier exploration, so they are not untouched external confirmation. Tokenization compactness and bigram loss are not measurements of fairness or historical culture.

Run `python src/run.py check` to verify the saved model files and recalculate all 64 point effects from the native raw tables.
