# Propensity score
## Logistic regression

```
               n0      n1    target
matching
bipartify  271.01  271.01  0.019609
psmpy      308.90  308.90  0.179349
Bipartify wins 95.0 percent of the times
           count      mean       std  min       25%       50%       75%       max
matching
bipartify  100.0  0.019609  0.015401  0.0  0.007605  0.016758  0.031192  0.075540
psmpy      100.0  0.179349  0.078107  0.0  0.134106  0.197840  0.233777  0.321543
TtestResult(statistic=20.02694032132374, pvalue=1.3534246565591814e-36, df=99)
```

## Random forest

```
               n0      n1    target
matching
bipartify  271.01  271.01  0.019965
psmpy      308.90  308.90  0.181289
Bipartify wins 95.0 percent of the times
           count      mean       std  min       25%       50%       75%       max
matching
bipartify  100.0  0.019965  0.015074  0.0  0.007512  0.018116  0.028046  0.064748
psmpy      100.0  0.181289  0.076752  0.0  0.131122  0.201006  0.242640  0.326220
TtestResult(statistic=20.15720904798034, pvalue=8.072900377574287e-37, df=99)
```

## PsmPy's LR

```
               n0      n1    target
matching
bipartify  271.01  271.01  0.025720
psmpy      308.90  308.90  0.031695
Bipartify wins 56.00000000000001 percent of the times
           count      mean       std       min       25%       50%       75%       max
matching
bipartify  100.0  0.025720  0.016306  0.000000  0.013711  0.023295  0.034927  0.086331
psmpy      100.0  0.031695  0.022693  0.002924  0.012991  0.029033  0.039765  0.092715
TtestResult(statistic=2.3408577816658247, pvalue=0.021244847958455992, df=99)
```

# Propensity logit
## Logistic regression

```
               n0      n1    target
matching
bipartify  271.01  271.01  0.019141
psmpy      308.90  308.90  0.179282
Bipartify wins 96.0 percent of the times
           count      mean       std  min       25%       50%       75%       max
matching
bipartify  100.0  0.019141  0.014435  0.0  0.007605  0.015475  0.030566  0.075540
psmpy      100.0  0.179282  0.077946  0.0  0.138420  0.197438  0.233157  0.321543
TtestResult(statistic=20.130269476655663, pvalue=8.981803839688106e-37, df=99)
```

## Random forest

```
               n0      n1    target
matching
bipartify  271.01  271.01  0.018365
psmpy      308.90  308.90  0.180431
Bipartify wins 96.0 percent of the times
           count      mean       std       min       25%       50%       75%       max
matching
bipartify  100.0  0.018365  0.013621  0.000000  0.007477  0.014902  0.027662  0.064748
psmpy      100.0  0.180431  0.077364  0.003356  0.131858  0.190185  0.233043  0.313665
TtestResult(statistic=20.03940942853258, pvalue=1.2880003154958874e-36, df=99)

```

## PsmPy's LR

```
               n0      n1    target
matching
bipartify  271.01  271.01  0.032731
psmpy      308.90  308.90  0.031695
Bipartify wins 52.0 percent of the times
           count      mean       std       min       25%       50%       75%       max
matching
bipartify  100.0  0.032731  0.023345  0.000000  0.013348  0.033090  0.046475  0.107914
psmpy      100.0  0.031695  0.022693  0.002924  0.012991  0.029033  0.039765  0.092715
TtestResult(statistic=-0.33750505517817087, pvalue=0.7364507531529914, df=99)
```