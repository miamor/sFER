MAHNOB: 13-703

```
python3 bi_lstm.py \
    -d MAHNOB \
train \
    -t 3 \
    -m checkpoints/MAHNOB_frames_50_3/bi_lstm-14_loss-0.23068_val_loss-0.17997.h5
```
```
python3 bi_lstm.py \
    -d MAHNOB \
train \
    -c CART
```


```
python3 bi_lstm.py \
    -d UvA \
eval \
    -c NN \
    -m checkpoints/MAHNOB_frames_50_3/bi_lstm-04_loss-0.20451_val_loss-0.16932.h5
```
```
python3 bi_lstm.py \
    -d UvA \
eval \
    -c CART \
    -m output/classifier_MAHNOB_CART.pkl
```

checkpoints/MAHNOB_frames_50/bi_lstm-08_loss-0.15145_val_loss-0.19306.h5
    + MAHNOB Test:  96.77%
    + MMI:          96.05%
    + SPOS:         53.12%
    + UvA:          66.66% (2/3)    [0 0 0] - [0 0 1]
    + CK+:          95.65%
    + AFEW:         57.61%
    
    + MAHNOB:
              precision    recall  f1-score   support

           0       0.95      0.98      0.96        42
           1       0.98      0.96      0.97        51

   micro avg       0.97      0.97      0.97        93
   macro avg       0.97      0.97      0.97        93
weighted avg       0.97      0.97      0.97        93

    + SPOS:
              precision    recall  f1-score   support

           0       0.16      0.39      0.23        28
           1       0.81      0.56      0.66       132

   micro avg       0.53      0.53      0.53       160
   macro avg       0.49      0.48      0.45       160
weighted avg       0.70      0.53      0.59       160

    + MMI: 
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        76
           1       0.00      0.00      0.00         0

   micro avg       0.96      0.96      0.96        76
   macro avg       0.50      0.48      0.49        76
weighted avg       1.00      0.96      0.98        76

    + CK+:
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        69
           1       0.00      0.00      0.00         0

   micro avg       0.96      0.96      0.96        69
   macro avg       0.50      0.48      0.49        69
weighted avg       1.00      0.96      0.98        69

    + AFEW: 
              precision    recall  f1-score   support

           0       0.56      0.96      0.71        49
           1       0.75      0.14      0.24        43

   micro avg       0.58      0.58      0.58        92
   macro avg       0.65      0.55      0.47        92
weighted avg       0.65      0.58      0.49        92





checkpoints/MAHNOB_frames_50_3/bi_lstm-02_loss-0.19902_val_loss-0.25891.h5
    + MAHNOB Test:  95%
    + MMI:          88.16%
    + SPOS:         70.62%
    + AFEW:         58.70%
    + UvA:          66.66% (2/3)    [0 0 0] - [0 0 1]

checkpoints/MAHNOB_frames_50_3/bi_lstm-16_loss-0.20491_val_loss-0.26126.h5
    + MAHNOB Test:  95%
    + MMI:          86.84%
    + SPOS:         70.00%
    + AFEW:         59.78%
    + UvA:          66.66% (2/3)    [0 0 0] - [0 0 1]



checkpoints/MAHNOB_frames_50_3_/bi_lstm-14_loss-0.23068_val_loss-0.17997.h5
    + MAHNOB Test:  100% (96.77%)
    + MMI:          97.37%
    + SPOS:         82.50%
    + AFEW:         58.70%
    + UvA:          33.33% (1/3)    [1 0 0] - [0 0 1]
    + CK+:          97.10%


    + MAHNOB:       96.77%              
              precision    recall  f1-score   support

           0       0.95      0.98      0.96        42
           1       0.98      0.96      0.97        51

   micro avg       0.97      0.97      0.97        93
   macro avg       0.97      0.97      0.97        93
weighted avg       0.97      0.97      0.97        93

    + MMI:          97.37%              
              precision    recall  f1-score   support

           0       1.00      0.97      0.99        76
           1       0.00      0.00      0.00         0

   micro avg       0.97      0.97      0.97        76
   macro avg       0.50      0.49      0.49        76
weighted avg       1.00      0.97      0.99        76

    + SPOS:         82.50%              
              precision    recall  f1-score   support

           0       0.50      0.14      0.22        28
           1       0.84      0.97      0.90       132

   micro avg       0.82      0.82      0.82       160
   macro avg       0.67      0.56      0.56       160
weighted avg       0.78      0.82      0.78       160

    CK+:            97.10%              
              precision    recall  f1-score   support

           0       1.00      0.97      0.99        69
           1       0.00      0.00      0.00         0

   micro avg       0.97      0.97      0.97        69
   macro avg       0.50      0.49      0.49        69
weighted avg       1.00      0.97      0.99        69

    AFEW:           58.70%              
              precision    recall  f1-score   support

           0       0.58      0.84      0.68        49
           1       0.62      0.30      0.41        43

   micro avg       0.59      0.59      0.59        92
   macro avg       0.60      0.57      0.54        92
weighted avg       0.60      0.59      0.55        92




output/classifier_MAHNOB_CART.pkl
    + UvA:          100% (2/3)    [[1 0] [1 0] [0 1]]

    + MMI:          100%              
                  precision    recall  f1-score   support

           0       1.00      1.00      1.00        76
           1       0.00      0.00      0.00         0

   micro avg       1.00      1.00      1.00        76
   macro avg       0.50      0.50      0.50        76
weighted avg       1.00      1.00      1.00        76
 samples avg       1.00      1.00      1.00        76

    + SPOS:         55%              
                  precision    recall  f1-score   support

           0       0.20      0.54      0.29        28
           1       0.85      0.55      0.67       132

   micro avg       0.55      0.55      0.55       160
   macro avg       0.53      0.54      0.48       160
weighted avg       0.74      0.55      0.60       160
 samples avg       0.55      0.55      0.55       160

    + MAHNOB Test:  95.70%              
              precision    recall  f1-score   support

           0       0.93      0.98      0.95        42
           1       0.98      0.94      0.96        51

   micro avg       0.96      0.96      0.96        93
   macro avg       0.96      0.96      0.96        93
weighted avg       0.96      0.96      0.96        93
 samples avg       0.96      0.96      0.96        93

    + CK+:          98.55%              
                  precision    recall  f1-score   support

           0       1.00      0.99      0.99        69
           1       0.00      0.00      0.00         0

   micro avg       0.99      0.99      0.99        69
   macro avg       0.50      0.49      0.50        69
weighted avg       1.00      0.99      0.99        69
 samples avg       0.99      0.99      0.99        69

    + AFEW:         58.69%              
              precision    recall  f1-score   support

           0       0.60      0.69      0.64        49
           1       0.57      0.47      0.51        43

   micro avg       0.59      0.59      0.59        92
   macro avg       0.58      0.58      0.58        92
weighted avg       0.58      0.59      0.58        92
 samples avg       0.59      0.59      0.59        92



output/classifier_MAHNOB_SVM.pkl
    MAHNOB:         72.04%
              precision    recall  f1-score   support

           0       0.62      0.95      0.75        42
           1       0.93      0.53      0.67        51

   micro avg       0.72      0.72      0.72        93
   macro avg       0.78      0.74      0.71        93
weighted avg       0.79      0.72      0.71        93

    + MMI:          93.42%
              precision    recall  f1-score   support

           0       1.00      0.93      0.97        76
           1       0.00      0.00      0.00         0

   micro avg       0.93      0.93      0.93        76
   macro avg       0.50      0.47      0.48        76
weighted avg       1.00      0.93      0.97        76

    + SPOS:         18.13%
              precision    recall  f1-score   support

           0       0.15      0.82      0.26        28
           1       0.55      0.05      0.08       132

   micro avg       0.18      0.18      0.18       160
   macro avg       0.35      0.43      0.17       160
weighted avg       0.48      0.18      0.11       160

    + CK+:          91.30%
              precision    recall  f1-score   support

           0       1.00      0.91      0.95        69
           1       0.00      0.00      0.00         0

   micro avg       0.91      0.91      0.91        69
   macro avg       0.50      0.46      0.48        69
weighted avg       1.00      0.91      0.95        69

    + AFEW:         54.35%
              precision    recall  f1-score   support

           0       0.54      1.00      0.70        49
           1       1.00      0.02      0.05        43

   micro avg       0.54      0.54      0.54        92
   macro avg       0.77      0.51      0.37        92
weighted avg       0.75      0.54      0.39        92


checkpoints/MAHNOB_frames_50_/bi_lstm-16_loss-0.12913_val_loss-0.20115.h5
    + MAHNOB Test:  85%
    + MMI:          98.68%
    + SPOS:         56.88%
    + UvA:          66.66% (2/3)    [0 0 0]

checkpoints/MAHNOB_frames_50/bi_lstm-06_loss-0.19239_val_loss-0.20530.h5
    + MAHNOB Test:  100%
    + MMI:          93.42%
    + SPOS:         59.38%
    + UvA:          66.66% (2/3)    [0 0 0]



checkpoints/MAHNOB_frames_50_4__14.05/19__loss-0.13128__val_loss-0.27536__precision-0.96526__val_precision-0.86842__recall-0.91745__val_recall-0.80488.h5
    + MMI:          acc: 83.87%
                    precision: 80.00%
                    recall: 85.71%
              precision    recall  f1-score   support

           0       1.00      0.38      0.55        76
           1       0.00      0.00      0.00         0

   micro avg       0.38      0.38      0.38        76
   macro avg       0.50      0.19      0.28        76
weighted avg       1.00      0.38      0.55        76

    + MAHNOB:
                  precision    recall  f1-score   support

           0       0.80      0.86      0.83        42
           1       0.88      0.82      0.85        51

   micro avg       0.84      0.84      0.84        93
   macro avg       0.84      0.84      0.84        93
weighted avg       0.84      0.84      0.84        93




checkpoints/MAHNOB_frames_50_4__14.05_bb/17__loss-0.16__val_loss-0.33__acc-0.93__val_acc-0.89.h5
    + MAHNOB:       84.95%              
              precision    recall  f1-score   support

           0       0.82      0.86      0.84        42
           1       0.88      0.84      0.86        51

   micro avg       0.85      0.85      0.85        93
   macro avg       0.85      0.85      0.85        93
weighted avg       0.85      0.85      0.85        93

    + MMI:          30.26%              
              precision    recall  f1-score   support

           0       1.00      0.30      0.46        76
           1       0.00      0.00      0.00         0

   micro avg       0.30      0.30      0.30        76
   macro avg       0.50      0.15      0.23        76
weighted avg       1.00      0.30      0.46        76

    + SPOS:         61.88               
              precision    recall  f1-score   support

           0       0.16      0.29      0.21        28
           1       0.82      0.69      0.75       132

   micro avg       0.62      0.62      0.62       160
   macro avg       0.49      0.49      0.48       160
weighted avg       0.70      0.62      0.65       160

    + CK+:          23.19%              
              precision    recall  f1-score   support

           0       1.00      0.23      0.38        69
           1       0.00      0.00      0.00         0

   micro avg       0.23      0.23      0.23        69
   macro avg       0.50      0.12      0.19        69
weighted avg       1.00      0.23      0.38        69

    + AFEW:         64.13%              
              precision    recall  f1-score   support

           0       0.61      0.92      0.73        49
           1       0.78      0.33      0.46        43

   micro avg       0.64      0.64      0.64        92
   macro avg       0.69      0.62      0.60        92
weighted avg       0.69      0.64      0.60        92




**checkpoints/MAHNOB_frames_50_0__14.05/35__vloss-0.26__vacc-0.90__vprecision-0.83__vrecall-0.95.h5**
    + MAHNOB:       94.62%              
              precision    recall  f1-score   support

           0       0.95      0.93      0.94        42
           1       0.94      0.96      0.95        51

   micro avg       0.95      0.95      0.95        93
   macro avg       0.95      0.94      0.95        93
weighted avg       0.95      0.95      0.95        93

    + MMI:          34.21%              
              precision    recall  f1-score   support

           0       1.00      0.34      0.51        76
           1       0.00      0.00      0.00         0

   micro avg       0.34      0.34      0.34        76
   macro avg       0.50      0.17      0.25        76
weighted avg       1.00      0.34      0.51        76

    + SPOS:         65%                 
              precision    recall  f1-score   support

           0       0.24      0.46      0.32        28
           1       0.86      0.69      0.76       132

   micro avg       0.65      0.65      0.65       160
   macro avg       0.55      0.58      0.54       160
weighted avg       0.75      0.65      0.69       160

    + CK+:          30.43%              
              precision    recall  f1-score   support

           0       1.00      0.30      0.47        69
           1       0.00      0.00      0.00         0

   micro avg       0.30      0.30      0.30        69
   macro avg       0.50      0.15      0.23        69
weighted avg       1.00      0.30      0.47        69

    + AFEW:         53.26%              
              precision    recall  f1-score   support

           0       0.59      0.41      0.48        49
           1       0.50      0.67      0.57        43

   micro avg       0.53      0.53      0.53        92
   macro avg       0.54      0.54      0.53        92
weighted avg       0.55      0.53      0.53        92



**Hide mouth**
CART:
    + MAHNOB:       67.43%
              precision    recall  f1-score   support

           0       0.16      0.67      0.25        51
           1       0.96      0.67      0.79       563

   micro avg       0.67      0.67      0.67       614
   macro avg       0.56      0.67      0.52       614
weighted avg       0.89      0.67      0.75       614
 samples avg       0.67      0.67      0.67       614


checkpoints/MAHNOB_frames_50_3_/bi_lstm-14_loss-0.23068_val_loss-0.17997.h5
    + MAHNOB:       70.85%
              precision    recall  f1-score   support

           0       0.20      0.80      0.31        51
           1       0.98      0.70      0.81       563

   micro avg       0.71      0.71      0.71       614
   macro avg       0.59      0.75      0.56       614
weighted avg       0.91      0.71      0.77       614


checkpoints/MAHNOB_frames_50/bi_lstm-08_loss-0.15145_val_loss-0.19306.h5
    + MAHNOB:       57.65%
              precision    recall  f1-score   support

           0       0.14      0.78      0.24        51
           1       0.97      0.56      0.71       563

   micro avg       0.58      0.58      0.58       614
   macro avg       0.55      0.67      0.47       614
weighted avg       0.90      0.58      0.67       614
