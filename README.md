Epoch 1/100
52/52 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.0624 - loss: 3.8757
Epoch 1: val_accuracy improved from None to 0.05012, saving model to best_model.keras
52/52 ━━━━━━━━━━━━━━━━━━━━ 177s 3s/step - accuracy: 0.0671 - loss: 3.7179 - val_accuracy: 0.0501 - val_loss: 3.1296
Epoch 2/100
52/52 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.0843 - loss: 3.4457
Epoch 2: val_accuracy did not improve from 0.05012
52/52 ━━━━━━━━━━━━━━━━━━━━ 165s 3s/step - accuracy: 0.1015 - loss: 3.3637 - val_accuracy: 0.0501 - val_loss: 3.1528
Epoch 3/100
52/52 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.1303 - loss: 3.1206
Epoch 3: val_accuracy did not improve from 0.05012
52/52 ━━━━━━━━━━━━━━━━━━━━ 162s 3s/step - accuracy: 0.1207 - loss: 3.1152 - val_accuracy: 0.0501 - val_loss: 3.2221
Epoch 4/100
52/52 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.1490 - loss: 2.9410
Epoch 4: val_accuracy did not improve from 0.05012
52/52 ━━━━━━━━━━━━━━━━━━━━ 143s 3s/step - accuracy: 0.1491 - loss: 2.9258 - val_accuracy: 0.0501 - val_loss: 3.2929
Epoch 5/100
52/52 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.1981 - loss: 2.7766
Epoch 5: val_accuracy did not improve from 0.05012
52/52 ━━━━━━━━━━━━━━━━━━━━ 145s 3s/step - accuracy: 0.1949 - loss: 2.7843 - val_accuracy: 0.0501 - val_loss: 3.5080
Epoch 6/100
52/52 ━━━━━━━━━━━━━━━━━━━━ 0s 3s/step - accuracy: 0.1996 - loss: 2.7586
Epoch 6: val_accuracy did not improve from 0.05012
52/52 ━━━━━━━━━━━━━━━━━━━━ 155s 3s/step - accuracy: 0.2178 - loss: 2.7338 - val_accuracy: 0.0501 - val_loss: 3.5505
Epoch 7/100
...
Epoch 96: val_accuracy did not improve from 0.96897
52/52 ━━━━━━━━━━━━━━━━━━━━ 101s 2s/step - accuracy: 0.8348 - loss: 1.1271 - val_accuracy: 0.9666 - val_loss: 0.4532



==========================
Best Epoch : 76
Best Validation val_accuracy : 0.9690
==========================


==========================
Best Epoch : 96
Best Validation val_loss : 0.4532
==========================


==========================
HASIL EVALUASI MODEL CNN
==========================
Accuracy : 0.9690

Classification Report:

                             precision    recall  f1-score   support

              Faza Humairah       0.95      1.00      0.98        21
               Jabbal Akbar       0.88      1.00      0.93        21
              Muhammad Alif       1.00      0.95      0.98        21
      Muhammad Heikal Fasya       0.95      1.00      0.98        21
Muhammad Rizki Arta Maulana       1.00      0.95      0.98        21
 Muhammad Zahrul Ath Thariq       0.95      1.00      0.98        21
              Nabila Balqis       1.00      1.00      1.00        21
       Natasya Babulrhaiyan       1.00      1.00      1.00        21
        Nur Fadhillah Zulfi       1.00      0.90      0.95        21
             Putri Al Violy       1.00      0.90      0.95        21
        Rahmat Isma Hidayat       1.00      1.00      1.00        21
              Riski Maulani       1.00      0.95      0.97        20
          Salesya Al Fatila       0.91      1.00      0.95        21
         Suci Wildani Rizka       1.00      0.95      0.98        21
             Syariqul Husni       0.95      1.00      0.98        21
...
                  macro avg       0.97      0.97      0.97       419
               weighted avg       0.97      0.97      0.97       419

==========================
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
Epoch 96: early stopping
Restoring model weights from the end of the best epoch: 76.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
