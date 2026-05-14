d:\FaceAttend_VC\venv\lib\site-packages\keras\src\layers\convolutional\base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
Epoch 1/30
53/53 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.2099 - loss: 2.7149
Epoch 1: val_accuracy improved from None to 0.05012, saving model to best_model.keras
53/53 ━━━━━━━━━━━━━━━━━━━━ 81s 1s/step - accuracy: 0.3260 - loss: 2.3886 - val_accuracy: 0.0501 - val_loss: 3.0030 - learning_rate: 5.0000e-04
Epoch 2/30
53/53 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.5338 - loss: 1.6958
Epoch 2: val_accuracy did not improve from 0.05012
53/53 ━━━━━━━━━━━━━━━━━━━━ 71s 1s/step - accuracy: 0.5724 - loss: 1.5732 - val_accuracy: 0.0501 - val_loss: 3.1526 - learning_rate: 5.0000e-04
Epoch 3/30
53/53 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.6333 - loss: 1.3018
Epoch 3: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.

Epoch 3: val_accuracy did not improve from 0.05012
53/53 ━━━━━━━━━━━━━━━━━━━━ 71s 1s/step - accuracy: 0.6591 - loss: 1.2060 - val_accuracy: 0.0501 - val_loss: 3.4721 - learning_rate: 5.0000e-04
Epoch 4/30
53/53 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.7426 - loss: 0.9697
Epoch 4: val_accuracy did not improve from 0.05012
53/53 ━━━━━━━━━━━━━━━━━━━━ 73s 1s/step - accuracy: 0.7392 - loss: 0.9633 - val_accuracy: 0.0501 - val_loss: 4.0762 - learning_rate: 2.5000e-04
Epoch 5/30
53/53 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.7861 - loss: 0.8641
Epoch 5: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.

Epoch 5: val_accuracy did not improve from 0.05012
53/53 ━━━━━━━━━━━━━━━━━━━━ 70s 1s/step - accuracy: 0.7913 - loss: 0.8444 - val_accuracy: 0.0501 - val_loss: 4.4815 - learning_rate: 2.5000e-04
...
53/53 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.8971 - loss: 0.4311
Epoch 30: val_accuracy did not improve from 0.94272
53/53 ━━━━━━━━━━━━━━━━━━━━ 60s 1s/step - accuracy: 0.9031 - loss: 0.4176 - val_accuracy: 0.9379 - val_loss: 0.3715 - learning_rate: 3.1250e-05
Restoring model weights from the end of the best epoch: 29.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
