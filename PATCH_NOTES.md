# pitcher-ai-class-006

Adds prediction-normalized confusion matrices to the HTML report.

New charts:

- reports/confusion_matrix_pitch_type_when_predicted.png
- reports/confusion_matrix_pitch_class_when_predicted.png

How to read them:

- Rows are what the model predicted.
- Columns are what actually happened.
- Each row sums to 100%.

These answer questions such as: when the model predicts Fastball, what did the pitch actually turn out to be?
