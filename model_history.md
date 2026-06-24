# Model History

## Version 1

Model: best_efficientnet_b0_fruit_combined_balanced.pth

Real-world Accuracy:
82.20%

Correct:
254 / 309

Notes:
Initial EfficientNet-B0 model trained on the balanced fruit dataset.

---

## Version 2

Model: best_efficientnet_b0_fruit_combined_balanced(2).pth

Real-world Accuracy:
84.79%

Correct:
262 / 309

Notes:
Improved data cleaning and dataset balancing.

---

## Version 3

Model: best_efficientnet_b0_fruit_combined_balanced(3).pth

Test Accuracy:
96.83%

Real-world Accuracy:
91.59%

Correct:
283 / 309

Notes:
Added additional real-world strawberry images to improve generalization.
This model achieved the best performance on both the test set and real-world evaluation dataset.

---

## Summary

| Version | Real-World Accuracy |
| ------- | ------------------- |
| V1      | 82.20%              |
| V2      | 84.79%              |
| V3      | 91.59%              |

Best model:
best_efficientnet_b0_fruit_combined_balanced(3).pth
