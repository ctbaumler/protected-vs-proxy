# The Impact of Explanations on Fairness in Human-AI Decision Making: Protected vs Proxy Features

By: [Navita Goyal](https://navitagoyal.github.io/)* `<navita@umd.edu>`, [Connor Baumler](https://ctbaumler.github.io/)* `<baumler@umd.edu>`, Tin Nguyen, and Hal Daumé III
(* Equal contribution)

```
@inproceedings{10.1145/3640543.3645210,
  author = {Goyal, Navita and Baumler, Connor and Nguyen, Tin and Daum\'{e} III, Hal},
  title = {The Impact of Explanations on Fairness in Human-AI Decision-Making: Protected vs Proxy Features},
  year = {2024},
  isbn = {9798400705083},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3640543.3645210},
  doi = {10.1145/3640543.3645210},
  booktitle = {Proceedings of the 29th International Conference on Intelligent User Interfaces},
  pages = {155–180},
  numpages = {26},
  keywords = {explanations, fairness, human-AI decision-making, indirect biases},
  location = {, Greenville, SC, USA, },
  series = {IUI '24}
}
```

## Usage
Data can be downloaded via [Kaggle](https://www.kaggle.com/datasets/yousuf28/prosper-loan). 

`prosper.py` includes all code for adding the synthetic protected/proxy features to the Prosper data, training each model, generating the explanations, and calculating dataset statistics (e.g., correlation between features). Please be sure to change the path to the Prosper dataset as needed.
