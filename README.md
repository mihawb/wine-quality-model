# White wine quality regression model

University project for Data Exploration class at Warsaw University of Technology. Dataset was analysed with LASSO and Ridge, and multiple regression algorithms, as well as out own implementation of gradient descent. Full analysis with descriptions and conclusion can be found in [analysis/dataset-analysis-final.ipynb](https://github.com/mihawb/wine-quality-model/blob/main/analysis/dataset-analysis-final.ipynb). 

### Authors
* [Michał Banaszczak](https://github.com/mihawb)
* [Patryk Chojnicki](https://github.com/Selthen)

### How to use utils directory
Since python by default only searches script's entry-point directory for imports, add this snippet to your notebook to import utilities
```
import sys
sys.path.append('../utils')
from module_name import *
```