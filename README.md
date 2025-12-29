# Value-Guided Decision Transformer: A Unified Reinforcement Learning Framework for Online and Offline Settings
![alt text](image.png)

## Overview

The Conditional Sequence Modeling (CSM) paradigm, benefiting from the transformer's powerful distribution modeling capabilities, has demonstrated considerable promise in Reinforcement Learning (RL) tasks. However, much of the work has focused on applying CSM to single online or offline settings, with the general architecture rarely explored. Additionally, existing methods primarily focus on deterministic trajectory modeling, overlooking the randomness of state transitions and the diversity of future trajectory distributions. Fortunately, value-based methods offer a viable solution for CSM, further bridging the potential gap between offline and online RL. In this paper, we propose Value-Guided Decision Transformer (VDT), which leverages value functions to perform advantage-weighting and behavior regularization on the Decision Transformer (DT), guiding the policy toward upper-bound optimal decisions during the offline training phase. In the online tuning phase, VDT further integrates value-based policy improvement with behavior cloning under the CSM architecture through limited interaction and data collection, achieving performance improvement within minimal timesteps. The predictive capability of value functions for future returns is also incorporated into the sampling process. Our method achieves competitive performance on various standard RL benchmarks, providing a feasible solution for developing CSM architectures in general scenarios.


## Quick Start
When your environment is ready, you could run
``` Bash
python main.py --env-name halfcheetah-medium-replay-v2
```


## 📝 Citation

If you find this work helpful in your research, please consider citing:

```
@inproceedings{zheng2025value,
  title={Value-Guided Decision Transformer: A Unified Reinforcement Learning Framework for Online and Offline Settings},
  author={Zheng, Hongling and Shen, Li and Luo, Yong and Ye, Deheng and Xu, Shuhan and Du, Bo and Shen, Jialie and Tao, Dacheng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}

```