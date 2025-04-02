# RQ1: Can input perturbations in the prompt lead LLMs to generate unsafe responses that they would normally reject?

-   everyone in previous work said perturbation created unsafe response. but behaviour raise due to perturbation is more nuanced.

## 1. Observed slight increase in safety.

|                           | Safe (%) | Unsafe (%) |
| :------------------------ | -------: | ---------: |
| original_response_safety  |    64.00 |      36.00 |
| perturbed_response_safety |    67.93 |      32.07 |

-   for llama safety increased from 85 to 90%. and for mistral to grow from 53 to 63%.
-   for `Original Response Safety`
    | model | Safe (%) | Unsafe (%) |
    |:--------|-----------:|-------------:|
    | llama2 | 85.64 | 14.36 |
    | llama3 | 49.45 | 50.55 |
    | llama31 | 67.09 | 32.91 |
    | mistral | 53.82 | 46.18 |

-   for `Perturbed Response Safety`
    | model | Safe (%) | Unsafe (%) |
    |:---------|-----------:|-------------:|
    | llama2 | 90.07 | 9.93 |
    | llama3 | 49.43 | 50.57 |
    | llaama31 | 68.73 | 31.27 |
    | mistral | 63.49 | 36.51 |

## 2. Overall Flip Percentages:

    -   safe_to_unsafe - 13.908358
    -   unsafe_to_safe - 17.837243
    -   safe_to_safe - 50.091642
    -   unsafe_to_unsafe - 18.162757
