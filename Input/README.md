### Input info

The time series used in this work is a collection of tension and current of the three phases of the lathes' motor. The time series were recorded from operations in a real machining system at the UFJF's Manufacturing Processes Laboratory in collaboration with the Laboratory of Industrial Automation and Computational Intelligence (LAIIC). This procedure used in this work is regulated by ISO 3685/1993. It consisted of executing successive machining operations and examining the regular intervals of the tool wear condition. This process is repeated until the tool wear reaches a pre-established limit. Considering the technical guidelines of ISO 3685/1993, the adopted limit was the flank wear's maximum length of 0.6 mm. Thus, the flank wears greater than the established limit was considered an inadequate condition. Furthermore, the machining conditions were: depth of cut of 0.5 mm, cutting speed of 120 m/min, and feed rate of 0.156 mm/rev.


 - All input files consists in 592 measurements.
 - Each measurement contain 6 time series (Voltage A, Voltage B, Voltage C, Current A, Current B, and Current C)
 - Each time series has 750 samples.
 - 352 of those measurements consists in the adequate condition tool and the other 240 were the inadequate condition tool.
   - Target = 0: Adequate Condition
   - Target = 1: Inadequate Condition


The dataset follows the following table:


| ID  | Time_ID | Voltage A | Voltage B | Voltage C | Current A | Current B | Current C | Target |
|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|--------|
| 1   | 1       |  1.91     | -0.74     | -0.50     | -0.52     |  0.43     | -1.03     |      1 |
| 1   | 2       |  1.06     | -1.76     | -0.37     | -1.12     |  0.45     |  1.17     |      1 |
| ... | ...     | ...       | ...       | ...       | ...       | ...       | ...       | ...    |
| 1   | 750     |  0.48     | -0.22     | -0.24     | -0.33     |  0.95     | -1.69     |      1 |
| 2   | 1       |  0.78     | -0.22     | -0.36     |  0.16     | -0.06     |  1.45     |      1 |
| 2   | 2       | -0.21     | -0.74     | -0.24     | -1.18     |  1.09     |  0.46     |      1 |
| ... | ...     | ...       | ...       | ...       | ...       | ...       | ...       | ...    |
| 2   | 750     | -1.18     |  0.79     |  0.24     |  1.65     | -0.09     | -0.09     |      1 |
| ... | ...     | ...       | ...       | ...       | ...       | ...       | ...       | ...    |
| 592 | 1       | -0.85     |  0.28     |  2.81     | -0.74     |  2.15     |  0.07     |      0 |
| 592 | 2       | -1.18     |  0.79     | -0.37     |  0.56     | -1.21     |  0.64     |      0 |
| ... | ...     | ...       | ...       | ...       | ...       | ...       | ...       | ...    |
| 592 | 750     | -0.83     |  1.82     | -0.56     |  1.52     |  0.60     | -0.97     |      0 |


There are seven datasets in this repository:

 - Input_1.csv - Original acquisition;
 - Input_2.csv - Original acquisition with  1dB AWGN;
 - Input_3.csv - Original acquisition with  3dB AWGN;
 - Input_4.csv - Original acquisition with  5dB AWGN;
 - Input_5.csv - Original acquisition with  7dB AWGN;
 - Input_6.csv - Original acquisition with 10dB AWGN.
